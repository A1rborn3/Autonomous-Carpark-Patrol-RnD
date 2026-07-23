import logging
import os

import cv2
import numpy as np
from skimage import morphology


class RoadGraphExtractor:
    def __init__(self, min_lane_width=2.5, pixels_per_meter=70, kernel_size=5):
        self.min_lane_width = min_lane_width
        self.pixels_per_meter = pixels_per_meter
        self.kernel_size = kernel_size
        self.threshold_m = self.min_lane_width / 2.0

    def extract_graph(self, occupancy, output_dir, blue_mask=None):
        logging.info("Extracting road graph from occupancy map...")

        _, occ_bin = cv2.threshold(occupancy, 254, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size))
        occ_clean = cv2.morphologyEx(occ_bin, cv2.MORPH_OPEN, kernel)
        occ_clean = cv2.morphologyEx(occ_clean, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite(os.path.join(output_dir, "occupancy_clean.png"), cv2.bitwise_not(occ_clean))

        free_space = cv2.bitwise_not(occ_clean)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(free_space, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            flood_filled = (labels == largest_label).astype(np.uint8) * 255
        else:
            flood_filled = free_space

        dist_transform = cv2.distanceTransform(free_space, cv2.DIST_L2, 5)
        pixel_threshold = self.threshold_m * self.pixels_per_meter
        dist_mask = (dist_transform >= pixel_threshold).astype(np.uint8) * 255
        aisle_mask = cv2.bitwise_and(flood_filled, dist_mask)
        cv2.imwrite(os.path.join(output_dir, "aisle_mask.png"), aisle_mask)

        try:
            skeleton = cv2.ximgproc.thinning(aisle_mask)
        except AttributeError:
            logging.warning("cv2.ximgproc.thinning is not available; falling back to skimage")
            try:
                skeleton_bool = morphology.skeletonize(aisle_mask > 0, method='zhang')
            except TypeError:
                skeleton_bool = morphology.skeletonize(aisle_mask > 0)
            skeleton = (skeleton_bool * 255).astype(np.uint8)

        cv2.imwrite(os.path.join(output_dir, "skeleton.png"), skeleton)

        skeleton_mask = (skeleton // 255).astype(np.uint8)
        nodes, edges = self._build_graph_from_skeleton(skeleton_mask)

        graph_vis = cv2.cvtColor(occ_clean, cv2.COLOR_GRAY2BGR)
        for edge in edges:
            n1 = next((n for n in nodes if n['id'] == edge['from_id']), None)
            n2 = next((n for n in nodes if n['id'] == edge['to_id']), None)
            if n1 and n2:
                cv2.line(graph_vis, (int(n1['x']), int(n1['y'])), (int(n2['x']), int(n2['y'])), (0, 0, 255), 2)
        for node in nodes:
            cv2.circle(graph_vis, (int(node['x']), int(node['y'])), 4, (255, 0, 0), -1)
        cv2.imwrite(os.path.join(output_dir, "final_graph.png"), graph_vis)

        return nodes, edges

    def _build_graph_from_skeleton(self, skeleton_mask):
        coords = self._extract_points_from_skeleton(skeleton_mask)
        if not coords:
            return [], []

        coord_set = set(coords)
        adjacency = {}
        for coord in coord_set:
            x, y = coord
            neighbors = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    candidate = (x + dx, y + dy)
                    if candidate in coord_set:
                        neighbors.append(candidate)
            adjacency[coord] = neighbors

        important_points = [coord for coord in coord_set if len(adjacency[coord]) != 2]
        if not important_points:
            important_points = list(coord_set)

        node_lookup = {}
        nodes = []
        edges = []
        edge_set = set()

        def add_node(coord):
            if coord not in node_lookup:
                node_lookup[coord] = len(nodes)
                nodes.append({'id': f'node_{node_lookup[coord]}', 'x': float(coord[0]), 'y': float(coord[1]), 'type': 'waypoint'})
            return node_lookup[coord]

        def add_edge(u, v):
            if u == v:
                return
            pair = tuple(sorted((u, v)))
            if pair not in edge_set:
                edge_set.add(pair)
                edges.append({'id': f'edge_{len(edges)}', 'from_id': nodes[u]['id'], 'to_id': nodes[v]['id']})

        for start in important_points:
            for neighbor in adjacency[start]:
                path = [start, neighbor]
                prev = start
                curr = neighbor
                while True:
                    if curr in important_points and curr != start:
                        break

                    next_candidates = [n for n in adjacency[curr] if n != prev]
                    if not next_candidates:
                        break
                    if len(next_candidates) > 1 and curr not in important_points:
                        break

                    prev, curr = curr, next_candidates[0]
                    path.append(curr)

                if len(path) >= 2:
                    simplified = self._simplify_path(path, max_offset_px=10.0)
                    if len(simplified) >= 2:
                        node_ids = [add_node(point) for point in simplified]
                        for i in range(len(node_ids) - 1):
                            add_edge(node_ids[i], node_ids[i + 1])

        return nodes, edges

    def _extract_points_from_skeleton(self, skeleton_mask):
        rows, cols = np.where(skeleton_mask > 0)
        if len(cols) == 0:
            return []
        return [(int(col), int(row)) for row, col in zip(rows, cols)]

    def _simplify_path(self, path, max_offset_px=10.0):
        if len(path) <= 3:
            return path

        simplified = list(path)
        changed = True
        while changed:
            changed = False
            for i in range(1, len(simplified) - 1):
                a = simplified[i - 1]
                b = simplified[i]
                c = simplified[i + 1]
                if self._is_collinear(a, b, c, max_offset_px):
                    simplified.pop(i)
                    changed = True
                    break
        return simplified

    def _is_collinear(self, a, b, c, max_offset_px):
        x1, y1 = a
        x2, y2 = b
        x3, y3 = c
        cross = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
        return cross <= max_offset_px

    def _prune_spurs(self, nodes, edges, dist_transform, entrance_centers, blue_mask=None):
           return nodes, edges #THIS IS JUST FOR TESTING TO SKIP MODULE, REMOVE FOR DEPLOYMENT
           # Build adjacency list
           adj = {n['id']: set() for n in nodes}
           for e in edges:
               adj[e['from_id']].add(e['to_id'])
               adj[e['to_id']].add(e['from_id'])
               
           node_dict = {n['id']: n for n in nodes}
           removed_nodes = set()
           
           while True:
               # Find all current endpoints
               degree1 = [n_id for n_id, nbrs in adj.items() if len(nbrs) == 1]
               pruned_any = False
               
               for leaf in degree1:
                   curr = leaf
                   prev = None
                   branch_len = 0
                   path_nodes = [curr]
                   
                   # Walk the branch until we hit an intersection (degree > 2)
                   while True:
                       nbrs = list(adj[curr])
                       # Break if we hit a branch point (degree >= 3)
                       if len(nbrs) >= 3:
                           break
                       # Break if we hit another endpoint (and it's not the start node)
                       if len(nbrs) == 1 and curr != leaf:
                           break
                       # Break if isolated
                       if len(nbrs) == 0:
                           break 
                       
                       # Get the next node that isn't the previous one
                       next_node = nbrs[0] if nbrs[0] != prev else (nbrs[1] if len(nbrs) > 1 else None)
                       if next_node is None:
                           break
                           
                       # Calculate distance
                       n1, n2 = node_dict[curr], node_dict[next_node]
                       dist = np.hypot(n1['x'] - n2['x'], n1['y'] - n2['y'])
                       branch_len += dist
                       
                       prev = curr
                       curr = next_node
                       path_nodes.append(curr)
                           
                   # Get the branch point (intersection)
                   bp = node_dict[curr]
                   
                   # We need R for multiple checks
                   bp_y = int(np.clip(bp['y'], 0, dist_transform.shape[0]-1))
                   bp_x = int(np.clip(bp['x'], 0, dist_transform.shape[1]-1))
                   R = float(dist_transform[bp_y, bp_x])
                   if R < 1.0:
                       R = (self.min_lane_width * self.pixels_per_meter) / 2.0
                       
                   is_protected = False
                   
                   # --- Manual Entrance Pinning ---
                   # Check if this leaf node is near the CENTROID of any painted entrance.
                   leaf_node = node_dict[leaf]
                   for center in entrance_centers:
                       dist_to_center = np.hypot(leaf_node['x'] - center['x'], leaf_node['y'] - center['y'])
                       # 4.2 meters tolerance from the exact center of the blue mass
                       if dist_to_center < 4.2 * self.pixels_per_meter: 
                           is_protected = True
                           break
                   
                   # --- Angle Protection ---
                   nbrs = list(adj[curr])
                   if len(nbrs) >= 3:
                       import math
                       vectors = {}
                       for nbr_id in nbrs:
                           nbr_node = node_dict[nbr_id]
                           vx = nbr_node['x'] - bp['x']
                           vy = nbr_node['y'] - bp['y']
                           mag = math.hypot(vx, vy)
                           if mag > 0:
                               vectors[nbr_id] = (vx/mag, vy/mag)
                           else:
                               vectors[nbr_id] = (0, 0)
                               
                       min_dot = 1.0
                       best_pair = None
                       nbr_ids = list(vectors.keys())
                       for i in range(len(nbr_ids)):
                           for j in range(i+1, len(nbr_ids)):
                               id1, id2 = nbr_ids[i], nbr_ids[j]
                               v1, v2 = vectors[id1], vectors[id2]
                               dot = v1[0]*v2[0] + v1[1]*v2[1]
                               if dot < min_dot:
                                   min_dot = dot
                                   best_pair = (id1, id2)
                                   
                       if best_pair and min_dot < -0.866:
                           if prev in best_pair:
                               is_protected = True
                               
                   if is_protected:
                       continue # Skip pruning
                   
                   # --- Length Pruning ---
                   local_max_spur_len = 2.0 * R
                   
                   if branch_len < local_max_spur_len:
                       for n_id in path_nodes[:-1]:
                           removed_nodes.add(n_id)
                           for neighbor in list(adj[n_id]):
                               adj[neighbor].discard(n_id)
                           adj[n_id].clear()
                       pruned_any = True
                       
               if not pruned_any:
                   break
                   
           # Filter the final nodes and edges
           final_nodes = [n for n in nodes if n['id'] not in removed_nodes]
           final_edges = [e for e in edges if e['from_id'] not in removed_nodes and e['to_id'] not in removed_nodes]
           
           # Recompute types
           degree = {n['id']: 0 for n in final_nodes}
           for e in final_edges:
               degree[e['from_id']] += 1
               degree[e['to_id']] += 1
               
           # Completely remove any nodes that were left totally isolated by the pruning
           final_nodes = [n for n in final_nodes if degree[n['id']] > 0]
               
           # Default all to waypoints
           for n in final_nodes:
               n['type'] = 'waypoint'
               
           # Tag ONLY the single closest node for each manually painted entrance center
           if entrance_centers:
               node_coords = np.array([[n['x'], n['y']] for n in final_nodes])
               node_tree = KDTree(node_coords)
               for center in entrance_centers:
                   dist, idx = node_tree.query([center['x'], center['y']])
                   if dist < 10.0 * self.pixels_per_meter:
                       final_nodes[idx]['type'] = 'entrance_exit'
   
           # FINAL STEP: Re-index everything so IDs are sequential and clean for the USER
           # This resolves the confusion where IDs suggest more nodes than actually exist.
           id_map = {}
           for i, node in enumerate(final_nodes):
               old_id = node['id']
               new_id = f"node_{i}"
               node['id'] = new_id
               id_map[old_id] = new_id
               
           for i, edge in enumerate(final_edges):
               edge['id'] = f"edge_{i}"
               edge['from_id'] = id_map[edge['from_id']]
               edge['to_id'] = id_map[edge['to_id']]
               # Remove internal indices as they are no longer accurate/needed
               if 'from_idx' in edge: del edge['from_idx']
               if 'to_idx' in edge: del edge['to_idx']
                       
           return final_nodes, final_edges


   #possibly new one??

   def _prune_spurs(self, nodes, edges, dist_transform, entrance_centers, blue_mask=None):
           """
           Aggressively and repeatably prunes terminating spurs while protecting 
           the main road centerline and marked entrances.
           """
           if not nodes or not edges:
               return nodes, edges
               
           node_dict = {n['id']: n for n in nodes}
           
           # Build adjacency list
           adj = {n['id']: set() for n in nodes}
           for e in edges:
               adj[e['from_id']].add(e['to_id'])
               adj[e['to_id']].add(e['from_id'])
               
           # Identify protected nodes (entrance/exits)
           protected_nodes = set()
           if entrance_centers:
               node_coords = np.array([[n['x'], n['y']] for n in nodes])
               node_tree = KDTree(node_coords)
               for center in entrance_centers:
                   dist, idx = node_tree.query([center['x'], center['y']])
                   if dist < 15.0:  # ~0.2m tolerance
                       protected_nodes.add(nodes[idx]['id'])
                       
           for n in nodes:
               if n.get('type') == 'entrance_exit':
                   protected_nodes.add(n['id'])
   
           # Aggressive spur length threshold: 1x min_lane_width in pixels
           max_spur_length = 1.0 * self.min_lane_width * self.pixels_per_meter
           
           changed = True
           iteration = 0
           while changed and iteration < 100:
               changed = False
               iteration += 1
               
               # Safety break: don't prune if the graph is already tiny
               if len(adj) < 4:
                   break
                   
               # Find all current leaf nodes (degree == 1)
               leaves = [n_id for n_id, neighbors in adj.items() if len(neighbors) == 1]
               
               for leaf_id in leaves:
                   if leaf_id in protected_nodes:
                       continue
                   if leaf_id not in adj:
                       continue
                       
                   curr = leaf_id
                   prev = None
                   path_nodes = [curr]
                   path_length = 0.0
                   is_valid_spur = False
                   
                   while True:
                       neighbors = list(adj[curr])
                       next_nodes = [n for n in neighbors if n != prev]
                       
                       if len(next_nodes) == 0:
                           break
                           
                       next_node = next_nodes[0]
                       n1 = node_dict[curr]
                       n2 = node_dict[next_node]
                       dist = np.hypot(n1['x'] - n2['x'], n1['y'] - n2['y'])
                       path_length += dist
                       
                       prev = curr
                       curr = next_node
                       path_nodes.append(curr)
                       
                       # Stop if we hit a junction (degree >= 3)
                       if len(adj[curr]) >= 3:
                           is_valid_spur = True
                           break
                       # Stop if we hit a protected node
                       if curr in protected_nodes:
                           is_valid_spur = True
                           break
                       # If we hit another leaf, this is a standalone segment
                       if len(adj[curr]) == 1 and curr != leaf_id:
                           if path_length < 50.0: # Prune if extremely short
                               is_valid_spur = True
                           else:
                               is_valid_spur = False # Keep reasonable standalone roads
                           break
                           
                   if is_valid_spur and path_length < max_spur_length:
                       changed = True
                       # Remove all nodes in the path EXCEPT the last one (the junction/protected node)
                       nodes_to_remove = path_nodes[:-1]
                       for n_id in nodes_to_remove:
                           if n_id in adj:
                               for neighbor in list(adj[n_id]):
                                   adj[neighbor].discard(n_id)
                               del adj[n_id]
                               
           # Rebuild final nodes and edges based on the pruned adjacency list
           valid_nodes = set(adj.keys())
           final_nodes = [n for n in nodes if n['id'] in valid_nodes]
           
           final_edges = []
           edge_id = 0
           seen_edges = set()
           for e in edges:
               if e['from_id'] in valid_nodes and e['to_id'] in valid_nodes:
                   pair = frozenset([e['from_id'], e['to_id']])
                   if pair not in seen_edges:
                       seen_edges.add(pair)
                       final_edges.append({
                           'id': f"edge_{edge_id}",
                           'from_id': e['from_id'],
                           'to_id': e['to_id']
                       })
                       edge_id += 1
                       
           # Re-index for clean output
           id_map = {}
           for i, node in enumerate(final_nodes):
               old_id = node['id']
               new_id = f"node_{i}"
               node['id'] = new_id
               id_map[old_id] = new_id
               
           reindexed_edges = []
           for i, edge in enumerate(final_edges):
               from_id = id_map.get(edge['from_id'])
               to_id = id_map.get(edge['to_id'])
               if from_id is not None and to_id is not None and from_id != to_id:
                   reindexed_edges.append({
                       'id': f"edge_{i}",
                       'from_id': from_id,
                       'to_id': to_id
                   })
                   
           return final_nodes, reindexed_edges