import numpy as np
import cv2
import logging
import os
from skimage import morphology
from scipy.spatial import KDTree

class RoadGraphExtractor:
    def __init__(self, min_lane_width=2.5, pixels_per_meter=70, kernel_size=5):
        self.min_lane_width = min_lane_width
        self.pixels_per_meter = pixels_per_meter
        self.kernel_size = kernel_size
        self.threshold_m = self.min_lane_width / 2.0

    def extract_graph(self, occupancy, output_dir, blue_mask=None):
        logging.info("Extracting road graph from occupancy map...")
        
        # 1. Preprocessing (invert so obstacles=255, free space=0)
        # We use a threshold of 254 so that ONLY pure white (255) is walkable.
        # Everything else (127 grey, 0 black, or user-painted areas) becomes an obstacle (255).
        _, occ_bin = cv2.threshold(occupancy, 254, 255, cv2.THRESH_BINARY_INV)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size))
        
        # 1.5. Noise Removal
        # Apply morphological OPENING (erosion followed by dilation) to remove isolated outlier pixels.
        # If an obstacle is smaller than the kernel size (e.g., 5x5 pixels), it gets erased.
        occ_clean = cv2.morphologyEx(occ_bin, cv2.MORPH_OPEN, kernel)
        
        # Apply morphological CLOSING to fill small holes inside actual obstacles
        occ_clean = cv2.morphologyEx(occ_clean, cv2.MORPH_CLOSE, kernel)
        
        cv2.imwrite(os.path.join(output_dir, "occupancy_clean.png"), cv2.bitwise_not(occ_clean))
        
        # 2. Aisle Network Segmentation
        free_space = cv2.bitwise_not(occ_clean)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(free_space, connectivity=8)
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
        
        # 3. Skeletonization
        try:
            skeleton = cv2.ximgproc.thinning(aisle_mask)
        except AttributeError:
            logging.warning("cv2.ximgproc.thinning is not available (opencv-contrib-python might not be installed). "
                            "Falling back to skimage.morphology.skeletonize(method='zhang') for consistent graph extraction.")
            try:
                skeleton_bool = morphology.skeletonize(aisle_mask > 0, method='zhang')
            except TypeError:
                # Fallback for very old scikit-image versions that don't support method='zhang'
                skeleton_bool = morphology.skeletonize(aisle_mask > 0)
            skeleton = (skeleton_bool * 255).astype(np.uint8)
            
        cv2.imwrite(os.path.join(output_dir, "skeleton.png"), skeleton)
        
        # 4. Graph Extraction
        logging.info("Detecting nodes and edges...")
        
        skel_norm = (skeleton // 255).astype(np.uint8)
        kernel_3x3 = np.array([[1, 1, 1],
                               [1, 10, 1],
                               [1, 1, 1]], dtype=np.uint8)
        
        neighbor_count = cv2.filter2D(skel_norm, -1, kernel_3x3, borderType=cv2.BORDER_CONSTANT)
        branch_nodes_mask = ((neighbor_count >= 13) & (skel_norm == 1))
        
        segments_mask = skel_norm.copy()
        segments_mask[branch_nodes_mask] = 0
        
        num_skel_labels, skel_labels = cv2.connectedComponents(segments_mask, connectivity=8)
        
        raw_lines = []
        for label in range(1, num_skel_labels):
            y_coords, x_coords = np.where(skel_labels == label)
            pts = np.column_stack((x_coords, y_coords))
            
            ordered_pts = self._chain_points(pts)
            if len(ordered_pts) > 1:
                ordered_pts_np = np.array(ordered_pts, dtype=np.int32).reshape((-1, 1, 2))
                # Use a larger epsilon to enforce organized, straight lines rather than jagged pixel paths
                approx = cv2.approxPolyDP(ordered_pts_np, epsilon=0.5 * self.pixels_per_meter, closed=False)
                line = [{'x': float(pt[0][0]), 'y': float(pt[0][1])} for pt in approx]
                raw_lines.append(line)
                
        refined_nodes, edges = self._build_topology(raw_lines, branch_nodes_mask, occ_clean)
        
        # Find the center of each manually painted blue entrance marker
        entrance_centers = []
        if blue_mask is not None and cv2.countNonZero(blue_mask) > 0:
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(blue_mask, connectivity=8)
            for i in range(1, num_labels):
                cx, cy = centroids[i]
                entrance_centers.append({'x': cx, 'y': cy})
                
        # Prune corner spurs and dead-ends
        refined_nodes, edges = self._prune_spurs(refined_nodes, edges, dist_transform, entrance_centers, blue_mask)

        
        # 5. Visual Debugging Output
        # Draw the final mathematical graph onto the occupancy map so we can see the pruned/straightened results
        graph_vis = cv2.cvtColor(occ_clean, cv2.COLOR_GRAY2BGR)
        
        # Draw edges (Red)
        for e in edges:
            n1 = next((n for n in refined_nodes if n['id'] == e['from_id']), None)
            n2 = next((n for n in refined_nodes if n['id'] == e['to_id']), None)
            if n1 and n2:
                cv2.line(graph_vis, (int(n1['x']), int(n1['y'])), (int(n2['x']), int(n2['y'])), (0, 0, 255), 2)
                
        # Draw nodes (Blue for waypoints, Green for entrance/exit)
        for n in refined_nodes:
            color = (0, 255, 0) if n.get('type') == 'entrance_exit' else (255, 0, 0)
            cv2.circle(graph_vis, (int(n['x']), int(n['y'])), 4, color, -1)
            
        cv2.imwrite(os.path.join(output_dir, "final_graph.png"), graph_vis)
        
        return refined_nodes, edges
        
    def _build_topology(self, raw_lines, branch_nodes_mask, occ_clean):
        nodes_pool = []
        segment_point_indices = []

        for line in raw_lines:
            if len(line) >= 2:
                start_idx = len(nodes_pool)
                nodes_pool.append({'x': float(line[0]['x']), 'y': float(line[0]['y'])})
                end_idx = len(nodes_pool)
                nodes_pool.append({'x': float(line[-1]['x']), 'y': float(line[-1]['y'])})
                segment_point_indices.append((start_idx, end_idx))
            elif line:
                point_idx = len(nodes_pool)
                nodes_pool.append({'x': float(line[0]['x']), 'y': float(line[0]['y'])})
                segment_point_indices.append((point_idx, point_idx))

        branch_y, branch_x = np.where(branch_nodes_mask)
        branch_pts = [{'x': float(x), 'y': float(y)} for x, y in zip(branch_x, branch_y)]
        nodes_pool.extend(branch_pts)

        merged_nodes = []
        node_mapping = {}
        if len(nodes_pool) > 0:
            raw_coords = np.array([[n['x'], n['y']] for n in nodes_pool], dtype=float)
            tree = KDTree(raw_coords)
            processed = set()
            for i in range(len(nodes_pool)):
                if i in processed:
                    continue

                indices = tree.query_ball_point(raw_coords[i], 2.0)
                mx = float(np.mean([raw_coords[idx][0] for idx in indices]))
                my = float(np.mean([raw_coords[idx][1] for idx in indices]))
                merged_idx = len(merged_nodes)
                merged_nodes.append({'x': mx, 'y': my, 'id': f"node_{merged_idx}"})
                for idx in indices:
                    node_mapping[idx] = merged_idx
                    processed.add(idx)

        refined_nodes = self._refine_node_positions(merged_nodes, occ_clean)
        for i, node in enumerate(refined_nodes):
            node['id'] = f"node_{i}"

        edges = []
        edge_set = set()
        edge_id = 0
        for start_idx, end_idx in segment_point_indices:
            u = node_mapping[start_idx]
            v = node_mapping[end_idx]
            if u != v:
                pair = frozenset([u, v])
                if pair not in edge_set:
                    edge_set.add(pair)
                    edges.append({
                        'id': f"edge_{edge_id}",
                        'from_id': f"node_{u}",
                        'to_id': f"node_{v}",
                        'from_idx': u,
                        'to_idx': v
                    })
                    edge_id += 1

        return refined_nodes, edges

    def _chain_points(self, pts):
        if len(pts) == 0: return []
        
        tree = KDTree(pts)
        pairs = tree.query_pairs(1.5)
        
        adj = {i: [] for i in range(len(pts))}
        for i, j in pairs:
            adj[i].append(j)
            adj[j].append(i)
            
        start_idx = 0
        for i in range(len(pts)):
            if len(adj[i]) <= 1:
                start_idx = i
                break
                
        visited = set()
        curr = start_idx
        ordered = []
        while curr is not None:
            visited.add(curr)
            ordered.append(pts[curr])
            next_node = None
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    next_node = neighbor
                    break
            curr = next_node
        return ordered

    def _refine_node_positions(self, nodes, occ_clean):
        obs_y, obs_x = np.where(occ_clean == 255)
        obs_points = np.column_stack((obs_x, obs_y))
        
        if len(obs_points) == 0:
            return nodes
            
        tree = KDTree(obs_points)
        refined = []
        
        for node in nodes:
            nx, ny = node['x'], node['y']
            distances, indices = tree.query([nx, ny], k=min(100, len(obs_points)))
            
            if len(indices) < 2:
                refined.append(node)
                continue
                
            p1 = obs_points[indices[0]]
            v1 = p1 - np.array([nx, ny])
            norm1 = np.linalg.norm(v1)
            if norm1 == 0:
                refined.append(node)
                continue
                
            v1_dir = v1 / norm1
            p2 = None
            
            for idx in indices[1:]:
                p_cand = obs_points[idx]
                v_cand = p_cand - np.array([nx, ny])
                norm_cand = np.linalg.norm(v_cand)
                if norm_cand == 0: continue
                
                v_cand_dir = v_cand / norm_cand
                dot = np.dot(v1_dir, v_cand_dir)
                angle = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
                
                if angle >= 135:
                    p2 = p_cand
                    break
                    
            if p2 is not None:
                mid_x = (p1[0] + p2[0]) / 2.0
                mid_y = (p1[1] + p2[1]) / 2.0
                refined.append({'x': mid_x, 'y': mid_y})
            else:
                refined.append(node)
                
        return refined
        
    def _build_edges(self, nodes):
        if len(nodes) < 2: return []
        
        node_coords = np.array([[n['x'], n['y']] for n in nodes])
        tree = KDTree(node_coords)
        
        edge_set = set()
        edges = []
        edge_id = 0
        
        for i in range(len(nodes)):
            k = min(3, len(nodes))
            distances, indices = tree.query(node_coords[i], k=k)
            for j in indices[1:]:
                pair = frozenset([i, j])
                if pair not in edge_set:
                    edge_set.add(pair)
                    edges.append({
                        'id': f'edge_{edge_id}',
                        'from_id': nodes[i]['id'],
                        'to_id': nodes[j]['id'],
                        'from_idx': i,
                        'to_idx': j
                    })
                    edge_id += 1
                    
        adj = {i: set() for i in range(len(nodes))}
        for e in edges:
            adj[e['from_idx']].add(e['to_idx'])
            adj[e['to_idx']].add(e['from_idx'])
            
        triangles = []
        for u in range(len(nodes)):
            for v in adj[u]:
                if v > u:
                    for w in adj[v]:
                        if w > v and u in adj[w]:
                            triangles.append((u, v, w))
                            
        edges_to_remove = set()
        for (u, v, w) in triangles:
            e1, e2, e3 = frozenset([u, v]), frozenset([v, w]), frozenset([w, u])
            d1 = np.linalg.norm(node_coords[u] - node_coords[v])
            d2 = np.linalg.norm(node_coords[v] - node_coords[w])
            d3 = np.linalg.norm(node_coords[w] - node_coords[u])
            
            longest = max(d1, d2, d3)
            if longest == d1: edges_to_remove.add(e1)
            elif longest == d2: edges_to_remove.add(e2)
            else: edges_to_remove.add(e3)
            
        final_edges = []
        degree = {n['id']: 0 for n in nodes}
        
        for e in edges:
            pair = frozenset([e['from_idx'], e['to_idx']])
            if pair not in edges_to_remove:
                final_edges.append(e)
                degree[e['from_id']] += 1
                degree[e['to_id']] += 1
                
        for n in nodes:
            if degree[n['id']] == 1:
                n['type'] = 'entrance_exit'
            else:
                n['type'] = 'waypoint'
                
        return final_edges

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
