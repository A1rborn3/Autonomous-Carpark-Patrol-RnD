import numpy as np
import cv2
import logging
import os
from skimage.morphology import medial_axis
from scipy.spatial import KDTree

class RoadGraphExtractor:
    def __init__(self, min_lane_width=2.5, pixels_per_meter=70, kernel_size=5):
        self.min_lane_width = min_lane_width
        self.pixels_per_meter = pixels_per_meter
        self.kernel_size = kernel_size
        self.threshold_m = self.min_lane_width / 2.0

    def extract_graph(self, occupancy, output_dir, blue_mask=None):
        logging.info("Extracting road graph from occupancy map...")
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Preprocessing: Only pure white (255) is walkable
        _, occ_bin = cv2.threshold(occupancy, 254, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size))
        occ_clean = cv2.morphologyEx(occ_bin, cv2.MORPH_OPEN, kernel)
        occ_clean = cv2.morphologyEx(occ_clean, cv2.MORPH_CLOSE, kernel)
        
        # 2. Extract largest connected component of free space
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(occ_clean, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            free_space = (labels == largest_label).astype(np.uint8) * 255
        else:
            free_space = occ_clean
            
        # 3. Distance transform and masking
        dist_transform = cv2.distanceTransform(free_space, cv2.DIST_L2, 5)
        pixel_threshold = self.threshold_m * self.pixels_per_meter
        dist_mask = (dist_transform >= pixel_threshold).astype(np.uint8) * 255
        aisle_mask = cv2.bitwise_and(free_space, dist_mask)
        
        # 4. Skeletonization using true medial axis transform
        skeleton_bool = medial_axis(aisle_mask > 0)
        skeleton = (skeleton_bool * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, "skeleton.png"), skeleton)
        
        # 5. Graph Extraction
        logging.info("Detecting nodes and edges...")
        skel_norm = (skeleton // 255).astype(np.uint8)
        
        kernel_3x3 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
        neighbor_count = cv2.filter2D(skel_norm, -1, kernel_3x3, borderType=cv2.BORDER_CONSTANT)
        
        junction_mask = (neighbor_count >= 3) & (skel_norm == 1)
        junction_y, junction_x = np.where(junction_mask)
        junctions = list(zip(junction_y, junction_x))
        
        junction_dilated = cv2.dilate(junction_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
        segments_mask = skel_norm.copy()
        segments_mask[junction_dilated > 0] = 0
        
        num_skel_labels, skel_labels = cv2.connectedComponents(segments_mask, connectivity=8)
        
        raw_segments = []
        for label in range(1, num_skel_labels):
            y_coords, x_coords = np.where(skel_labels == label)
            pts = np.column_stack((x_coords, y_coords))
            if len(pts) < 2:
                continue
            
            ordered_pts = self._chain_points(pts)
            if len(ordered_pts) >= 2:
                pts_np = np.array(ordered_pts, dtype=np.int32).reshape((-1, 1, 2))
                epsilon = 1.5 * self.pixels_per_meter
                approx = cv2.approxPolyDP(pts_np, epsilon, closed=False)
                line = [{'x': float(pt[0][0]), 'y': float(pt[0][1])} for pt in approx]
                raw_segments.append(line)
                
        # 6. Cluster candidate nodes
        candidate_nodes = []
        for y, x in junctions:
            candidate_nodes.append({'x': float(x), 'y': float(y), 'type': 'junction'})
            
        for line in raw_segments:
            for i, pt in enumerate(line):
                ptype = 'segment_end' if (i == 0 or i == len(line) - 1) else 'turn'
                candidate_nodes.append({'x': pt['x'], 'y': pt['y'], 'type': ptype})
                
        coords = np.array([[n['x'], n['y']] for n in candidate_nodes])
        tree = KDTree(coords)
        merge_radius = 10.0
        
        merged_nodes = []
        node_mapping = {}
        processed = set()
        
        for i in range(len(candidate_nodes)):
            if i in processed:
                continue
            indices = tree.query_ball_point(coords[i], merge_radius)
            valid_indices = [idx for idx in indices if idx not in processed]
            if not valid_indices:
                continue
                
            mx = float(np.mean([coords[idx][0] for idx in valid_indices]))
            my = float(np.mean([coords[idx][1] for idx in valid_indices]))
            
            types = [candidate_nodes[idx]['type'] for idx in valid_indices]
            if 'junction' in types:
                ntype = 'junction'
            elif 'turn' in types:
                ntype = 'turn'
            else:
                ntype = 'segment_end'
                
            merged_idx = len(merged_nodes)
            merged_nodes.append({'x': mx, 'y': my, 'type': ntype, 'id': f"node_{merged_idx}"})
            
            for idx in valid_indices:
                node_mapping[idx] = merged_idx
                processed.add(idx)
                
        # 7. Build initial topology
        edges = []
        edge_set = set()
        edge_id = 0
        
        for line in raw_segments:
            if len(line) < 2:
                continue
            mapped_line = []
            for pt in line:
                dist, idx = tree.query([pt['x'], pt['y']])
                mapped_line.append(node_mapping[idx])
                
            clean_mapped_line = []
            for node_idx in mapped_line:
                if not clean_mapped_line or clean_mapped_line[-1] != node_idx:
                    clean_mapped_line.append(node_idx)
                    
            for i in range(len(clean_mapped_line) - 1):
                u = clean_mapped_line[i]
                v = clean_mapped_line[i+1]
                if u != v:
                    pair = frozenset([u, v])
                    if pair not in edge_set:
                        edge_set.add(pair)
                        edges.append({
                            'id': f"edge_{edge_id}",
                            'from_id': f"node_{u}",
                            'to_id': f"node_{v}"
                        })
                        edge_id += 1
                        
        # 8. Aggressive Simplification: Remove collinear degree-2 nodes
        changed = True
        while changed:
            changed = False
            adj = {n['id']: [] for n in merged_nodes}
            for e in edges:
                adj[e['from_id']].append(e['to_id'])
                adj[e['to_id']].append(e['from_id'])
                
            node_dict = {n['id']: n for n in merged_nodes}
            nodes_to_remove = []
            
            for node in merged_nodes:
                if node['type'] in ['junction', 'entrance_exit']:
                    continue
                if len(adj[node['id']]) == 2:
                    n1_id, n2_id = adj[node['id']]
                    n1 = node_dict[n1_id]
                    n2 = node_dict[n2_id]
                    if self._is_collinear(n1, node, n2, threshold_cos=-0.866):
                        nodes_to_remove.append(node['id'])
                        
            if nodes_to_remove:
                changed = True
                for n_id in nodes_to_remove:
                    n1_id, n2_id = adj[n_id]
                    edges = [e for e in edges if not (e['from_id'] == n_id or e['to_id'] == n_id)]
                    if n1_id != n2_id:
                        pair = frozenset([n1_id, n2_id])
                        if not any(frozenset([e['from_id'], e['to_id']]) == pair for e in edges):
                            edges.append({
                                'id': f"edge_{len(edges)}",
                                'from_id': n1_id,
                                'to_id': n2_id
                            })
                merged_nodes = [n for n in merged_nodes if n['id'] not in nodes_to_remove]
                
        # 9. Identify entrance centers BEFORE pruning so they can be protected
        entrance_centers = []
        if blue_mask is not None and cv2.countNonZero(blue_mask) > 0:
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(blue_mask, connectivity=8)
            for i in range(1, num_labels):
                cx, cy = centroids[i]
                entrance_centers.append({'x': float(cx), 'y': float(cy)})
                
        # 10. Aggressive Spur Pruning (Fixed to catch short terminating nodes)
        merged_nodes, edges = self._prune_spurs(merged_nodes, edges, dist_transform, entrance_centers, blue_mask)

        # 11. Visual Debugging Output
        graph_vis = cv2.cvtColor(occ_clean, cv2.COLOR_GRAY2BGR)
        
        for e in edges:
            n1 = next((n for n in merged_nodes if n['id'] == e['from_id']), None)
            n2 = next((n for n in merged_nodes if n['id'] == e['to_id']), None)
            if n1 and n2:
                cv2.line(graph_vis, (int(n1['x']), int(n1['y'])), (int(n2['x']), int(n2['y'])), (0, 0, 255), 2)
                
        for n in merged_nodes:
            if n.get('type') == 'entrance_exit':
                color = (0, 255, 0)      # Green
            elif n.get('type') == 'junction':
                color = (255, 0, 255)    # Magenta
            else:
                color = (255, 0, 0)      # Blue
            cv2.circle(graph_vis, (int(n['x']), int(n['y'])), 4, color, -1)
            
        cv2.imwrite(os.path.join(output_dir, "final_graph.png"), graph_vis)
        logging.info(f"Graph extraction complete. Nodes: {len(merged_nodes)}, Edges: {len(edges)}")
        
        return merged_nodes, edges

    def _prune_spurs(self, nodes, edges, dist_transform, entrance_centers, blue_mask=None):
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
                    if len(nbrs) >= 3:
                        break
                    if len(nbrs) == 1 and curr != leaf:
                        break
                    if len(nbrs) == 0:
                        break 
                    
                    next_node = nbrs[0] if nbrs[0] != prev else (nbrs[1] if len(nbrs) > 1 else None)
                    if next_node is None:
                        break
                        
                    n1, n2 = node_dict[curr], node_dict[next_node]
                    dist = np.hypot(n1['x'] - n2['x'], n1['y'] - n2['y'])
                    branch_len += dist
                    
                    prev = curr
                    curr = next_node
                    path_nodes.append(curr)
                        
                # Get the branch point (intersection)
                bp = node_dict[curr]
                
                bp_y = int(np.clip(bp['y'], 0, dist_transform.shape[0]-1))
                bp_x = int(np.clip(bp['x'], 0, dist_transform.shape[1]-1))
                R = float(dist_transform[bp_y, bp_x])
                if R < 1.0:
                    R = (self.min_lane_width * self.pixels_per_meter) / 2.0
                    
                is_protected = False
                
                # --- Manual Entrance Pinning ---
                leaf_node = node_dict[leaf]
                for center in entrance_centers:
                    dist_to_center = np.hypot(leaf_node['x'] - center['x'], leaf_node['y'] - center['y'])
                    if dist_to_center < 4.2 * self.pixels_per_meter: 
                        is_protected = True
                        break
                
                # --- Angle Protection ---
                nbrs = list(adj[curr])
                if len(nbrs) >= 3:
                    vectors = {}
                    for nbr_id in nbrs:
                        nbr_node = node_dict[nbr_id]
                        vx = nbr_node['x'] - bp['x']
                        vy = nbr_node['y'] - bp['y']
                        mag = np.hypot(vx, vy)
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
                
                # --- FIXED Length Pruning ---
                # In narrow corridors, R is small, making 2.0*R too small to catch short artifacts.
                # We enforce an absolute minimum threshold (1.0 lane width) to aggressively 
                # clean up short terminating stubs, while still respecting R in wide areas.
                absolute_min_spur_threshold = 1.0 * self.min_lane_width * self.pixels_per_meter
                local_max_spur_len = max(absolute_min_spur_threshold, 2.0 * R)
                
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
        
        # FINAL SURGICAL CLEANUP: Remove any remaining degree-1 nodes connected by a tiny edge
        # This catches artifacts that slipped through the branch walking logic
        changed = True
        while changed:
            changed = False
            adj = {n['id']: set() for n in final_nodes}
            for e in final_edges:
                adj[e['from_id']].add(e['to_id'])
                adj[e['to_id']].add(e['from_id'])
            
            node_dict = {n['id']: n for n in final_nodes}
            to_remove = []
            
            for n_id, neighbors in adj.items():
                if len(neighbors) == 1 and node_dict[n_id].get('type') != 'entrance_exit':
                    neighbor_id = list(neighbors)[0]
                    # If the neighbor is also a degree-1 node, it's an isolated segment, keep it
                    if len(adj[neighbor_id]) == 1:
                        continue 
                    
                    n1 = node_dict[n_id]
                    n2 = node_dict[neighbor_id]
                    dist = np.hypot(n1['x'] - n2['x'], n1['y'] - n2['y'])
                    
                    # If the terminating edge is shorter than 0.5 lane widths, prune it
                    if dist < (0.5 * self.min_lane_width * self.pixels_per_meter):
                        to_remove.append(n_id)
            
            if to_remove:
                changed = True
                for n_id in to_remove:
                    final_nodes = [n for n in final_nodes if n['id'] != n_id]
                    final_edges = [e for e in final_edges if e['from_id'] != n_id and e['to_id'] != n_id]

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

        # FINAL STEP: Re-index everything so IDs are sequential and clean
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
            if 'from_idx' in edge: del edge['from_idx']
            if 'to_idx' in edge: del edge['to_idx']
                    
        return final_nodes, final_edges

    def _chain_points(self, pts):
        """Orders unordered skeleton pixels into a continuous path."""
        if len(pts) == 0: return []
        if len(pts) == 1: return [pts[0]]
        
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

    def _is_collinear(self, A, B, C, threshold_cos=-0.866):
        """Checks if point B is nearly collinear with A and C (<= 30 degree deviation)."""
        v1 = np.array([A['x'] - B['x'], A['y'] - B['y']])
        v2 = np.array([C['x'] - B['x'], C['y'] - B['y']])
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return True
            
        dot = np.dot(v1, v2) / (norm1 * norm2)
        return dot < threshold_cos