import os
import argparse
from pathlib import Path
from dataclasses import dataclass
import json
import math
import cv2
import numpy as np

@dataclass       
class Node:
    node_id: int
    x: float
    y: float
    edges: list

@dataclass
class Edges:
    id: int
    from_id: int
    to_id: int

class GraphProcessor:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)

    def find_edges_per_node(self):
        automated_graphs = sorted((self.output_dir / "Automated_Output").glob("*_graph.json"))
        manual_graphs = sorted((self.output_dir / "Manual_Output").glob("*.json"))
        graph_path = automated_graphs[0] if automated_graphs else (manual_graphs[0] if manual_graphs else None)

        if graph_path is None:
            print(
                "Graph file not found in "
                f"{self.output_dir / 'Automated_Output'} or "
                f"{self.output_dir / 'Manual_Output'}"
            )
            return

        print(f"Found: {graph_path}")
            
        with open(graph_path, 'r') as file:
            data = json.load(file)
            
        all_edges = []
        for edges_data in data["edges"]:
            edge_id = int(edges_data['id'].split('_')[1])
            from_id = int(edges_data['from_id'].split('_')[1])
            to_id = int(edges_data['to_id'].split('_')[1])
            
            edge_instance = Edges(id=edge_id, from_id=from_id, to_id=to_id)
            all_edges.append(edge_instance)

        if all_edges:
            print(f"Total Edges: {len(all_edges)}")
        else:
            print("No edges found in the JSON file.")
            return

        all_nodes = []
        for node_data in data["nodes"]:
            node_id = int(node_data['id'].split('_')[1]) 
            x = float(node_data['x'])
            y = float(node_data['y'])
            node_edges = [] 
            
            for edge in all_edges:
                if edge.from_id == node_id:
                    node_edges.append(edge.id)
                if edge.to_id == node_id:
                    node_edges.append(edge.id)        
                    
            node_instance = Node(node_id=node_id, x=x, y=y, edges=node_edges)
            all_nodes.append(node_instance)

        if all_nodes:
            print(f"Total nodes: {len(all_nodes)}")
        else:
            print("No nodes found in the JSON file.")
            return

        # Trigger the new boundary generation function
        self.generate_road_boundaries(all_nodes, all_edges)


    def generate_road_boundaries(self, all_nodes, all_edges):
        print("\n--- Generating Road Boundaries ---")
        
        #Get metadata for coordinate conversion (meters <-> pixels)
        metadata_path = self.output_dir / "map_metadata.json"
        pixels_per_meter = 70.0
        min_x, max_x, min_y, max_y = 0.0, 0.0, 0.0, 0.0
        
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    meta = json.load(f)
                    pixels_per_meter = float(meta.get('resolution', 70.0))
                    bounds = meta.get('bounds', {})
                    min_x = float(bounds.get('min_x', 0.0))
                    max_x = float(bounds.get('max_x', 0.0))
                    min_y = float(bounds.get('min_y', 0.0))
                    max_y = float(bounds.get('max_y', 0.0))
            except Exception as e:
                print(f"Warning: Could not parse metadata: {e}")
                
        #Load occupancy image and compute Distance Transform
        occ_path = self.output_dir / "occupancy_clean.png"
        if not occ_path.exists():
            occ_path = self.output_dir / "merged_occupancy.png"
            
        occupancy_img = cv2.imread(str(occ_path), cv2.IMREAD_GRAYSCALE)
        if occupancy_img is None:
            print("Error: Failed to load occupancy image.")
            return

        _, occ_bin = cv2.threshold(occupancy_img, 254, 255, cv2.THRESH_BINARY)
        dist_transform = cv2.distanceTransform(occ_bin, cv2.DIST_L2, 5)

        boundary_nodes = []
        new_node_id_counter = 10000
        clearance_m = 0.5
        
        for target_node in all_nodes:
            dir_x, dir_y = 0.0, 0.0
            valid_dir = False

            #HANDLE 2-EDGE NODES (Standard waypoints/turns)
            if len(target_node.edges) == 2:
                connected_nodes = []
                for edge_id in target_node.edges:
                    edge_obj = next((e for e in all_edges if e.id == edge_id), None)
                    if not edge_obj: continue
                    connected_id = edge_obj.to_id if edge_obj.from_id == target_node.node_id else edge_obj.from_id
                    connected_node = next((n for n in all_nodes if n.node_id == connected_id), None)
                    if connected_node:
                        connected_nodes.append(connected_node)
                        
                if len(connected_nodes) == 2:
                    node1, node2 = connected_nodes[0], connected_nodes[1]
                    
                    #Calculate vectors from target to connected nodes
                    map_v1x = node1.x - target_node.x
                    map_v1y = node1.y - target_node.y
                    map_v2x = node2.x - target_node.x
                    map_v2y = node2.y - target_node.y
                    
                    # NORMALIZE them so edge length doesn't bias the angle!
                    mag1 = math.hypot(map_v1x, map_v1y)
                    mag2 = math.hypot(map_v2x, map_v2y)
                    
                    if mag1 > 0.01 and mag2 > 0.01:
                        map_v1x /= mag1
                        map_v1y /= mag1
                        map_v2x /= mag2
                        map_v2y /= mag2
                        
                        # Convert to image space (flip Y)
                        img_v1x, img_v1y = map_v1x, -map_v1y
                        img_v2x, img_v2y = map_v2x, -map_v2y
                        
                        #True angle bisector (sum of normalized vectors)
                        bisector_x = img_v1x + img_v2x
                        bisector_y = img_v1y + img_v2y

                        if math.hypot(bisector_x, bisector_y) < 0.01:
                            # Straight segments have opposite vectors, so their
                            # angle-bisector sum is zero. Use a cross-section
                            # perpendicular to the segment instead.
                            road_dir_x = -img_v1y
                            road_dir_y = img_v1x
                        else:
                            road_dir_x = bisector_x
                            road_dir_y = bisector_y
                        valid_dir = True

            # HANDLE 1-EDGE NODES (Dead-ends / Entrances)
            elif len(target_node.edges) == 1:
                edge_obj = next((e for e in all_edges if e.id == target_node.edges[0]), None)
                if edge_obj:
                    connected_id = edge_obj.to_id if edge_obj.from_id == target_node.node_id else edge_obj.from_id
                    connected_node = next((n for n in all_nodes if n.node_id == connected_id), None)
                    if connected_node:
                        map_road_dir_x = connected_node.x - target_node.x
                        map_road_dir_y = connected_node.y - target_node.y
                        
                        # Flip the perpendicular vector to swap side 1 and side 2
                        # This prevents the "X" crossing when connecting boundary nodes later
                        # Preserve the original terminal cross-section convention.
                        road_dir_x = -map_road_dir_y
                        road_dir_y = -map_road_dir_x
                        
                        valid_dir = True
                        
            elif len(target_node.edges) == 3:
                connected_nodes = []
                for edge_id in target_node.edges:
                    edge_obj = next((e for e in all_edges if e.id == edge_id), None)
                    if not edge_obj:
                        continue

                    connected_id = edge_obj.to_id if edge_obj.from_id == target_node.node_id else edge_obj.from_id
                    connected_node = next((n for n in all_nodes if n.node_id == connected_id), None)
                    if connected_node:
                        connected_nodes.append(connected_node)

                if len(connected_nodes) == 3:
                    start_px_x = (target_node.x - min_x) * pixels_per_meter
                    start_px_y = (max_y - target_node.y) * pixels_per_meter

                    img_dirs = []
                    for cn in connected_nodes:
                        vx = cn.x - target_node.x
                        vy = cn.y - target_node.y
                        mag = math.hypot(vx, vy)

                        if mag < 0.01:
                            continue

                        # Convert direction to image space, matching your existing Y flip.
                        img_dirs.append((vx / mag, -vy / mag))

                    if len(img_dirs) == 3:
                        pair_idx = 0

                        # For a 3-way junction, take each pair of connected edges.
                        # This creates up to 3 junction corner nodes.
                        for i in range(3):
                            for j in range(i + 1, 3):
                                # Works because indices are only 0, 1, 2.
                                k = 3 - i - j

                                v1x, v1y = img_dirs[i]
                                v2x, v2y = img_dirs[j]
                                v3x, v3y = img_dirs[k]

                                # Same bisector idea as the 2-edge case.
                                bis_x = v1x + v2x
                                bis_y = v1y + v2y
                                bis_mag = math.hypot(bis_x, bis_y)

                                # If the two edges are nearly opposite, use a perpendicular fallback.
                                if bis_mag < 0.01:
                                    bis_x = -v1y
                                    bis_y = v1x
                                    bis_mag = math.hypot(bis_x, bis_y)

                                if bis_mag < 0.01:
                                    pair_idx += 1
                                    continue

                                bis_x /= bis_mag
                                bis_y /= bis_mag

                                # Choose the bisector direction pointing away from the third leg.
                                # If junction corners appear on the wrong side, flip this condition.
                                if (bis_x * v3x + bis_y * v3y) > 0.0:
                                    bis_x = -bis_x
                                    bis_y = -bis_y

                                dist_px = self._ray_march_to_boundary(
                                    dist_transform,
                                    start_px_x,
                                    start_px_y,
                                    bis_x,
                                    bis_y,
                                    clearance_m,
                                    pixels_per_meter
                                )

                                if dist_px is not None and dist_px > 0.0:
                                    new_px_x = start_px_x + (bis_x * dist_px)
                                    new_px_y = start_px_y + (bis_y * dist_px)

                                    new_m_x = (new_px_x / pixels_per_meter) + min_x
                                    new_m_y = max_y - (new_px_y / pixels_per_meter)

                                    # Avoid duplicate junction corners that are extremely close.
                                    too_close = False
                                    for existing in boundary_nodes:
                                        if existing.get("parent_node") == target_node.node_id:
                                            if math.hypot(existing["x"] - new_m_x, existing["y"] - new_m_y) < 0.05:
                                                too_close = True
                                                break

                                    if not too_close:
                                        boundary_nodes.append({
                                            "id": f"node_{new_node_id_counter}",
                                            "x": float(new_m_x),
                                            "y": float(new_m_y),
                                            "type": "boundary",
                                            "parent_node": target_node.node_id,
                                            "side": f"junction_pair_{pair_idx}"
                                        })
                                        new_node_id_counter += 1

                                pair_idx += 1

                # Skip the normal 2-edge/1-edge perpendicular ray march for 3-edge nodes.
                continue
            #CALCULATE PERPENDICULAR & RAY MARCH
            if valid_dir:
                road_dir_mag = math.hypot(road_dir_x, road_dir_y)
                if road_dir_mag < 0.01:
                    continue
                    
                dir_x = road_dir_x / road_dir_mag
                dir_y = road_dir_y / road_dir_mag
                
                start_px_x = (target_node.x - min_x) * pixels_per_meter
                start_px_y = (max_y - target_node.y) * pixels_per_meter 
                
                dist_px_1 = self._ray_march_to_boundary(
                    dist_transform, start_px_x, start_px_y, dir_x, dir_y, clearance_m, pixels_per_meter
                )
                
                dist_px_2 = self._ray_march_to_boundary(
                    dist_transform, start_px_x, start_px_y, -dir_x, -dir_y, clearance_m, pixels_per_meter
                )
                
                if dist_px_1 is not None:
                    new_px_x = start_px_x + (dir_x * dist_px_1)
                    new_px_y = start_px_y + (dir_y * dist_px_1)
                    new_m_x = (new_px_x / pixels_per_meter) + min_x
                    new_m_y = max_y - (new_px_y / pixels_per_meter)
                    
                    boundary_nodes.append({
                        "id": f"node_{new_node_id_counter}",
                        "x": float(new_m_x), "y": float(new_m_y),
                        "type": "boundary", "parent_node": target_node.node_id, "side": "perp_side_1"
                    })
                    new_node_id_counter += 1
                    
                if dist_px_2 is not None:
                    new_px_x = start_px_x + (-dir_x * dist_px_2)
                    new_px_y = start_px_y + (-dir_y * dist_px_2)
                    new_m_x = (new_px_x / pixels_per_meter) + min_x
                    new_m_y = max_y - (new_px_y / pixels_per_meter)
                    
                    boundary_nodes.append({
                        "id": f"node_{new_node_id_counter}",
                        "x": float(new_m_x), "y": float(new_m_y),
                        "type": "boundary", "parent_node": target_node.node_id, "side": "perp_side_2"
                    })
                    new_node_id_counter += 1

        boundary_nodes = self.fix_overlapping_cross_sections(boundary_nodes)
        boundary_nodes, boundary_edges = self.connect_boundary_graph(
            all_nodes,
            all_edges,
            boundary_nodes,
            dist_transform,
            pixels_per_meter,
            min_x,
            max_y,
            clearance_m
        )

        boundary_nodes = self.add_edge_references_to_boundary_nodes(
            boundary_nodes,
            boundary_edges
        )

        output_graph = {"nodes": boundary_nodes, "edges": boundary_edges}

        output_path = self.output_dir / "Smart_Parking_Park_boundary_graph.json"
        with open(output_path, 'w') as f:
            json.dump(output_graph, f, indent=2)
            
        print(f"Successfully generated {len(boundary_nodes)} boundary nodes.")
        print(f"Saved to: {output_path}")

        self.visualize_boundaries(all_nodes, all_edges, boundary_nodes)

        self.visualize_boundary_graph(
            all_nodes,
            all_edges,
            boundary_nodes,
            boundary_edges,
            pixels_per_meter,
            min_x,
            max_y
        )



    def _ray_march_to_boundary(self, dist_transform, start_x, start_y, dir_x, dir_y, clearance_m, pixels_per_meter):
        """
        Ray marches along the direction vector. At each step, it checks the Distance Transform 
        to see if the current point is within the clearance zone of ANY obstacle.
        """
        x, y = float(start_x), float(start_y)
        step = 1.0  # 1 pixel step size
        max_steps = int(10 * pixels_per_meter)  # Safety limit (10 meters)
        clearance_px = clearance_m * pixels_per_meter
        
        img_h, img_w = dist_transform.shape
        prev_dist = 0.0
        
        for _ in range(max_steps):
            x += dir_x * step
            y += dir_y * step
            ix, iy = int(round(x)), int(round(y))
            
            # Check image bounds
            if iy < 0 or iy >= img_h or ix < 0 or ix >= img_w:
                break
                
            # Check the distance to the nearest obstacle from this exact point
            dist_to_obs = dist_transform[iy, ix]
            
            # If we have entered the clearance zone of an obstacle
            if dist_to_obs <= clearance_px:
                # Return the previous step's distance to guarantee we are strictly >= clearance away
                return prev_dist
                
            prev_dist = math.hypot(x - start_x, y - start_y)
                
        return None

    def visualize_boundaries(self, all_nodes, all_edges, boundary_nodes):
        print("\n--- Generating Visual Debug Image ---")
        
        metadata_path = self.output_dir / "map_metadata.json"
        pixels_per_meter = 70.0
        min_x, max_y = 0.0, 0.0
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                meta = json.load(f)
                pixels_per_meter = float(meta.get('resolution', 70.0))
                bounds = meta.get('bounds', {})
                min_x = float(bounds.get('min_x', 0.0))
                max_y = float(bounds.get('max_y', 0.0))
                
        occ_path = self.output_dir / "merged_occupancy.png"
        if not occ_path.exists():
            occ_path = self.output_dir / "occupancy_clean.png"
            
        occupancy_img = cv2.imread(str(occ_path), cv2.IMREAD_GRAYSCALE)
        if occupancy_img is None:
            print("Error: Failed to load occupancy image for visualization.")
            return
            
        vis_img = cv2.cvtColor(occupancy_img, cv2.COLOR_GRAY2BGR)
        
        def m_to_px(x_m, y_m):
            px = int((x_m - min_x) * pixels_per_meter)
            py = int((max_y - y_m) * pixels_per_meter)
            return (px, py)

        node_dict = {n.node_id: n for n in all_nodes}
        
        # 1. Draw Original Edges (Red)
        for edge in all_edges:
            n1 = node_dict.get(edge.from_id)
            n2 = node_dict.get(edge.to_id)
            if n1 and n2:
                cv2.line(vis_img, m_to_px(n1.x, n1.y), m_to_px(n2.x, n2.y), (0, 0, 255), 2)

        # 2. Draw Original Nodes (Blue)
        for n in all_nodes:
            cv2.circle(vis_img, m_to_px(n.x, n.y), 6, (255, 0, 0), -1)

        # 3. Draw Cross-Sections (Magenta) to visualize overlaps
        parents = {}
        for bn in boundary_nodes:
            pid = bn['parent_node']
            if isinstance(pid, list): 
                continue # Skip merged nodes to avoid crashes
            if pid not in parents: 
                parents[pid] = {}
            parents[pid][bn['side']] = bn
            
        for pid, sides in parents.items():
            if 'perp_side_1' in sides and 'perp_side_2' in sides:
                p1 = m_to_px(sides['perp_side_1']['x'], sides['perp_side_1']['y'])
                p2 = m_to_px(sides['perp_side_2']['x'], sides['perp_side_2']['y'])
                cv2.line(vis_img, p1, p2, (255, 0, 255), 1) # Magenta lines

        # 4. Draw Boundary Nodes and Connecting Lines
        for bn in boundary_nodes:
            bn_px = m_to_px(bn['x'], bn['y'])
            
            # Check if this is the extra merged node
            is_merged = isinstance(bn['parent_node'], list) or bn.get('type') == 'boundary_merged'
            
            if is_merged:
                # Draw the extra merged node prominently (Larger white circle with black border)
                cv2.circle(vis_img, bn_px, 8, (0, 0, 0), -1)   # Black outer ring
                cv2.circle(vis_img, bn_px, 6, (255, 255, 255), -1) # White inner fill
                
                # Optional: Draw a small text label so you can spot it easily
                cv2.putText(vis_img, "M", (int(bn_px[0])+10, int(bn_px[1])-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            else:
                # Standard boundary nodes (Green for side 1, Yellow for side 2)
                color = (0, 255, 0) if bn['side'] == 'perp_side_1' else (0, 255, 255)
                cv2.circle(vis_img, bn_px, 4, color, -1)
                
                # Draw cyan line connecting standard nodes to their parent
                parent_node = node_dict.get(bn['parent_node'])
                if parent_node:
                    parent_px = m_to_px(parent_node.x, parent_node.y)
                    cv2.line(vis_img, parent_px, bn_px, (255, 255, 0), 1) 

        # 5. Save the image
        out_path = self.output_dir / "boundary_visual_debug.png"
        cv2.imwrite(str(out_path), vis_img)
        print(f"Saved visual debug to: {out_path}")




    '''THIS IS THE START OF WHERE THE EDGES ARE CALCULATED, ALL ABOVE IS THE NODE CALC.... Currently unused'''

    def _walk_point_to_boundary(self, dist_transform, start_px_x, start_px_y, clearance_px):
        """
        Iteratively walks a point along the gradient of the distance transform 
        until it is exactly 'clearance_px' away from the obstacle.
        """
        curr_x, curr_y = float(start_px_x), float(start_px_y)
        img_h, img_w = dist_transform.shape
        
        # Safety limit to prevent infinite loops
        for _ in range(100): 
            ix, iy = int(round(curr_x)), int(round(curr_y))
            
            # Check bounds
            if iy < 0 or iy >= img_h or ix < 0 or ix >= img_w:
                break
                
            # If we have reached the 0.5m clearance, stop walking
            if dist_transform[iy, ix] >= clearance_px:
                break
                
            # Look at 8 neighbors to find the direction of "most free space" (gradient ascent)
            best_val = -1
            best_dx, best_dy = 0, 0
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = ix + dx, iy + dy
                    if 0 <= nx < img_w and 0 <= ny < img_h:
                        if dist_transform[ny, nx] > best_val:
                            best_val = dist_transform[ny, nx]
                            best_dx, best_dy = dx, dy
                            
            # Move 1 pixel in the direction of most free space
            curr_x += best_dx
            curr_y += best_dy
            
        return curr_x, curr_y


    
    
    def _on_segment(self, p, q, r):
            """Checks if point q lies on segment pr."""
            if (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and 
                min(p[1], r[1]) <= q[1] <= max(p[1], r[1])):
                return True
            return False

    def _segments_intersect(self, p1, p2, p3, p4):
        """Checks if line segment p1-p2 intersects with line segment p3-p4."""
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
        ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
            
        if d1 == 0 and self._on_segment(p3, p1, p4): return True
        if d2 == 0 and self._on_segment(p3, p2, p4): return True
        if d3 == 0 and self._on_segment(p1, p3, p2): return True
        if d4 == 0 and self._on_segment(p1, p4, p2): return True
        
        return False

    def fix_overlapping_cross_sections(self, boundary_nodes):
        print("\n--- Fixing Overlapping Cross Sections ---")
        
        max_id_num = 10000
        for n in boundary_nodes:
            try:
                num = int(n['id'].split('_')[1])
                if num > max_id_num: max_id_num = num
            except: pass
        new_id_counter = max_id_num + 1

        changed = True
        while changed:
            changed = False
            
            parents = {}
            for node in boundary_nodes:
                if isinstance(node['parent_node'], list) or node.get('type') == 'boundary_merged':
                    continue
                pid = node['parent_node']
                if pid not in parents: parents[pid] = []
                parents[pid].append(node)

            segments = []
            for pid, nodes in parents.items():
                if len(nodes) == 2:
                    n1 = nodes[0] if nodes[0]['side'] == 'perp_side_1' else nodes[1]
                    n2 = nodes[1] if nodes[0]['side'] == 'perp_side_1' else nodes[0]
                    segments.append({'parent': pid, 'n1': n1, 'n2': n2})

            for i in range(len(segments)):
                for j in range(i + 1, len(segments)):
                    s1 = segments[i]
                    s2 = segments[j]
                    
                    p1 = (s1['n1']['x'], s1['n1']['y'])
                    p2 = (s1['n2']['x'], s1['n2']['y'])
                    p3 = (s2['n1']['x'], s2['n1']['y'])
                    p4 = (s2['n2']['x'], s2['n2']['y'])
                    
                    if self._segments_intersect(p1, p2, p3, p4):
                        print(f"Cross-section overlap detected between Parent {s1['parent']} and Parent {s2['parent']}")
                        
                        pairs = [
                            (s1['n1'], s2['n1'], math.hypot(p1[0]-p3[0], p1[1]-p3[1])),
                            (s1['n1'], s2['n2'], math.hypot(p1[0]-p4[0], p1[1]-p4[1])),
                            (s1['n2'], s2['n1'], math.hypot(p2[0]-p3[0], p2[1]-p3[1])),
                            (s1['n2'], s2['n2'], math.hypot(p2[0]-p4[0], p2[1]-p4[1]))
                        ]
                        pairs.sort(key=lambda x: x[2])
                        node_a, node_b = pairs[0][0], pairs[0][1]
                        
                        print(f"Merging closest corners: {node_a['id']} and {node_b['id']}")
                        
                        merged_node = {
                            "id": f"node_{new_id_counter}",
                            "x": (node_a['x'] + node_b['x']) / 2.0,
                            "y": (node_a['y'] + node_b['y']) / 2.0,
                            "type": "boundary_merged",
                            "parent_node": [node_a['parent_node'], node_b['parent_node']], 
                            "side": "merged_corner"
                        }
                        new_id_counter += 1
                        
                        boundary_nodes = [n for n in boundary_nodes if n['id'] != node_a['id'] and n['id'] != node_b['id']]
                        boundary_nodes.append(merged_node)
                        
                        changed = True
                        break 
                if changed: break
                    
        print(f"Cross-section fix complete. Final Boundary Nodes: {len(boundary_nodes)}")
        return boundary_nodes


    def connect_boundary_graph(
        self,
        all_nodes,
        all_edges,
        boundary_nodes,
        dist_transform,
        pixels_per_meter,
        min_x,
        max_y,
        clearance_m
    ):
        print("\n--- Connecting Boundary Nodes ---")

        boundary_edges = []
        edge_set = set()
        processed_connections = set()
        edge_id_counter = 0

        # Find the next available boundary node ID.
        node_id_counter = 10000
        for bn in boundary_nodes:
            try:
                num = int(str(bn.get("id", "node_0")).split("_")[1])
                if num >= node_id_counter:
                    node_id_counter = num + 1
            except Exception:
                pass

        clearance_px = clearance_m * pixels_per_meter
        node_lookup = {n.node_id: n for n in all_nodes}

        # Used to find the deepest point inside occupancy.
        # dist_transform > 0 is assumed to be free space.
        # dist_transform == 0 is assumed to be occupancy / obstacle space.
        free_mask = (dist_transform > 0).astype(np.uint8) * 255
        occupied_mask = cv2.bitwise_not(free_mask)
        inside_distance = cv2.distanceTransform(occupied_mask, cv2.DIST_L2, 5)


        # Build a lookup of boundary children by parent node.
        # Merged nodes with multiple parents are made available to each parent.
        children_by_parent = {}
        for bn in boundary_nodes:
            # Inserted waypoints should not be treated as parent boundary children.
            if bn.get("type") in ("boundary_inserted", "boundary_waypoint"):
                continue

            parent = bn.get("parent_node")
            if parent is None:
                continue

            if isinstance(parent, list):
                for pid in parent:
                    children_by_parent.setdefault(pid, []).append(bn)
            else:
                children_by_parent.setdefault(parent, []).append(bn)

        # Deterministic ordering.
        for pid in children_by_parent:
            children_by_parent[pid].sort(key=lambda item: str(item.get("id", "")))

        def m_to_px(x_m, y_m):
            return (
                (x_m - min_x) * pixels_per_meter,
                (max_y - y_m) * pixels_per_meter
            )

        def px_to_m(x_px, y_px):
            return (
                (x_px / pixels_per_meter) + min_x,
                max_y - (y_px / pixels_per_meter)
            )

        def side_sign(point_dict, parent_a, parent_b):
            """
            Returns:
                1  if point is on one side of line parent_a -> parent_b
               -1  if point is on the other side
                0  if effectively on the line
            """
            cross = (
                (parent_b.x - parent_a.x) * (point_dict["y"] - parent_a.y)
                - (parent_b.y - parent_a.y) * (point_dict["x"] - parent_a.x)
            )

            if cross > 1e-6:
                return 1
            if cross < -1e-6:
                return -1
            return 0

        def node_involves_both_parents(bn, pid_a, pid_b):
            """
            True for merged boundary nodes whose parent_node list contains both endpoints.
            These should not create a direct edge between those two parents.
            """
            parent = bn.get("parent_node")
            if isinstance(parent, list):
                return pid_a in parent and pid_b in parent
            return False

        def add_edge_record(from_id, to_id):
            nonlocal edge_id_counter

            if from_id == to_id:
                return

            key = tuple(sorted([str(from_id), str(to_id)]))
            if key in edge_set:
                return

            edge_set.add(key)

            boundary_edges.append({
                "id": f"edge_{edge_id_counter}",
                "from_id": str(from_id),
                "to_id": str(to_id)
            })

            edge_id_counter += 1

        def segment_primary_collision_point(px1, py1, px2, py2, segment_index=0):
            """
            Finds the single best correction point for a segment.

            If the segment enters actual occupancy, return the DEEPEST occupied point
            (the occupied point farthest from free space). Walking this one point out
            lets a single inserted node route the polyline around the obstacle.

            Only if there is no occupancy at all, and the segment cuts deeply into the
            clearance buffer (less than half the clearance), return the tightest point.
            """
            dx = px2 - px1
            dy = py2 - py1
            length = math.hypot(dx, dy)

            if length < 1e-6:
                return None

            steps = max(2, int(math.ceil(length)))
            img_h, img_w = dist_transform.shape

            deepest_occupied = None
            min_clearance_point = None

            # Only treat a clearance violation as real if it cuts deep
            # (less than half the clearance). Shallow grazes are ignored.
            clearance_hard_px = clearance_px * 0.5

            for i in range(steps + 1):
                t = float(i) / float(steps)
                x = px1 + dx * t
                y = py1 + dy * t

                ix = int(round(x))
                iy = int(round(y))

                if ix < 0 or iy < 0 or ix >= img_w or iy >= img_h:
                    continue

                margin = float(dist_transform[iy, ix])

                # Deep clearance cut only (shallow grazes ignored).
                if margin < clearance_hard_px:
                    if min_clearance_point is None or margin < min_clearance_point["margin"]:
                        min_clearance_point = {
                            "priority": 1,
                            "x": x,
                            "y": y,
                            "t": t,
                            "ix": ix,
                            "iy": iy,
                            "margin": margin,
                            "inside_depth": 0.0,
                            "segment_index": segment_index
                        }

                # Actual occupancy collision: keep the deepest occupied point.
                if margin <= 0.0:
                    depth = float(inside_distance[iy, ix])
                    if deepest_occupied is None or depth > deepest_occupied["inside_depth"]:
                        deepest_occupied = {
                            "priority": 2,
                            "x": x,
                            "y": y,
                            "t": t,
                            "ix": ix,
                            "iy": iy,
                            "margin": margin,
                            "inside_depth": depth,
                            "segment_index": segment_index
                        }

            if deepest_occupied is not None:
                return deepest_occupied

            if min_clearance_point is not None:
                return min_clearance_point

            return None

        def walk_to_clearance(px, py):
            """
            Walks a violating point out of occupancy and then out of the clearance buffer.

            If the point is deeply inside occupancy, it first moves toward free space
            using inside_distance. Once it reaches free space, it hands over to the
            existing _walk_point_to_boundary method.
            """
            x, y = float(px), float(py)
            img_h, img_w = dist_transform.shape

            # First escape occupied space if necessary.
            for _ in range(200):
                ix = int(round(x))
                iy = int(round(y))

                if ix < 0 or iy < 0 or ix >= img_w or iy >= img_h:
                    break

                if float(dist_transform[iy, ix]) >= clearance_px:
                    return x, y

                # If inside occupancy, move toward nearest free space.
                if float(inside_distance[iy, ix]) > 0.0:
                    best_val = float(inside_distance[iy, ix])
                    best_dx = 0
                    best_dy = 0
                    best_margin = float(dist_transform[iy, ix])

                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue

                            nx, ny = ix + dx, iy + dy

                            if 0 <= nx < img_w and 0 <= ny < img_h:
                                val = float(inside_distance[ny, nx])
                                margin = float(dist_transform[ny, nx])

                                # Prefer the neighbor closest to free space.
                                # If tied, prefer the neighbor with better clearance margin.
                                if val < best_val - 1e-6 or (
                                    abs(val - best_val) <= 1e-6 and margin > best_margin
                                ):
                                    best_val = val
                                    best_dx = dx
                                    best_dy = dy
                                    best_margin = margin

                    if best_dx == 0 and best_dy == 0:
                        break

                    x += best_dx
                    y += best_dy
                    continue

                # No longer inside occupancy.
                break

            # Use your original clearance walker for the final move to the clearance boundary.
            return self._walk_point_to_boundary(
                dist_transform,
                x,
                y,
                clearance_px
            )

        def add_boundary_connection(from_bn, to_bn):
            """
            Connect two boundary nodes.

            Primary correction:
                Insert one node at the last point exiting occupancy.

            Backup:
                If the boundary connection is still violating afterwards, repeat
                up to 3 times total.
            """
            nonlocal node_id_counter

            if not from_bn or not to_bn:
                return

            if from_bn.get("id") == to_bn.get("id"):
                return

            connection_key = tuple(sorted([str(from_bn["id"]), str(to_bn["id"])]))
            if connection_key in processed_connections:
                return

            processed_connections.add(connection_key)

            # The connection starts as a straight line between the two boundary nodes.
            polyline = [from_bn, to_bn]

            # Additional nodes are a backup, not the default primary method.
            max_insertions = 3
            insertions = 0

            while insertions < max_insertions:
                best = None

                # Check every current segment in the polyline.
                for seg_idx in range(len(polyline) - 1):
                    a = polyline[seg_idx]
                    b = polyline[seg_idx + 1]

                    ax, ay = m_to_px(a["x"], a["y"])
                    bx, by = m_to_px(b["x"], b["y"])

                    candidate = segment_primary_collision_point(ax, ay, bx, by, seg_idx)

                    if candidate is None:
                        continue

                    if best is None:
                        best = candidate
                        continue

                    # Prefer actual occupancy exit points over clearance-only fallback points.
                    if candidate["priority"] > best["priority"]:
                        best = candidate

                    # For equal priority, prefer the later point along the polyline.
                    elif candidate["priority"] == best["priority"]:
                        candidate_order = (
                            candidate["segment_index"],
                            candidate.get("t", 0.0)
                        )
                        best_order = (
                            best["segment_index"],
                            best.get("t", 0.0)
                        )

                        if candidate_order > best_order:
                            best = candidate

                # No remaining collision/clearance issue.
                if best is None:
                    break

                walked_x, walked_y = walk_to_clearance(best["x"], best["y"])

                # If walking did not move the point, stop to avoid useless duplicate nodes.
                if math.hypot(walked_x - best["x"], walked_y - best["y"]) < 0.25:
                    break

                wx_m, wy_m = px_to_m(walked_x, walked_y)

                inserted_node = {
                    "id": f"node_{node_id_counter}",
                    "x": float(wx_m),
                    "y": float(wy_m),
                    "type": "boundary_inserted",
                    "parent_node": None,
                    "side": "inserted_waypoint",
                    "source_boundary_nodes": [
                        str(from_bn["id"]),
                        str(to_bn["id"])
                    ]
                }

                node_id_counter += 1
                boundary_nodes.append(inserted_node)

                # Insert the new waypoint into the polyline.
                insert_at = best["segment_index"] + 1
                polyline.insert(insert_at, inserted_node)

                insertions += 1

            # Optional warning if the edge is still violating after the backup insertions.
            still_violating = False
            for seg_idx in range(len(polyline) - 1):
                a = polyline[seg_idx]
                b = polyline[seg_idx + 1]

                ax, ay = m_to_px(a["x"], a["y"])
                bx, by = m_to_px(b["x"], b["y"])

                if segment_primary_collision_point(ax, ay, bx, by, seg_idx) is not None:
                    still_violating = True
                    break

            if still_violating:
                print(
                    f"Warning: boundary connection {from_bn.get('id')} -> {to_bn.get('id')} "
                    f"still violates clearance after {insertions} insertions."
                )

            # Add edges along the final polyline.
            for i in range(len(polyline) - 1):
                add_edge_record(polyline[i]["id"], polyline[i + 1]["id"])

        def best_pairs_for_edge(parent_a_id, parent_b_id):
            """
            For an original graph edge between parent A and parent B, choose boundary-child
            pairings that do not cross the original edge.

            Returns a list of pairs:
                [(child_of_A, child_of_B), ...]
            """
            parent_a = node_lookup.get(parent_a_id)
            parent_b = node_lookup.get(parent_b_id)

            if parent_a is None or parent_b is None:
                return []

            a_children = children_by_parent.get(parent_a_id, [])
            b_children = children_by_parent.get(parent_b_id, [])

            if not a_children or not b_children:
                return []

            chosen_by_side = {}
            fallback_candidates = []

            for ca in a_children:
                # If this child already belongs to both parents, skip it for this parent-parent edge.
                if node_involves_both_parents(ca, parent_a_id, parent_b_id):
                    continue

                for cb in b_children:
                    if node_involves_both_parents(cb, parent_a_id, parent_b_id):
                        continue

                    if ca.get("id") == cb.get("id"):
                        continue

                    p1 = (ca["x"], ca["y"])
                    p2 = (cb["x"], cb["y"])
                    pa = (parent_a.x, parent_a.y)
                    pb = (parent_b.x, parent_b.y)

                    # As requested, test only against the original edge between these two parents.
                    crosses = self._segments_intersect(p1, p2, pa, pb)

                    dist = math.hypot(
                        ca["x"] - cb["x"],
                        ca["y"] - cb["y"]
                    )

                    s1 = side_sign(ca, parent_a, parent_b)
                    s2 = side_sign(cb, parent_a, parent_b)

                    side_key = s1 if s1 != 0 else s2
                    if side_key == 0:
                        side_key = 999

                    fallback_candidates.append((dist, ca, cb, side_key, crosses))

                    if crosses:
                        continue

                    # Require both children to be on the same physical side of the original edge,
                    # unless one side value is effectively zero.
                    if s1 != 0 and s2 != 0 and s1 != s2:
                        continue

                    current = chosen_by_side.get(side_key)
                    if current is None or dist < current[0]:
                        chosen_by_side[side_key] = (dist, ca, cb)

            if chosen_by_side:
                pairs = []
                for side_key in sorted(chosen_by_side.keys()):
                    _, ca, cb = chosen_by_side[side_key]
                    pairs.append((ca, cb))
                return pairs

            # Requested fallback:
            # If both possible pairings cross, connect side_1 to side_1 and side_2 to side_2.
            def find_by_side(nodes, side_name):
                for n in nodes:
                    if n.get("side") == side_name:
                        return n
                return None

            forced_pairs = []
            for side_name in ("perp_side_1", "perp_side_2"):
                na = find_by_side(a_children, side_name)
                nb = find_by_side(b_children, side_name)

                if na and nb and na.get("id") != nb.get("id"):
                    forced_pairs.append((na, nb))

            if forced_pairs:
                return forced_pairs

            # Final fallback: shortest available pairing.
            if fallback_candidates:
                fallback_candidates.sort(key=lambda item: item[0])
                return [(fallback_candidates[0][1], fallback_candidates[0][2])]

            return []

        # ------------------------------------------------------------------
        # 1. Connect along every original graph edge.
        # ------------------------------------------------------------------
        for edge in all_edges:
            pairs = best_pairs_for_edge(edge.from_id, edge.to_id)

            for from_bn, to_bn in pairs:
                add_boundary_connection(from_bn, to_bn)

        # ------------------------------------------------------------------
        # 2. Cap terminal nodes by connecting their two boundary children.
        # ------------------------------------------------------------------
        for original_node in all_nodes:
            if len(original_node.edges) != 1:
                continue

            terminal_children = children_by_parent.get(original_node.node_id, [])

            if len(terminal_children) == 2:
                add_boundary_connection(terminal_children[0], terminal_children[1])

            elif len(terminal_children) > 2:
                # If more than two children exist because of merges, choose the widest cap.
                best_pair = None
                best_dist = -1.0

                for i in range(len(terminal_children)):
                    for j in range(i + 1, len(terminal_children)):
                        d = math.hypot(
                            terminal_children[i]["x"] - terminal_children[j]["x"],
                            terminal_children[i]["y"] - terminal_children[j]["y"]
                        )

                        if d > best_dist:
                            best_dist = d
                            best_pair = (terminal_children[i], terminal_children[j])

                if best_pair is not None:
                    add_boundary_connection(best_pair[0], best_pair[1])

        print(f"Generated {len(boundary_edges)} boundary edges.")
        return boundary_nodes, boundary_edges

    def add_edge_references_to_boundary_nodes(self, boundary_nodes, boundary_edges):
        """
        Adds an 'edges' list to each boundary node containing the IDs of all
        boundary edges connected to that node.

        This is done automatically from the final boundary_edges list so the
        node edge references stay consistent.
        """
        print("\n--- Adding Edge References to Boundary Nodes ---")

        node_map = {}

        # Ensure every boundary node has an edge list.
        for bn in boundary_nodes:
            if "edges" not in bn or not isinstance(bn.get("edges"), list):
                bn["edges"] = []

            node_map[bn["id"]] = bn

        # Populate node edge lists from the final boundary edge list.
        for edge in boundary_edges:
            edge_id = str(edge.get("id"))
            from_id = str(edge.get("from_id"))
            to_id = str(edge.get("to_id"))

            if from_id in node_map:
                if edge_id not in node_map[from_id]["edges"]:
                    node_map[from_id]["edges"].append(edge_id)

            if to_id in node_map:
                if edge_id not in node_map[to_id]["edges"]:
                    node_map[to_id]["edges"].append(edge_id)

        # Optional: sort edge IDs numerically for cleaner JSON output.
        for bn in boundary_nodes:
            try:
                bn["edges"].sort(key=lambda eid: int(str(eid).split("_")[1]))
            except Exception:
                pass

        connected_count = sum(1 for bn in boundary_nodes if len(bn.get("edges", [])) > 0)
        print(f"Added edge references to {connected_count} boundary nodes.")

        return boundary_nodes

    def visualize_boundary_graph(
        self,
        all_nodes,
        all_edges,
        boundary_nodes,
        boundary_edges,
        pixels_per_meter,
        min_x,
        max_y
    ):
        print("\n--- Generating Boundary Graph Visual Debug ---")

        occ_path = self.output_dir / "occupancy_clean.png"
        if not occ_path.exists():
            occ_path = self.output_dir / "merged_occupancy.png"

        occupancy_img = cv2.imread(str(occ_path), cv2.IMREAD_GRAYSCALE)
        if occupancy_img is None:
            print("Error: Failed to load occupancy image for boundary graph visualization.")
            return

        vis_img = cv2.cvtColor(occupancy_img, cv2.COLOR_GRAY2BGR)

        def m_to_px(x_m, y_m):
            return (
                int((x_m - min_x) * pixels_per_meter),
                int((max_y - y_m) * pixels_per_meter)
            )

        node_dict = {n.node_id: n for n in all_nodes}

        # Draw original graph edges.
        for edge in all_edges:
            n1 = node_dict.get(edge.from_id)
            n2 = node_dict.get(edge.to_id)

            if n1 and n2:
                cv2.line(
                    vis_img,
                    m_to_px(n1.x, n1.y),
                    m_to_px(n2.x, n2.y),
                    (0, 0, 255),
                    2
                )

        # Draw original graph nodes.
        for n in all_nodes:
            cv2.circle(
                vis_img,
                m_to_px(n.x, n.y),
                5,
                (255, 0, 0),
                -1
            )

        # Draw new boundary edges.
        bn_dict = {bn["id"]: bn for bn in boundary_nodes}

        for b_edge in boundary_edges:
            n1 = bn_dict.get(b_edge["from_id"])
            n2 = bn_dict.get(b_edge["to_id"])

            if n1 and n2:
                cv2.line(
                    vis_img,
                    m_to_px(n1["x"], n1["y"]),
                    m_to_px(n2["x"], n2["y"]),
                    (0, 165, 255),
                    2
                )

        # Draw boundary nodes.
        for bn in boundary_nodes:
            px = m_to_px(bn["x"], bn["y"])
            btype = bn.get("type", "")
            side = bn.get("side", "")

            if btype == "boundary_inserted":
                # Inserted clearance-walking waypoint.
                cv2.circle(vis_img, px, 5, (0, 255, 255), -1)
                cv2.circle(vis_img, px, 2, (0, 0, 0), -1)

            elif btype == "boundary_merged" or isinstance(bn.get("parent_node"), list):
                # Merged corner node.
                cv2.circle(vis_img, px, 7, (0, 0, 0), -1)
                cv2.circle(vis_img, px, 5, (255, 255, 255), -1)

            elif str(side).startswith("junction_pair_"):
                # 3-edge junction corner node.
                cv2.circle(vis_img, px, 5, (255, 0, 255), -1)

            else:
                # Standard boundary child nodes.
                color = (0, 255, 0) if side == "perp_side_1" else (0, 255, 255)
                cv2.circle(vis_img, px, 4, color, -1)

        out_path = self.output_dir / "boundary_graph_visual_debug.png"
        cv2.imwrite(str(out_path), vis_img)
        print(f"Saved boundary graph visual debug to: {out_path}")


def main():
    default_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output/Smart_Parking_Park")
    parser = argparse.ArgumentParser(description="Run GraphProcessor (minimal runner)")
    parser.add_argument("--output-dir", "-o", default=default_output, help="Path to output directory")
    args = parser.parse_args()

    print("Running GraphProcessor on:", args.output_dir)
    gp = GraphProcessor(args.output_dir)
    gp.find_edges_per_node()


if __name__ == "__main__":
    main()