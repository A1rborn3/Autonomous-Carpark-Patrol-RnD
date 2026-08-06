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
        graph_path = self.output_dir / "Smart_Parking_Park_graph.json"
        if graph_path.exists():
            print(f"Found: {graph_path}")
        else:
            print(f"Graph file not found at: {graph_path}")
            return
            
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
        occ_path = self.output_dir / "merged_occupancy.png"
        if not occ_path.exists():
            occ_path = self.output_dir / "occupancy_clean.png"
            
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
                        road_dir_x = -map_road_dir_y 
                        road_dir_y = -map_road_dir_x #-/+ to edit 90 degree rotation
                        
                        valid_dir = True
            #elif for 3 edge nodes
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
        output_graph = {"nodes": boundary_nodes, "edges": []}
        output_path = self.output_dir / "Smart_Parking_Park_boundary_graph.json"
        with open(output_path, 'w') as f:
            json.dump(output_graph, f, indent=2)
            
        print(f"Successfully generated {len(boundary_nodes)} boundary nodes.")
        print(f"Saved to: {output_path}")

        self.visualize_boundaries(all_nodes, all_edges, boundary_nodes)



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