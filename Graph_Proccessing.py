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
                        road_dir_y = -map_road_dir_x
                        
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
        
        # 1. Get metadata for coordinate conversion
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
                
        # 2. Load occupancy image
        occ_path = self.output_dir / "merged_occupancy.png"
        if not occ_path.exists():
            occ_path = self.output_dir / "occupancy_clean.png"
            
        occupancy_img = cv2.imread(str(occ_path), cv2.IMREAD_GRAYSCALE)
        if occupancy_img is None:
            print("Error: Failed to load occupancy image for visualization.")
            return
            
        # Convert to BGR so we can draw colors
        vis_img = cv2.cvtColor(occupancy_img, cv2.COLOR_GRAY2BGR)
        
        # Helper to convert meters to pixels (with Y-axis flip)
        def m_to_px(x_m, y_m):
            px = int((x_m - min_x) * pixels_per_meter)
            py = int((max_y - y_m) * pixels_per_meter)
            return (px, py)

        # 3. Draw Original Edges (Red)
        node_dict = {n.node_id: n for n in all_nodes}
        for edge in all_edges:
            n1 = node_dict.get(edge.from_id)
            n2 = node_dict.get(edge.to_id)
            if n1 and n2:
                cv2.line(vis_img, m_to_px(n1.x, n1.y), m_to_px(n2.x, n2.y), (0, 0, 255), 2)

        # 4. Draw Original Nodes (Blue)
        for n in all_nodes:
            cv2.circle(vis_img, m_to_px(n.x, n.y), 6, (255, 0, 0), -1)
            # Optional: draw node ID text
            # cv2.putText(vis_img, str(n.node_id), m_to_px(n.x, n.y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 5. Draw Boundary Nodes and Connecting Lines
        for bn in boundary_nodes:
            bn_px = m_to_px(bn['x'], bn['y'])
            
            # Color code the sides (Green for side 1, Yellow for side 2)
            color = (0, 255, 0) if bn['side'] == 'perp_side_1' else (0, 255, 255)
            cv2.circle(vis_img, bn_px, 4, color, -1)
            
            # Draw a line from the parent node to the boundary node to show the offset
            parent_node = node_dict.get(bn['parent_node'])
            if parent_node:
                parent_px = m_to_px(parent_node.x, parent_node.y)
                cv2.line(vis_img, parent_px, bn_px, (255, 255, 0), 1) # Cyan lines

        # 6. Save the image
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

    def _resolve_segment_intersection(self, node_a, node_b, dist_transform, clearance_m, pixels_per_meter, min_x, max_y, depth=0):
        """
        Recursively checks a line segment between two boundary nodes. 
        If it intersects an obstacle, it inserts a new 'corner' node on the 0.5m boundary.
        """
        # Prevent infinite recursion on extremely sharp corners
        if depth > 5:  
            return [node_b]

        # Convert meters to pixels
        px1 = int((node_a['x'] - min_x) * pixels_per_meter)
        py1 = int((max_y - node_a['y']) * pixels_per_meter)
        px2 = int((node_b['x'] - min_x) * pixels_per_meter)
        py2 = int((max_y - node_b['y']) * pixels_per_meter)
        
        # Sample points along the line to find the closest point to an obstacle
        min_dist = float('inf')
        worst_px, worst_py = px1, py1
        
        # Check 50 points along the line segment
        for i in range(51):
            t = i / 50.0
            curr_x = px1 + (px2 - px1) * t
            curr_y = py1 + (py2 - py1) * t
            ix, iy = int(round(curr_x)), int(round(curr_y))
            
            if 0 <= iy < dist_transform.shape[0] and 0 <= ix < dist_transform.shape[1]:
                dist = dist_transform[iy, ix]
                if dist < min_dist:
                    min_dist = dist
                    worst_px, worst_py = curr_x, curr_y

        clearance_px = clearance_m * pixels_per_meter

        # If the closest point is safely outside the clearance zone, the line is good!
        if min_dist >= clearance_px:
            return [node_b]

        # --- INTERSECTION DETECTED ---
        # 1. Walk the worst point out to the 0.5m boundary
        corner_px_x, corner_px_y = self._walk_point_to_boundary(
            dist_transform, worst_px, worst_py, clearance_px
        )
        
        # 2. Convert the new corner pixel back to meters
        corner_m_x = (corner_px_x / pixels_per_meter) + min_x
        corner_m_y = max_y - (corner_px_y / pixels_per_meter)
        
        # 3. Create the new corner node (FIXED ID GENERATION)
        corner_node = {
            # Generate a unique ID using memory addresses to prevent collisions
            "id": f"corner_{id(node_a)}_{id(node_b)}_{depth}", 
            "x": float(corner_m_x),
            "y": float(corner_m_y),
            "type": "corner",
            "parent_node": node_a.get('parent_node'),
            "side": node_a.get('side')
        }
        
        # 4. Recursively resolve the two new halves (A -> Corner, and Corner -> B)
        left_half = self._resolve_segment_intersection(
            node_a, corner_node, dist_transform, clearance_m, pixels_per_meter, min_x, max_y, depth + 1
        )
        right_half = self._resolve_segment_intersection(
            corner_node, node_b, dist_transform, clearance_m, pixels_per_meter, min_x, max_y, depth + 1
        )
        
        return left_half + right_half





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