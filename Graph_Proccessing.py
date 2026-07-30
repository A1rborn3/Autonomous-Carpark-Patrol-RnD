import os
import argparse
from pathlib import Path
from dataclasses import dataclass
import json
import math

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
            
            #create the instance
            edge_instance = Edges(id=edge_id, from_id=from_id, to_id=to_id)
            
            #save to list
            all_edges.append(edge_instance)
            
            print(f"Edge ID: {edge_instance.id}, From: {edge_instance.from_id}, To: {edge_instance.to_id}")

        
        if all_edges:
            print(f"Total Edges: {len(all_edges)} item 3 : Edge ID: {all_edges[3].id} From: {all_edges[3].from_id} To: {all_edges[3].to_id}")
        else:
            print("No edges found in the JSON file.")

        all_nodes = []

        for node_data in data["nodes"]:
            #assign basic data
            node_id = int(node_data['id'].split('_')[1]) 
            x = float(node_data['x'])
            y = float(node_data['y'])
            node_edges = [] 
            #itterate over all edges and see if any connect to current node
            for edge in all_edges:
                if edge.from_id == node_id:
                    node_edges.append(edge.id)
                if edge.to_id == node_id:
                    node_edges.append(edge.id)        
            node_instance = Node(node_id=node_id, x=x, y=y, edges=node_edges)

            all_nodes.append(node_instance)

        if all_nodes:
            print(f"Total nodes: {len(all_nodes)} item 3 : Edge ID: {all_nodes[2].node_id} From: {all_nodes[2].edges[0]} To: ")
            if len(all_nodes[2].edges)>1:
                print(f"{all_nodes[2].edges[1]}")
        else:
            print("No nodes found in the JSON file.")


        #testing with node 2?
        #taget node is the centre node of however many
        print("\n--- Testing Angle & Magnitude Calculation ---")
        self.find_angle_of_edges(all_nodes, all_edges, target_node_index=2)

    def find_angle_of_edges(self, all_nodes, all_edges, target_node_index):
        target_node = all_nodes[target_node_index]

        if len(target_node.edges) == 2:
            connected_nodes = []
            
            # 1. Find the actual Edge objects and the connected Node objects
            for edge_id in target_node.edges:
                edge_obj = next((e for e in all_edges if e.id == edge_id), None)
                if not edge_obj: continue
                    
                if edge_obj.from_id == target_node.node_id:
                    connected_id = edge_obj.to_id
                else:
                    connected_id = edge_obj.from_id
                    
                connected_node = next((n for n in all_nodes if n.node_id == connected_id), None)
                if connected_node:
                    connected_nodes.append(connected_node)
                    
            if len(connected_nodes) == 2:
                node1 = connected_nodes[0]
                node2 = connected_nodes[1]
                
                # 2. Calculate vectors from target node to the connected nodes
                v1x = node1.x - target_node.x
                v1y = node1.y - target_node.y
                v2x = node2.x - target_node.x
                v2y = node2.y - target_node.y
                
                # 3. Calculate the magnitude (length) of the two edges
                mag1 = math.hypot(v1x, v1y)
                mag2 = math.hypot(v2x, v2y)
                
                if mag1 == 0 or mag2 == 0:
                    print("Error: One of the edges has zero length.")
                    return None
                
                # ==========================================
                # NEW MATH: Finding the Half-Angle (Bisector)
                # ==========================================
                
                # A. Normalize the vectors (shrink them to exactly length 1)
                # This keeps only their direction, removing the distance
                n1x, n1y = v1x / mag1, v1y / mag1
                n2x, n2y = v2x / mag2, v2y / mag2
                
                # B. Add the normalized vectors together
                # This mathematically results in a vector pointing exactly 
                # at the "half angle" between the two edges!
                bisector_x = n1x + n2x
                bisector_y = n1y + n2y
                
                # C. Calculate the magnitude of this new bisector vector
                bisector_mag = math.hypot(bisector_x, bisector_y)
                
                # Edge case: If edges point in exact opposite directions (180 deg straight line)
                if bisector_mag == 0:
                    print("Edges are perfectly straight (180 degrees). No unique inside angle.")
                    return None
                
                # D. Normalize the bisector to get a clean direction vector (length = 1)
                dir_x = bisector_x / bisector_mag
                dir_y = bisector_y / bisector_mag
                
                # Calculate the angle in degrees just for printing/debugging
                bisector_angle_rad = math.atan2(dir_y, dir_x)
                bisector_angle_deg = math.degrees(bisector_angle_rad)
                
                # Also calculate the inside angle just to print it
                dot_product = (v1x * v2x) + (v1y * v2y)
                cos_theta = max(-1.0, min(1.0, dot_product / (mag1 * mag2)))
                inside_angle_deg = math.degrees(math.acos(cos_theta))
                
                print(f"Target Node {target_node.node_id}:")
                print(f"  Inside Angle: {inside_angle_deg:.2f}°")
                print(f"  Bisector Angle (Half-way): {bisector_angle_deg:.2f}°")
                print(f"  Direction Vector: ({dir_x:.4f}, {dir_y:.4f})")
                
                # Return the data so you can plot your new node
                return {
                    "dir_x": dir_x,
                    "dir_y": dir_y,
                    "inside_angle_deg": inside_angle_deg,
                    "bisector_angle_deg": bisector_angle_deg
                }
            else:
                print("Could not find both connected nodes in the node list.")
                return None
        else:
            print(f"Node {target_node.node_id} has {len(target_node.edges)} edges. Skipping.")
            return None

    
        

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
    