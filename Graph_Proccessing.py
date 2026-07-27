import os
import argparse
from pathlib import Path
from dataclasses import dataclass
import json


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
            id = int(edges_data['id'].split('_')[1])
            from_id = int(edges_data['from_id'].split('_')[1])
            to_id = int(edges_data['to_id'].split('_')[1])
            
            #create the instance
            edge_instance = Edges(id=id, from_id=from_id, to_id=to_id)
            
            #save to list
            all_edges.append(edge_instance)
            
            print(f"Edge ID: {edge_instance.id}, From: {edge_instance.from_id}, To: {edge_instance.to_id}")

        
        if all_edges:
            print(f"Total Edges: {len(all_edges)} item 3 : Edge ID: {all_edges[3].id} From: {all_edges[3].from_id} To: {all_edges[3].to_id}")
        else:
            print("No edges found in the JSON file.")
#im tired but now it prints the json file lol
#will use this class below to store the info for easier use 

#all edges saved to the edge class with refeance to the node ids, next need to store nodes

@dataclass       
class Node:
    id: int
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
    