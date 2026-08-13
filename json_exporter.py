import json
import os

class JSONExporter:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def export_graph(self, nodes, edges, bounds, resolution, filename="road_graph.json"):
        """
        Exports nodes and edges in Cartesian coordinates (meters).
        
        :param nodes: List of nodes with 'x', 'y' in pixels.
        :param edges: List of edges with 'from_id', 'to_id'.
        :param bounds: Dict with 'min_x', 'max_x', 'min_y', 'max_y' in meters.
        :param resolution: Pixels per meter.
        """
        cartesian_nodes = []
        for node in nodes:
            # Convert pixel (u, v) to meter (x, y)
            mx = (node['x'] / resolution) + bounds['min_x']
            my = bounds['max_y'] - (node['y'] / resolution)
            
            cartesian_nodes.append({
                'id': node['id'],
                'x': round(mx, 4),
                'y': round(my, 4),
                'type': node.get('type', 'waypoint')
            })
            
        # Edges don't need coordinate conversion, just referencing IDs
        data = {
            'metadata': {
                'bounds': bounds,
                'resolution': resolution
            },
            'nodes': cartesian_nodes,
            'edges': edges
        }
        
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)
            
        print(f"Road graph exported to {output_path}")
        return output_path

    def export_metadata(self, bounds, resolution, filename="map_metadata.json"):
        """Saves basic map metadata for the GUI to align layers."""
        data = {
            'bounds': bounds,
            'resolution': resolution
        }
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)
        return output_path

    def export_unitree_waypoints(self, nodes, edges, bounds, resolution, filename_prefix="road_graph"):
        """Helper method to export Unitree Go2 robot dog waypoints directly from JSONExporter."""
        from unitree_exporter import UnitreeGo2Exporter
        exporter = UnitreeGo2Exporter(self.output_dir)
        return exporter.export_unitree_waypoints(nodes, edges, bounds, resolution, filename_prefix=filename_prefix)

