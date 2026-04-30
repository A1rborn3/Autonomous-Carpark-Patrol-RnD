import simplekml
import math
import os

class KMLExporter:
    def __init__(self, base_lat, base_lon):
        """
        :param base_lat: Latitude of the local origin (0,0,0).
        :param base_lon: Longitude of the local origin (0,0,0).
        """
        self.base_lat = base_lat
        self.base_lon = base_lon
        self.R = 6378137.0 # Earth radius in meters

    def local_to_gps(self, x, y):
        """Simple flat-earth approximation for small areas."""
        d_lat = y / self.R
        d_lon = x / (self.R * math.cos(math.pi * self.base_lat / 180.0))
        
        lat = self.base_lat + (d_lat * 180.0 / math.pi)
        lon = self.base_lon + (d_lon * 180.0 / math.pi)
        return lat, lon

    def export_all(self, ortho_path, bounds, nodes, edges, resolution, occ_path=None, output_kml="map.kml"):
        """Exports the orthomosaic, occupancy map, and road graph to a single KML file."""
        print(f"Exporting Unified KML: {output_kml}")
        kml = simplekml.Kml()
        
        # Calculate GPS bounds for overlays
        south, west = self.local_to_gps(bounds['min_x'], bounds['min_y'])
        north, east = self.local_to_gps(bounds['max_x'], bounds['max_y'])
        
        # 1. Folders
        overlays_folder = kml.newfolder(name="Map Overlays")
        graph_folder = kml.newfolder(name="Road Graph")
        nodes_folder = graph_folder.newfolder(name="Nodes")
        edges_folder = graph_folder.newfolder(name="Edges")
        
        # 2. Add Overlays
        def add_overlay(folder, name, img_path):
            overlay = folder.newgroundoverlay(name=name)
            try:
                rel_path = os.path.relpath(img_path, os.path.dirname(output_kml))
            except ValueError:
                rel_path = os.path.abspath(img_path)
            
            overlay.icon.href = rel_path
            overlay.latlonbox.north = north
            overlay.latlonbox.south = south
            overlay.latlonbox.east = east
            overlay.latlonbox.west = west
            return overlay

        add_overlay(overlays_folder, "Orthomosaic", ortho_path)
        if occ_path:
            occ_overlay = add_overlay(overlays_folder, "Occupancy Map", occ_path)
            occ_overlay.visibility = 0 
            
        # 3. Add Graph
        min_x = bounds['min_x']
        max_y = bounds['max_y']
        
        node_coords = {}
        for n in nodes:
            local_x = (n['x'] / resolution) + min_x
            local_y = max_y - (n['y'] / resolution)
            
            lat, lon = self.local_to_gps(local_x, local_y)
            node_coords[n['id']] = (lon, lat)
            
            pnt = nodes_folder.newpoint(name=f"Node {n['id'].split('_')[1]}", coords=[(lon, lat)])
            if n.get('type') == 'entrance_exit':
                pnt.name += " (Entrance/Exit)"
                
        for e in edges:
            lon1, lat1 = node_coords[e['from_id']]
            lon2, lat2 = node_coords[e['to_id']]
            
            linestring = edges_folder.newlinestring(name=f"Edge {e['id'].split('_')[1]}")
            linestring.coords = [(lon1, lat1), (lon2, lat2)]
            linestring.extendeddata.newdata(name='from_id', value=e['from_id'])
            linestring.extendeddata.newdata(name='to_id', value=e['to_id'])
            
        kml.save(output_kml)
        print(f"Unified KML saved to {output_kml}")
        return output_kml
