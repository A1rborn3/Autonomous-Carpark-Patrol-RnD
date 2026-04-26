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

    def export_orthomosaic(self, ortho_path, bounds, occ_path=None, output_kml="map.kml"):
        """Creates georeferenced KML GroundOverlays."""
        print(f"Exporting KML: {output_kml}")
        kml = simplekml.Kml()
        
        # Calculate GPS bounds
        south, west = self.local_to_gps(bounds['min_x'], bounds['min_y'])
        north, east = self.local_to_gps(bounds['max_x'], bounds['max_y'])
        
        def add_overlay(name, img_path):
            overlay = kml.newgroundoverlay(name=name)
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

        # 1. Orthomosaic Layer
        add_overlay("Orthomosaic", ortho_path)

        # 2. Occupancy Layer (Optional)
        if occ_path:
            occ_overlay = add_overlay("Occupancy Map", occ_path)
            # Default occupancy to half transparent so you can see both initially
            occ_overlay.visibility = 0 

        kml.save(output_kml)
        print(f"KML saved to {output_kml}")
        return output_kml
