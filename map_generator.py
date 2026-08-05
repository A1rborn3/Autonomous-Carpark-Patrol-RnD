import numpy as np
import cv2
import json
import os

class MapGenerator:
    def __init__(self, resolution=20):
        """
        :param resolution: Pixels per meter. Default is 20 (0.05m per pixel).
        """
        self.resolution = resolution

    def generate_maps(self, ground_pcd, obstacle_pcd):
        """
        Projects 3D points to 2D to create orthomosaic and occupancy grid.
        Assumes points are aligned to Z=0.
        """
        print("Generating 2D maps...")
        
        ground_pts = np.asarray(ground_pcd.points)
        ground_clr = np.asarray(ground_pcd.colors)
        obs_pts = np.asarray(obstacle_pcd.points)
        
        # Calculate bounds
        all_pts = np.vstack([ground_pts, obs_pts]) if obs_pts.size > 0 else ground_pts
        min_x, min_y = np.min(all_pts[:, 0]), np.min(all_pts[:, 1])
        max_x, max_y = np.max(all_pts[:, 0]), np.max(all_pts[:, 1])
        
        # Grid dimensions
        width = int(np.ceil((max_x - min_x) * self.resolution)) + 1
        height = int(np.ceil((max_y - min_y) * self.resolution)) + 1
        
        print(f"Map Size: {width}x{height} pixels")
        
        # Initialize maps
        # Orthomosaic (RGB)
        orthomosaic = np.zeros((height, width, 3), dtype=np.uint8)
        # Occupancy Grid (Grayscale: 255=walkable, 0=blocked, 127=unknown)
        occupancy = np.full((height, width), 127, dtype=np.uint8)
        
        # 1. Project Ground Points (Orthomosaic & Walkable)
        if ground_pts.size > 0:
            # Coordinates to pixel indices
            # X -> column, Y -> row (inverted)
            cols = ((ground_pts[:, 0] - min_x) * self.resolution).astype(int)
            rows = ((max_y - ground_pts[:, 1]) * self.resolution).astype(int)
            
            # Clip to be safe
            cols = np.clip(cols, 0, width - 1)
            rows = np.clip(rows, 0, height - 1)
            
            # Update orthomosaic (RGB to BGR for OpenCV)
            colors_bgr = (ground_clr[:, ::-1] * 255).astype(np.uint8)
            orthomosaic[rows, cols] = colors_bgr
            
            # Update occupancy
            occupancy[rows, cols] = 255
            
        # 2. Project Obstacle Points (Blocked)
        if obs_pts.size > 0:
            o_cols = ((obs_pts[:, 0] - min_x) * self.resolution).astype(int)
            o_rows = ((max_y - obs_pts[:, 1]) * self.resolution).astype(int)
            
            o_cols = np.clip(o_cols, 0, width - 1)
            o_rows = np.clip(o_rows, 0, height - 1)
            
            # Mark as blocked
            occupancy[o_rows, o_cols] = 0
            
        # Optional: Morphological cleanup to fill gaps
        kernel = np.ones((3, 3), np.uint8)
        orthomosaic = cv2.dilate(orthomosaic, kernel, iterations=1)
        occupancy = cv2.morphologyEx(occupancy, cv2.MORPH_CLOSE, kernel)
        
        # Bounding box for georeferencing
        bounds = {
            'min_x': min_x, 'max_x': max_x,
            'min_y': min_y, 'max_y': max_y
        }
        
        return orthomosaic, occupancy, bounds

    def save_maps(self, orthomosaic, occupancy, bounds, ortho_path="orthomosaic.png", occ_path="obstacle_occupancy.png"):
        """Saves the generated maps to disk along with their metric metadata."""
        cv2.imwrite(ortho_path, orthomosaic)
        # Convert grayscale occupancy to BGR so external scripts/editors can paint pure colors (like Blue) on it
        occupancy_bgr = cv2.cvtColor(occupancy, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(occ_path, occupancy_bgr)
        
        # Save map metadata
        meta_data = {
            "resolution": self.resolution,
            "bounds": bounds
        }
        
        out_dir = os.path.dirname(ortho_path)
        meta_path = os.path.join(out_dir, "map_metadata.json") if out_dir else "map_metadata.json"
        
        with open(meta_path, 'w') as f:
            json.dump(meta_data, f, indent=4)
            
        print(f"Maps saved to {ortho_path} and {occ_path}")
        print(f"Metadata saved to {meta_path}")
    def apply_parking_annotations(self, occupancy, json_path):
        """
        Reads parking space annotations from JSON and marks them on the occupancy map.
        - Parking Spots: Black (0, 0, 0)
        - Entrances/Exits: Blue (255, 0, 0)
        """
        if not os.path.exists(json_path):
            return occupancy
            
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        parking_spaces = data.get("parking_spaces", [])
        if not parking_spaces:
            return occupancy
            
        # Ensure occupancy is BGR color to support blue
        if len(occupancy.shape) == 2:
            occupancy = cv2.cvtColor(occupancy, cv2.COLOR_GRAY2BGR)
            
        # Draw each polygon on the occupancy map
        for space in parking_spaces:
            points_m = np.array(space["points"])
            poly_type = space.get("type", "parking_space")
            
            # Convert meters to pixels
            bounds = data.get("metadata", {}).get("bounds")
            res = data.get("metadata", {}).get("resolution", self.resolution)
            
            if not bounds:
                continue
                
            pts_px = []
            for pt in points_m:
                u = int(round((pt[0] - bounds["min_x"]) * res))
                v = int(round((bounds["max_y"] - pt[1]) * res))
                pts_px.append([u, v])
                
            pts_px = np.array(pts_px, np.int32).reshape((-1, 1, 2))
            
            # BGR: Pure Blue is (255, 0, 0), Black is (0, 0, 0)
            color = (255, 0, 0) if poly_type == "entrance_exit" else (0, 0, 0)


            
            # Fill with color
            cv2.fillPoly(occupancy, [pts_px], color)
            
            # Add a small buffer/thickness
            cv2.polylines(occupancy, [pts_px], isClosed=True, color=color, thickness=int(0.5 * res))
            
        return occupancy

    def apply_robot_route_annotations(self, occupancy, json_path):
        """
        Reads manual robot route annotations from JSON and marks them on the occupancy map.
        Marks drawn polylines in Blue (255, 0, 0) and ensures surrounding corridor is white (255, 255, 255).
        """
        if not os.path.exists(json_path):
            return occupancy

        with open(json_path, 'r') as f:
            data = json.load(f)

        lines = data.get("lines", [])
        if not lines:
            return occupancy

        if len(occupancy.shape) == 2:
            occupancy = cv2.cvtColor(occupancy, cv2.COLOR_GRAY2BGR)

        bounds = data.get("metadata", {}).get("bounds")
        res = data.get("metadata", {}).get("resolution", self.resolution)

        if not bounds:
            return occupancy

        lane_width_px = int(round(1.5 * res))  # Walkable lane around path
        blue_line_px = max(2, int(round(0.3 * res)))  # Blue core line

        for line_item in lines:
            points_m = np.array(line_item["points"])
            if len(points_m) < 2:
                continue

            pts_px = []
            for pt in points_m:
                u = int(round((pt[0] - bounds["min_x"]) * res))
                v = int(round((bounds["max_y"] - pt[1]) * res))
                pts_px.append([u, v])

            pts_px = np.array(pts_px, np.int32).reshape((-1, 1, 2))

            # First, ensure walkable corridor (white)
            cv2.polylines(occupancy, [pts_px], isClosed=False, color=(255, 255, 255), thickness=lane_width_px)
            # Then draw blue core path
            cv2.polylines(occupancy, [pts_px], isClosed=False, color=(255, 0, 0), thickness=blue_line_px)

        return occupancy


