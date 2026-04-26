import numpy as np
import cv2

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

    def save_maps(self, orthomosaic, occupancy, ortho_path="orthomosaic.png", occ_path="occupancy.png"):
        """Saves the generated maps to disk."""
        cv2.imwrite(ortho_path, orthomosaic)
        # For occupancy, we might want to save it as a high-contrast image
        cv2.imwrite(occ_path, occupancy)
        print(f"Maps saved to {ortho_path} and {occ_path}")
