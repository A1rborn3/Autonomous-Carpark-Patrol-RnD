import open3d as o3d
import numpy as np

class PointCloudProcessor:
    def __init__(self, voxel_size=0.05, obstacle_height=0.1, max_height=2.5):
        """
        :param voxel_size: Size of the voxel for downsampling (meters).
        :param obstacle_height: Minimum height above ground to be considered an obstacle.
        :param max_height: Maximum height of points to consider (clips ceiling/roof).
        """
        self.voxel_size = voxel_size
        self.obstacle_height = obstacle_height
        self.max_height = max_height

    def load_ply(self, filepath):
        """Loads a .ply file and ensures it has points and colors."""
        print(f"Loading .ply file: {filepath}")
        pcd = o3d.io.read_point_cloud(filepath)
        
        if not pcd.has_points():
            raise ValueError(f"Point cloud at {filepath} is empty or could not be loaded.")
            
        if not pcd.has_colors():
            print("Warning: Point cloud does not have vertex colors. Using default gray.")
            pcd.paint_uniform_color([0.5, 0.5, 0.5])
            
        return pcd

    def clean_and_downsample(self, pcd):
        """Removes statistical outliers and downsamples the cloud."""
        print(f"Original points: {len(pcd.points)}")
        
        # 1. Statistical Outlier Removal
        # nb_neighbors=20, std_ratio=2.0 is a common robust setting
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        print(f"Points after outlier removal: {len(pcd.points)}")
        
        # 2. Voxel Downsampling
        pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        print(f"Points after downsampling (voxel={self.voxel_size}m): {len(pcd.points)}")
        
        return pcd

    def segment_ground(self, pcd):
        """
        Uses RANSAC to find the ground plane.
        Returns the ground points, non-ground points, and the plane model.
        """
        print("Segmenting ground plane...")
        # distance_threshold=0.05 is usually good for flat surfaces with some noise
        # We can increase this slightly if the car park surface is very uneven
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.05,
                                                 ransac_n=3,
                                                 num_iterations=1000)
        
        [a, b, c, d] = plane_model
        print(f"Detected ground plane: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
        
        ground_pcd = pcd.select_by_index(inliers)
        non_ground_pcd = pcd.select_by_index(inliers, invert=True)
        
        return ground_pcd, non_ground_pcd, plane_model

    def align_to_ground(self, pcd, plane_model):
        """
        Rotates and translates the point cloud so the ground plane is at Z=0.
        """
        [a, b, c, d] = plane_model
        normal = np.array([a, b, c])
        normal /= np.linalg.norm(normal)
        
        # We want the normal to point to +Z [0, 0, 1]
        target = np.array([0, 0, 1])
        
        # If normal is roughly opposite to Z, flip it (assuming Z is generally "up")
        if np.dot(normal, target) < 0:
            normal = -normal
            d = -d

        # Axis of rotation
        v = np.cross(normal, target)
        s = np.linalg.norm(v)
        c_val = np.dot(normal, target)
        
        if s < 1e-6:
            rotation = np.eye(3)
        else:
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            rotation = np.eye(3) + vx + np.matmul(vx, vx) * ((1 - c_val) / (s**2))
            
        # Transform the cloud
        pcd.rotate(rotation, center=(0, 0, 0))
        
        # After rotation, the plane is a'x + b'y + c'z + d = 0. 
        # Since we rotated normal to [0,0,1], it's now 0x + 0y + 1z + d' = 0 -> z = -d'
        # We need to find the new d'. Let's just re-segment or use the fact that 
        # any point p on plane satisfies n.p + d = 0.
        # Let's just find the average Z of points that were inliers.
        # Better yet, translate by the offset of the plane from origin along Z.
        points = np.asarray(pcd.points)
        # The plane is now horizontal, so we just shift Z.
        # We'll use the median Z of the rotated points to find the ground level.
        # This is more robust than relying on the rotated 'd'.
        # Actually, let's just translate by -d if we were careful with normal.
        # But wait, pcd.rotate doesn't change 'd'. 
        # Let's just use the median Z of the points to be sure.
        z_offset = np.median(points[:, 2])
        pcd.translate((0, 0, -z_offset))
        
        return pcd

    def extract_obstacles(self, pcd):
        """
        Extracts obstacles and clips ceiling.
        Assumes pcd is ALREADY ALIGNED to Z=0.
        """
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        # Obstacles: 0.1m < Z < 2.5m
        obstacle_mask = (points[:, 2] > self.obstacle_height) & (points[:, 2] < self.max_height)
        # Ground: Z <= 0.1m (includes speed bumps and markings)
        ground_mask = (points[:, 2] <= self.obstacle_height)
        
        obs_pcd = o3d.geometry.PointCloud()
        obs_pcd.points = o3d.utility.Vector3dVector(points[obstacle_mask])
        obs_pcd.colors = o3d.utility.Vector3dVector(colors[obstacle_mask])
        
        ground_pcd = o3d.geometry.PointCloud()
        ground_pcd.points = o3d.utility.Vector3dVector(points[ground_mask])
        ground_pcd.colors = o3d.utility.Vector3dVector(colors[ground_mask])
        
        return ground_pcd, obs_pcd
