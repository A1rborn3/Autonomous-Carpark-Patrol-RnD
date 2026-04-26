import numpy as np
import open3d as o3d
import os

def create_test_ply(filename="test_cloud.ply"):
    # Ensure PLYInput directory exists
    if not os.path.exists("PLYInput"):
        os.makedirs("PLYInput")
    
    filepath = os.path.join("PLYInput", filename)
    print(f"Generating synthetic test cloud: {filepath}")
    
    # 1. Create Ground Plane (10m x 10m)
    x = np.linspace(-5, 5, 200)
    y = np.linspace(-5, 5, 200)
    xv, yv = np.meshgrid(x, y)
    
    # Ground points at Z=0
    ground_pts = np.vstack([xv.flatten(), yv.flatten(), np.zeros_like(xv.flatten())]).T
    
    # Add some "road markings" (white stripes)
    ground_colors = np.ones_like(ground_pts) * 0.3 # Dark gray road
    marking_mask = (np.abs(xv.flatten()) < 0.2) | (np.abs(yv.flatten() % 2) < 0.2)
    ground_colors[marking_mask] = [1.0, 1.0, 1.0] # White markings
    
    # 2. Add speed bumps (low height, should be walkable)
    bump_mask = (xv.flatten() > 2) & (xv.flatten() < 3)
    ground_pts[bump_mask, 2] = 0.05 # 5cm high
    
    # 3. Add Obstacles (Pillars)
    pillar1_center = [-3, -3]
    pillar2_center = [3, 3]
    
    def create_pillar(center, height=3.0, radius=0.3):
        z = np.linspace(0, height, 50)
        theta = np.linspace(0, 2*np.pi, 20)
        zv, tv = np.meshgrid(z, theta)
        px = center[0] + radius * np.cos(tv).flatten()
        py = center[1] + radius * np.sin(tv).flatten()
        pz = zv.flatten()
        pts = np.vstack([px, py, pz]).T
        clrs = np.tile([0.8, 0.2, 0.2], (len(pts), 1)) # Red pillars
        return pts, clrs

    p1_pts, p1_clrs = create_pillar(pillar1_center)
    p2_pts, p2_clrs = create_pillar(pillar2_center)
    
    # 4. Add Ceiling (at 3m)
    cx = np.linspace(-5, 5, 50)
    cy = np.linspace(-5, 5, 50)
    cxv, cyv = np.meshgrid(cx, cy)
    ceil_pts = np.vstack([cxv.flatten(), cyv.flatten(), np.full_like(cxv.flatten(), 3.0)]).T
    ceil_clrs = np.tile([0.5, 0.5, 0.8], (len(ceil_pts), 1)) # Blue ceiling
    
    # Combine everything
    all_pts = np.vstack([ground_pts, p1_pts, p2_pts, ceil_pts])
    all_clrs = np.vstack([ground_colors, p1_clrs, p2_clrs, ceil_clrs])
    
    # Create Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    pcd.colors = o3d.utility.Vector3dVector(all_clrs)
    
    o3d.io.write_point_cloud(filepath, pcd)
    print(f"Test cloud saved with {len(all_pts)} points.")

if __name__ == "__main__":
    create_test_ply()
