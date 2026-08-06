import argparse
import math
import time
import numpy as np
import open3d as o3d

# Unitree SDK2 Python imports
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient


class Relocalizer3D:
    def __init__(self, ply_path: str, voxel_size: float = 0.1):
        """Loads the .ply map and sets up Open3D alignment structures."""
        self.voxel_size = voxel_size
        print(f"[Map Loader] Loading target 3D map: {ply_path}")
        
        # Load the complete .ply map
        self.map_pcd = o3d.io.read_point_cloud(ply_path)
        if not self.map_pcd.has_points():
            raise RuntimeError(f"Failed to read points from {ply_path}")

        # Downsample and estimate surface normals for Point-to-Plane ICP
        self.map_down = self.map_pcd.voxel_down_sample(self.voxel_size)
        self.map_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.voxel_size * 2.0, max_nn=30
            )
        )
        print(f"[Map Loader] Ready! Map points: {len(self.map_pcd.points)} raw -> {len(self.map_down.points)} downsampled.")

    def preprocess_scan(self, points_3d: np.ndarray) -> o3d.geometry.PointCloud:
        """Converts live LiDAR numpy data into a processed Open3D point cloud."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)

        # Filter self-body reflections (<0.3m) and far noisy returns (>25.0m)
        distances = np.linalg_norm(points_3d, axis=1)
        valid_mask = (distances > 0.3) & (distances < 25.0)
        pcd = pcd.select_by_index(np.where(valid_mask)[0])

        # Downsample live cloud and compute normals
        pcd_down = pcd.voxel_down_sample(self.voxel_size)
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.voxel_size * 2.0, max_nn=30
            )
        )
        return pcd_down

    def find_pose_icp(self, live_pcd: o3d.geometry.PointCloud) -> tuple[np.ndarray, float]:
        """Aligns live 3D scan against .ply map using Point-to-Plane ICP registration."""
        max_distance_threshold = self.voxel_size * 3.0
        initial_guess = np.identity(4)

        icp_result = o3d.pipelines.registration.registration_icp(
            live_pcd,
            self.map_down,
            max_distance_threshold,
            initial_guess,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )

        # Returns 4x4 homogenous matrix and fitness score (percentage of aligned points, 0.0 to 1.0)
        return icp_result.transformation, icp_result.fitness


class Go2RelocalizationNode:
    def __init__(self, args):
        self.args = args
        self.relocalizer = Relocalizer3D(args.ply_map, voxel_size=args.voxel_size)
        self.latest_scan_buffer = []

        # Initialize Unitree DDS Network Interface
        print(f"[DDS] Initializing channel on interface: {args.net}")
        ChannelFactoryInitialize(0, args.net)

        # Initialize High-level Motion Client
        self.sport_client = SportClient()
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()

    def lidar_callback(self, msg):
        """Captures live LiDAR point cloud arrays from DDS channel."""
        # Unpack raw binary payload into Nx3 coordinate points
        if hasattr(msg, "data") and len(msg.data) > 0:
            raw_floats = np.frombuffer(msg.data, dtype=np.float32)
            if len(raw_floats) % 3 == 0:
                cloud_data = raw_floats.reshape(-1, 3)
                self.latest_scan_buffer.append(cloud_data)

    def execute_relocalization(self):
        # 1. Subscribe to real-time LiDAR topic
        print(f"[DDS] Subscribing to LiDAR topic: {self.args.lidar_topic}")
        ChannelSubscriber(self.args.lidar_topic, self.args.lidar_msg_type).Init(self.lidar_callback, 10)
        time.sleep(1.0)

        # 2. Stand up robot
        print("\n[Robot Action] Standing up...")
        self.sport_client.StandUp()
        time.sleep(2.0)

        # 3. Perform 360-degree spin while buffering LiDAR frames
        print(f"[Robot Action] Spinning 360 degrees for {self.args.spin_time}s to scan environment...")
        self.latest_scan_buffer.clear()
        start_time = time.time()

        while time.time() - start_time < self.args.spin_time:
            # Command slow rotation (0.5 rad/s)
            self.sport_client.Move(0.0, 0.0, self.args.spin_rate)
            time.sleep(0.1)

        # Stop turning
        self.sport_client.Move(0.0, 0.0, 0.0)
        time.sleep(0.5)

        # 4. Check received points
        if not self.latest_scan_buffer:
            print("\n[ERROR] No LiDAR point cloud data received from the robot!")
            self.sport_client.StandDown()
            return

        # Combine all rotation frames into one dense 360-degree point cloud
        merged_points = np.vstack(self.latest_scan_buffer)
        print(f"[3D Processing] Aggregated {len(merged_points)} total 3D points during spin.")

        # 5. Preprocess live scan & align against .ply map
        live_pcd = self.relocalizer.preprocess_scan(merged_points)
        print("[3D Processing] Aligning live scan against .ply map structure...")
        transform_matrix, fitness = self.relocalizer.find_pose_icp(live_pcd)

        # 6. Extract position (X, Y, Z) and Orientation (Yaw)
        x_pos = transform_matrix[0, 3]
        y_pos = transform_matrix[1, 3]
        z_pos = transform_matrix[2, 3]

        # Extract yaw angle from top-left 2x2 rotation submatrix
        r11, r10 = transform_matrix[0, 0], transform_matrix[1, 0]
        yaw_rad = math.atan2(r10, r11)
        yaw_deg = math.degrees(yaw_rad)

        # 7. Output results to screen
        print("\n" + "=" * 50)
        print("           3D RELOCALIZATION RESULTS           ")
        print("=" * 50)
        print(f" Confidence Score (Fitness): {fitness:.4f} / 1.0000")
        print(f" Position X              : {x_pos:.3f} meters")
        print(f" Position Y              : {y_pos:.3f} meters")
        print(f" Height Z                : {z_pos:.3f} meters")
        print(f" Yaw Orientation         : {yaw_deg:.2f}° ({yaw_rad:.3f} rad)")
        print("=" * 50)

        if fitness < self.args.min_fitness:
            print(f"[WARNING] Confidence {fitness:.2f} is below safety threshold ({self.args.min_fitness}). Check map match.")
        else:
            print("[SUCCESS] Robot locked onto 3D .ply map successfully!")

        # 8. Stand down and finish
        print("\n[Robot Action] Sitting down and exiting...")
        self.sport_client.StandDown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Go2 3D PLY Relocalization Tool")
    parser.add_argument("--ply-map", type=str, required=True, help="Path to your carpark .ply file")
    parser.add_argument("--net", type=str, default="eth0", help="Network interface (e.g., eth0 or wlan0)")
    parser.add_argument("--lidar-topic", type=str, default="rt/utlidar/cloud", help="Unitree LiDAR DDS topic")
    parser.add_argument("--lidar-msg-type", type=str, default="unitree_sdk2py.idl.unitree_go.msg.dds_.PointCloud2_", help="DDS message type")
    parser.add_argument("--voxel-size", type=float, default=0.1, help="Downsampling voxel resolution (meters)")
    parser.add_argument("--spin-time", type=float, default=8.0, help="Seconds spent spinning")
    parser.add_argument("--spin-rate", type=float, default=0.5, help="Spin speed (radians/sec)")
    parser.add_argument("--min-fitness", type=float, default=0.35, help="Minimum matching confidence")

    args = parser.parse_args()
    node = Go2RelocalizationNode(args)
    node.execute_relocalization()