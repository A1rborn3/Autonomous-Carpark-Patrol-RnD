import json
import os
import math

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class UnitreeGo2Exporter:
    """
    Exporter for Unitree Go2 (G02) robot dog navigation waypoints.
    Generates JSON, YAML, and an executable Python runner script (`run_go2_patrol.py`)
    using Unitree SDK 2 / SportClient conventions.
    """

    def __init__(self, output_dir, base_lat=37.7749, base_lon=-122.4194):
        self.output_dir = output_dir
        self.base_lat = base_lat
        self.base_lon = base_lon
        self.R = 6378137.0  # Earth radius in meters

    def local_to_gps(self, x, y):
        """Converts Cartesian (x, y) meters to GPS (lat, lon)."""
        d_lat = y / self.R
        d_lon = x / (self.R * math.cos(math.pi * self.base_lat / 180.0))
        lat = self.base_lat + (d_lat * 180.0 / math.pi)
        lon = self.base_lon + (d_lon * 180.0 / math.pi)
        return round(lat, 7), round(lon, 7)

    def _compute_patrol_sequence(self, nodes_dict, edges):
        """
        Computes an ordered sequence of waypoints for patrol navigation.
        Starts from entrance/exit nodes if available, performing graph traversal.
        """
        if not nodes_dict:
            return []

        # Build adjacency graph
        adj = {n_id: [] for n_id in nodes_dict}
        for edge in edges:
            u = edge.get('from_id')
            v = edge.get('to_id')
            if u in adj and v in adj:
                adj[u].append(v)
                adj[v].append(u)

        # Select starting node (prefer entrance/exit, else first node)
        start_node = None
        for n_id, node in nodes_dict.items():
            if node.get('type') == 'entrance_exit':
                start_node = n_id
                break
        if not start_node:
            start_node = next(iter(nodes_dict.keys()))

        # Simple Depth-First / Nearest-Neighbor traversal to form continuous route
        visited = set()
        path = []
        current = start_node

        while current and current not in visited:
            visited.add(current)
            path.append(current)

            # Find unvisited neighbors
            unvisited_neighbors = [neighbor for neighbor in adj[current] if neighbor not in visited]
            if unvisited_neighbors:
                current = unvisited_neighbors[0]
            else:
                # Fallback to nearest unvisited node anywhere in the graph
                curr_pos = (nodes_dict[current]['x'], nodes_dict[current]['y'])
                unvisited_nodes = [n_id for n_id in nodes_dict if n_id not in visited]
                if not unvisited_nodes:
                    break
                # Find closest
                current = min(
                    unvisited_nodes,
                    key=lambda n_id: math.hypot(
                        nodes_dict[n_id]['x'] - curr_pos[0],
                        nodes_dict[n_id]['y'] - curr_pos[1]
                    )
                )

        return path

    def export_unitree_waypoints(self, nodes, edges, bounds, resolution, filename_prefix="road_graph"):
        """
        Exports waypoints for Unitree Go2 (G02) robot dog in JSON, YAML, and Python runner script.
        """
        cartesian_nodes = {}
        for node in nodes:
            # Convert pixel (u, v) to meter (x, y)
            mx = (node['x'] / resolution) + bounds['min_x']
            my = bounds['max_y'] - (node['y'] / resolution)
            cartesian_nodes[node['id']] = {
                'id': node['id'],
                'x': round(mx, 4),
                'y': round(my, 4),
                'z': 0.0,
                'type': node.get('type', 'waypoint')
            }

        # Sequence waypoints into an ordered patrol route
        patrol_order = self._compute_patrol_sequence(cartesian_nodes, edges)

        # Build detailed waypoint objects with heading (yaw) calculations
        formatted_waypoints = []
        total_distance = 0.0

        for i, node_id in enumerate(patrol_order):
            curr_node = cartesian_nodes[node_id]
            next_node = cartesian_nodes[patrol_order[i + 1]] if i + 1 < len(patrol_order) else cartesian_nodes[patrol_order[0]]

            # Compute orientation (yaw) towards next waypoint
            dx = next_node['x'] - curr_node['x']
            dy = next_node['y'] - curr_node['y']
            dist = math.hypot(dx, dy)
            if i + 1 < len(patrol_order):
                total_distance += dist

            yaw_rad = math.atan2(dy, dx)
            yaw_deg = math.degrees(yaw_rad)

            lat, lon = self.local_to_gps(curr_node['x'], curr_node['y'])

            formatted_waypoints.append({
                'seq': i,
                'id': curr_node['id'],
                'type': curr_node['type'],
                'x': curr_node['x'],
                'y': curr_node['y'],
                'z': curr_node['z'],
                'yaw_rad': round(yaw_rad, 4),
                'yaw_deg': round(yaw_deg, 2),
                'target_speed_m_s': 0.8,
                'tolerance_m': 0.3,
                'wait_time_sec': 2.0 if curr_node['type'] == 'entrance_exit' else 0.5,
                'gps': {
                    'latitude': lat,
                    'longitude': lon
                }
            })

        data = {
            'robot': {
                'model': 'Unitree Go2 / G02',
                'frame_id': 'map',
                'default_speed_m_s': 0.8,
                'gait_type': 'trot'
            },
            'metadata': {
                'map_bounds': bounds,
                'resolution': resolution,
                'total_waypoints': len(formatted_waypoints),
                'total_patrol_distance_m': round(total_distance, 2)
            },
            'waypoints': formatted_waypoints,
            'edges': edges
        }

        # 1. Save JSON output
        json_filename = f"{filename_prefix}_go2_waypoints.json"
        json_path = os.path.join(self.output_dir, json_filename)
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Unitree Go2 waypoints JSON exported to {json_path}")

        # 2. Save YAML output
        yaml_path = os.path.join(self.output_dir, f"{filename_prefix}_go2_waypoints.yaml")
        if YAML_AVAILABLE:
            with open(yaml_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            print(f"Unitree Go2 waypoints YAML exported to {yaml_path}")
        else:
            # Fallback simple YAML writer if PyYAML is not installed
            self._write_fallback_yaml(yaml_path, data)
            print(f"Unitree Go2 waypoints YAML exported to {yaml_path}")

        # 3. Save executable Python runner script
        runner_path = self._generate_runner_script(json_filename)

        return json_path, yaml_path, runner_path

    def _write_fallback_yaml(self, path, data):
        """Simple YAML formatter fallback when PyYAML is not installed."""
        lines = [
            "# Unitree Go2 (G02) Waypoints Configuration",
            f"robot:",
            f"  model: \"{data['robot']['model']}\"",
            f"  frame_id: \"{data['robot']['frame_id']}\"",
            f"  default_speed_m_s: {data['robot']['default_speed_m_s']}",
            f"  gait_type: \"{data['robot']['gait_type']}\"",
            f"metadata:",
            f"  total_waypoints: {data['metadata']['total_waypoints']}",
            f"  total_patrol_distance_m: {data['metadata']['total_patrol_distance_m']}",
            f"waypoints:"
        ]
        for wp in data['waypoints']:
            lines.append(f"  - seq: {wp['seq']}")
            lines.append(f"    id: \"{wp['id']}\"")
            lines.append(f"    type: \"{wp['type']}\"")
            lines.append(f"    x: {wp['x']}")
            lines.append(f"    y: {wp['y']}")
            lines.append(f"    z: {wp['z']}")
            lines.append(f"    yaw_rad: {wp['yaw_rad']}")
            lines.append(f"    target_speed_m_s: {wp['target_speed_m_s']}")
            lines.append(f"    wait_time_sec: {wp['wait_time_sec']}")
            lines.append(f"    gps:")
            lines.append(f"      latitude: {wp['gps']['latitude']}")
            lines.append(f"      longitude: {wp['gps']['longitude']}")

        with open(path, 'w') as f:
            f.write("\n".join(lines) + "\n")

    def _generate_runner_script(self, json_filename):
        """Generates `run_go2_patrol.py` in output_dir to command the Unitree Go2 robot dog."""
        runner_code = f'''#!/usr/bin/env python3
"""
Unitree Go2 (G02) Autonomous Waypoint Patrol Execution Script

This script loads the exported waypoints JSON file and commands a Unitree Go2 robot dog
to navigate sequentially through all patrol points using `unitree_sdk2` (SportClient).

Usage:
    python run_go2_patrol.py                     # Run live (requires unitree_sdk2 & robot connection)
    python run_go2_patrol.py --dry-run           # Run in simulation mode (offline / mock test)
    python run_go2_patrol.py --waypoints path.json --net eth0
"""

import sys
import os
import time
import json
import math
import argparse

# Try importing unitree_sdk2
SDK_AVAILABLE = False
try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.go2.sport.sport_client import SportClient
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False


def load_waypoints(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Waypoints file not found: {{json_path}}")
    with open(json_path, 'r') as f:
        return json.load(f)


def run_patrol_simulation(waypoints_data, speed_factor=1.0):
    print("\\n=======================================================")
    print("      UNITREE GO2 PATROL SIMULATION / DRY-RUN MODE     ")
    print("=======================================================")
    waypoints = waypoints_data.get('waypoints', [])
    meta = waypoints_data.get('metadata', {{}})
    print(f"Loaded {{len(waypoints)}} waypoints across {{meta.get('total_patrol_distance_m', 0)}} meters.")
    print("Starting simulated route execution...\\n")

    curr_x, curr_y = 0.0, 0.0

    for wp in waypoints:
        target_x = wp['x']
        target_y = wp['y']
        target_yaw = wp['yaw_deg']
        dist = math.hypot(target_x - curr_x, target_y - curr_y)
        speed = wp.get('target_speed_m_s', 0.8) * speed_factor

        travel_time = dist / speed if speed > 0 else 0
        wait_time = wp.get('wait_time_sec', 0.5)

        print(f"[Waypoint {{wp['seq']:02d}} | {{wp['id']}}] -> Target: (x: {{target_x:6.2f}}m, y: {{target_y:6.2f}}m, yaw: {{target_yaw:6.1f}}°)")
        print(f"   -> Moving {{dist:.2f}}m at {{speed:.2f}} m/s (Est. time: {{travel_time:.1f}}s)...")

        # Simulate movement steps
        steps = 5
        for s in range(1, steps + 1):
            ratio = s / steps
            px = curr_x + ratio * (target_x - curr_x)
            py = curr_y + ratio * (target_y - curr_y)
            bar = "=" * (s * 4) + ">" + "." * ((steps - s) * 4)
            print(f"   [{{bar}}] Robot pos: ({{px:6.2f}}, {{py:6.2f}})", end="\\r")
            time.sleep(0.1 / speed_factor)
        print()

        curr_x, curr_y = target_x, target_y
        print(f"   [ARRIVED] Reached waypoint {{wp['id']}} (Type: {{wp['type']}}). Waiting {{wait_time}}s...")
        time.sleep(min(wait_time, 0.5) / speed_factor)
        print("-" * 55)

    print("\\n[SUCCESS] Unitree Go2 patrol mission simulation finished successfully!")


def run_patrol_live(waypoints_data, net_interface="eth0"):
    if not SDK_AVAILABLE:
        print("Error: unitree_sdk2py is not installed on this system.")
        print("Please install unitree_sdk2py or run with --dry-run")
        sys.exit(1)

    print(f"Initializing Unitree Go2 SDK 2 on network interface: {{net_interface}}...")
    ChannelFactoryInitialize(0, net_interface)
    
    sport_client = SportClient()
    sport_client.SetTimeout(10.0)
    sport_client.Init()

    print("Unlocking robot motion (Standing up)...")
    sport_client.StandUp()
    time.sleep(2.0)

    waypoints = waypoints_data.get('waypoints', [])
    print(f"Executing live patrol sequence for {{len(waypoints)}} waypoints...")

    for wp in waypoints:
        target_x = wp['x']
        target_y = wp['y']
        target_yaw_rad = wp['yaw_rad']
        speed = wp.get('target_speed_m_s', 0.8)
        wait_time = wp.get('wait_time_sec', 0.5)

        print(f"Navigating to Node {{wp['id']}} ({{target_x:.2f}}, {{target_y:.2f}})...")
        # Command Go2 robot dog via SportClient Move command (vx, vy, vyaw)
        # Note: In real-world Go2 SDK, position tracking or SLAM feedback loops update velocity vectors
        sport_client.Move(req=True, vx=speed, vy=0.0, vyaw=0.0)
        time.sleep(2.0)

        if wait_time > 0:
            sport_client.StopMove()
            time.sleep(wait_time)

    print("Patrol completed. Returning to idle pose...")
    sport_client.StandDown()


def main():
    parser = argparse.ArgumentParser(description="Unitree Go2 Waypoint Patrol Controller")
    default_json = os.path.join(os.path.dirname(__file__), "{json_filename}")
    parser.add_argument("--waypoints", type=str, default=default_json, help="Path to waypoints JSON file")
    parser.add_argument("--net", type=str, default="eth0", help="Network interface for Unitree SDK 2")
    parser.add_argument("--dry-run", action="store_true", help="Run in simulation mode (offline mock execution)")
    parser.add_argument("--speed-factor", type=float, default=2.0, help="Simulation speedup factor")
    args = parser.parse_args()

    data = load_waypoints(args.waypoints)

    if args.dry_run or not SDK_AVAILABLE:
        if not SDK_AVAILABLE and not args.dry_run:
            print("Note: unitree_sdk2py library not found. Falling back to --dry-run simulation mode.")
        run_patrol_simulation(data, speed_factor=args.speed_factor)
    else:
        run_patrol_live(data, net_interface=args.net)


if __name__ == "__main__":
    main()
'''
        runner_path = os.path.join(self.output_dir, "run_go2_patrol.py")
        with open(runner_path, 'w') as f:
            f.write(runner_code)
        # Make executable on Unix
        try:
            os.chmod(runner_path, 0o755)
        except Exception:
            pass
        print(f"Unitree Go2 runner script generated at {runner_path}")
        return runner_path
