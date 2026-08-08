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

    def __init__(self, output_dir):
        self.output_dir = output_dir

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
                'wait_time_sec': 2.0 if curr_node['type'] == 'entrance_exit' else 0.5
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

        # 4. Save visual final route output image(s)
        self.export_route_image(nodes, edges, bounds, resolution, filename_prefix)

        return json_path, yaml_path, runner_path

    def export_route_image(self, nodes, edges, bounds, resolution, filename_prefix="road_graph"):
        """
        Generates and saves visual output image(s) of the final route overlaid on the orthomosaic/map image.
        Saves:
          - final_route.png
          - {filename_prefix}_final_route.png
          - final_graph.png
        into self.output_dir.
        """
        import cv2
        import numpy as np

        if not nodes:
            return None

        os.makedirs(self.output_dir, exist_ok=True)

        # 1. Locate background map image
        parent_dir = os.path.dirname(self.output_dir)
        candidate_bg_paths = [
            os.path.join(parent_dir, "orthomosaic.png"),
            os.path.join(parent_dir, "merged_occupancy.png"),
            os.path.join(parent_dir, "obstacle_occupancy.png"),
            os.path.join(self.output_dir, "orthomosaic.png"),
            os.path.join(self.output_dir, "merged_occupancy.png"),
            os.path.join(self.output_dir, "obstacle_occupancy.png"),
        ]

        bg_img = None
        for path in candidate_bg_paths:
            if os.path.exists(path):
                loaded = cv2.imread(path)
                if loaded is not None:
                    bg_img = loaded
                    break

        min_x = bounds.get('min_x', 0) if isinstance(bounds, dict) else 0
        max_y = bounds.get('max_y', 100) if isinstance(bounds, dict) else 100
        min_y = bounds.get('min_y', 0) if isinstance(bounds, dict) else 0
        max_x = bounds.get('max_x', 100) if isinstance(bounds, dict) else 100
        res = resolution if resolution and resolution > 0 else 70

        # Build node pixel dictionary: id -> {'px', 'py', 'type', 'id'}
        pixel_nodes = {}
        for n in nodes:
            nx, ny = n['x'], n['y']
            # Check if coordinates are in meters (within bounding box range)
            if min_x <= nx <= max_x and min_y <= ny <= max_y and (max_x - min_x) < 500:
                px = int(round((nx - min_x) * res))
                py = int(round((max_y - ny) * res))
            else:
                px = int(round(nx))
                py = int(round(ny))
            pixel_nodes[n['id']] = {
                'px': px,
                'py': py,
                'type': n.get('type', 'waypoint'),
                'id': n['id']
            }

        # If no background image found, create canvas based on node pixel coordinates
        if bg_img is None:
            all_px = [p['px'] for p in pixel_nodes.values()]
            all_py = [p['py'] for p in pixel_nodes.values()]
            max_w = max(all_px + [1000]) + 100
            max_h = max(all_py + [1000]) + 100
            bg_img = np.full((max_h, max_w, 3), 240, dtype=np.uint8)
        elif len(bg_img.shape) == 2:
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_GRAY2BGR)

        vis_img = bg_img.copy()
        h, w = vis_img.shape[:2]

        # 2. Draw Parking Spaces Overlay if present in parent_dir
        park_json_path = os.path.join(parent_dir, "orthomosaic_parking_spaces.json")
        if not os.path.exists(park_json_path):
            if os.path.exists(parent_dir):
                for f in os.listdir(parent_dir):
                    if f.endswith("_parking_spaces.json"):
                        park_json_path = os.path.join(parent_dir, f)
                        break

        if os.path.exists(park_json_path):
            try:
                with open(park_json_path, 'r') as pf:
                    pdata = json.load(pf)
                    is_meters = pdata.get("coordinate_system") == "meters"
                    for poly in pdata.get("parking_spaces", []):
                        poly_pts = []
                        for pt in poly["points"]:
                            if is_meters:
                                x_p = int(round((pt[0] - min_x) * res))
                                y_p = int(round((max_y - pt[1]) * res))
                            else:
                                x_p, y_p = int(pt[0]), int(pt[1])
                            poly_pts.append([x_p, y_p])
                        if len(poly_pts) >= 3:
                            pts_arr = np.array(poly_pts, np.int32).reshape((-1, 1, 2))
                            color = (255, 150, 0) if poly.get('type') == 'entrance_exit' else (0, 200, 100)
                            cv2.polylines(vis_img, [pts_arr], True, color, 1, cv2.LINE_AA)
            except Exception as pe:
                print(f"Note: could not overlay parking spaces: {pe}")

        # 3. Compute patrol sequence for ordered line drawing
        patrol_path = self._compute_patrol_sequence(
            {n_id: {'x': n['px'], 'y': n['py'], 'type': n['type']} for n_id, n in pixel_nodes.items()},
            edges
        )

        lines_to_draw = []
        if patrol_path and len(patrol_path) > 1:
            for i in range(len(patrol_path) - 1):
                u_id, v_id = patrol_path[i], patrol_path[i + 1]
                if u_id in pixel_nodes and v_id in pixel_nodes:
                    lines_to_draw.append((pixel_nodes[u_id], pixel_nodes[v_id], i))
        else:
            for i, e in enumerate(edges):
                u_id, v_id = e.get('from_id'), e.get('to_id')
                if u_id in pixel_nodes and v_id in pixel_nodes:
                    lines_to_draw.append((pixel_nodes[u_id], pixel_nodes[v_id], i))

        total_distance_px = 0.0
        for p1, p2, seq_idx in lines_to_draw:
            pt1 = (p1['px'], p1['py'])
            pt2 = (p2['px'], p2['py'])
            dist_px = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
            total_distance_px += dist_px

            # Thick cyan line for patrol path
            cv2.line(vis_img, pt1, pt2, (255, 220, 0), 3, cv2.LINE_AA)

            # Draw direction arrow along segment
            if dist_px > 15:
                mid_x = int(0.6 * pt2[0] + 0.4 * pt1[0])
                mid_y = int(0.6 * pt2[1] + 0.4 * pt1[1])
                angle = math.atan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
                arrow_len = 12
                p_arrow1 = (
                    int(mid_x - arrow_len * math.cos(angle - math.pi / 6)),
                    int(mid_y - arrow_len * math.sin(angle - math.pi / 6))
                )
                p_arrow2 = (
                    int(mid_x - arrow_len * math.cos(angle + math.pi / 6)),
                    int(mid_y - arrow_len * math.sin(angle + math.pi / 6))
                )
                cv2.line(vis_img, (mid_x, mid_y), p_arrow1, (0, 140, 255), 2, cv2.LINE_AA)
                cv2.line(vis_img, (mid_x, mid_y), p_arrow2, (0, 140, 255), 2, cv2.LINE_AA)

        total_distance_m = total_distance_px / res if res > 0 else 0.0

        # 4. Draw Waypoint Nodes & Labels
        seq_map = {n_id: i for i, n_id in enumerate(patrol_path)} if patrol_path else {}
        for n_id, n in pixel_nodes.items():
            cx, cy = n['px'], n['py']
            ntype = n['type']
            seq = seq_map.get(n_id, None)
            label = f"WP {seq}" if seq is not None else str(n_id)

            if ntype == 'entrance_exit':
                cv2.circle(vis_img, (cx, cy), 8, (0, 230, 0), -1, cv2.LINE_AA)
                cv2.circle(vis_img, (cx, cy), 8, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.circle(vis_img, (cx, cy), 6, (0, 215, 255), -1, cv2.LINE_AA)
                cv2.circle(vis_img, (cx, cy), 6, (0, 0, 0), 1, cv2.LINE_AA)

            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.45
            thick = 1
            offset_x, offset_y = cx + 8, cy - 8

            cv2.putText(vis_img, label, (offset_x + 1, offset_y + 1), font, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
            cv2.putText(vis_img, label, (offset_x, offset_y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

        # 5. Draw Header Banner
        banner_h = 50
        overlay = vis_img.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, vis_img, 0.25, 0, vis_img)

        folder_label = os.path.basename(self.output_dir)
        title_text = f"PATROL ROUTE MAP ({folder_label.upper()})"
        subtitle_text = f"Waypoints: {len(pixel_nodes)}  |  Est. Distance: {total_distance_m:.1f}m  |  Resolution: {res} px/m"

        cv2.putText(vis_img, title_text, (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis_img, subtitle_text, (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

        # Save output images
        route_img_path = os.path.join(self.output_dir, "final_route.png")
        prefix_route_img_path = os.path.join(self.output_dir, f"{filename_prefix}_final_route.png")
        graph_img_path = os.path.join(self.output_dir, "final_graph.png")

        cv2.imwrite(route_img_path, vis_img)
        cv2.imwrite(prefix_route_img_path, vis_img)
        cv2.imwrite(graph_img_path, vis_img)

        print(f"Final route output image saved to {route_img_path}")
        return route_img_path

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

        with open(path, 'w') as f:
            f.write("\n".join(lines) + "\n")

    def _generate_runner_script(self, json_filename):
        """Generates `run_go2_patrol.py` in output_dir to command the Unitree Go2 robot dog.

        Navigation is closed-loop, driven by live pose feedback (position + yaw)
        from the robot's SportModeState, with proportional heading control, a
        real distance-based arrival check, a ~20 Hz-polled kill switch, and
        onboard obstacle avoidance via ObstaclesAvoidClient (mirroring
        Go2KeyboardController.py conventions).
        """
        runner_code = '''#!/usr/bin/env python3
"""
Unitree Go2 (G02) Autonomous Waypoint Patrol Execution Script

Loads the exported waypoints JSON file and commands a Unitree Go2 robot dog to
navigate sequentially through all patrol points using `unitree_sdk2` (SportClient).

Navigation is closed-loop, driven by live pose feedback (position + yaw) from
the robot's SportModeState:
  * Proportional heading control (vyaw) turns the robot toward each target
    before/while driving forward, and the waypoint's yaw_deg is honored on
    arrival.
  * "Arrived" is a real distance check against the waypoint (with tolerance),
    not a fixed timer.
  * The kill switch is polled every control tick (~20 Hz), so a spacebar
    press stops the robot within ~50ms.
  * Obstacle avoidance is enabled the same way as Go2KeyboardController.py:
    UseRemoteCommandFromApi(True) + SwitchSet(True) after standing, and all
    movement during navigation goes through ObstaclesAvoidClient.Move()
    instead of SportClient.Move() while avoidance is active, so the robot's
    onboard avoidance can override/shape commanded velocities around
    obstacles it detects. Cleanly disabled again in stand_down().

Usage:
    python run_go2_patrol.py                     # live (requires unitree_sdk2 & robot)
    python run_go2_patrol.py --dry-run            # simulation mode (offline / mock test)
    python run_go2_patrol.py --waypoints path.json --net eth0
"""

import sys
import os
import time
import json
import math
import argparse
import select
import termios
import tty

SDK_AVAILABLE = False
try:
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize,
        ChannelPublisher,
        ChannelSubscriber,
    )
    from unitree_sdk2py.go2.sport.sport_client import SportClient
    from unitree_sdk2py.go2.obstacles_avoid.obstacles_avoid_client import ObstaclesAvoidClient
    # SportModeState carries the robot's estimated position/yaw (odometry).
    # Go2 uses the unitree_go idl (G1/H1-2 use unitree_hg instead).
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False


# ---------- Tunables ----------
ARRIVAL_TOLERANCE_M = 0.15       # how close counts as "reached" the waypoint
YAW_TOLERANCE_DEG = 8.0          # how close counts as "facing" the target yaw
MAX_LINEAR_SPEED = 1.0           # m/s safety cap, overrides waypoint speed if higher
MAX_YAW_RATE = 1.0               # rad/s cap for turning
CONTROL_HZ = 20.0                # control loop rate
HEADING_KP = 2.0                 # proportional gain: rad/s per rad of heading error
TURN_IN_PLACE_THRESHOLD_DEG = 30 # if heading error exceeds this, stop and turn first


def load_waypoints(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Waypoints file not found: {json_path}")
    with open(json_path, "r") as f:
        return json.load(f)


def _check_for_space_kill():
    """Non-blocking single-key check. Returns True if space was pressed."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        if select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            if key == " ":
                return True
    except (termios.error, OSError, EOFError):
        return False
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return False


def angle_diff_rad(target_rad, current_rad):
    """Shortest signed angular difference, wrapped to [-pi, pi]."""
    d = target_rad - current_rad
    return (d + math.pi) % (2 * math.pi) - math.pi


class Go2PatrolController:
    def __init__(self, network_interface: str = "eth0"):
        if not SDK_AVAILABLE:
            raise RuntimeError("unitree_sdk2py not installed; cannot run live mode")

        ChannelFactoryInitialize(0, network_interface)

        self.sport_client = SportClient()
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()

        self.obstacle_client = ObstaclesAvoidClient()
        self.obstacle_client.Init()
        self.obstacle_avoidance_enabled = False  # actually enabled later, once standing

        # Robot pose, updated by the state subscriber callback.
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_yaw = 0.0  # radians
        self._pose_lock_ready = False

        self._state_sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        self._state_sub.Init(self._on_state, 10)

        self.is_standing = False

        # Wait briefly for first state message so we're not navigating blind.
        t0 = time.time()
        while not self._pose_lock_ready and time.time() - t0 < 3.0:
            time.sleep(0.05)
        if not self._pose_lock_ready:
            print("Warning: no pose feedback received yet; navigation will be unreliable "
                  "until state messages arrive.")

    def _on_state(self, msg):
        # Field names depend on SDK version — check SportModeState_ definition.
        # Typically something like msg.position = [x, y, z], msg.imu_state.rpy = [r,p,y]
        self.pose_x = msg.position[0]
        self.pose_y = msg.position[1]
        self.pose_yaw = msg.imu_state.rpy[2]
        self._pose_lock_ready = True

    def enable_obstacle_avoidance(self):
        self.obstacle_client.UseRemoteCommandFromApi(True)
        self.obstacle_client.SwitchSet(True)
        self.obstacle_avoidance_enabled = True
        time.sleep(0.5)

    def disable_obstacle_avoidance(self):
        self.obstacle_client.SwitchSet(False)
        self.obstacle_client.UseRemoteCommandFromApi(False)
        self.obstacle_avoidance_enabled = False

    def _move(self, vx, vy, vyaw):
        """Route through the obstacle-avoidance client when enabled, same as
        the keyboard controller: ObstaclesAvoidClient.Move() lets the robot's
        onboard avoidance modify/block the commanded velocity around obstacles."""
        if self.obstacle_avoidance_enabled:
            self.obstacle_client.Move(vx, vy, vyaw)
        else:
            self.sport_client.Move(vx=vx, vy=vy, vyaw=vyaw)

    def _stop(self):
        """Zero velocity, routed through whichever client currently has
        authority over locomotion (mirrors _move()). sport_client.StopMove()
        alone does nothing while ObstaclesAvoidClient is the active command
        source, which is why the robot would previously sail through
        waypoints instead of stopping."""
        if self.obstacle_avoidance_enabled:
            self.obstacle_client.Move(0.0, 0.0, 0.0)
        else:
            self.sport_client.StopMove()

    def stand_up(self):
        self.sport_client.StopMove()
        time.sleep(0.2)
        self.sport_client.StandUp()
        self.is_standing = True
        time.sleep(2.5)
        self.sport_client.ClassicWalk(True)
        self.enable_obstacle_avoidance()

    def stand_down(self):
        self._stop()
        self.disable_obstacle_avoidance()
        #self.sport_client.Euler(0.0, 0.0, 0.0)
        time.sleep(0.2)
        self.sport_client.StandDown()
        self.is_standing = False
        time.sleep(1.5)

    def navigate_to_waypoint(self, wp):
        """Closed-loop drive toward a single waypoint. Returns False if killed."""
        target_x = wp["x"]
        target_y = wp["y"]
        target_yaw_deg = wp.get("yaw_deg")
        max_speed = min(wp.get("target_speed_m_s", 0.8), MAX_LINEAR_SPEED)

        period = 1.0 / CONTROL_HZ

        while True:
            if _check_for_space_kill():
                print("\\nKill switch pressed. Stopping.")
                self._stop()
                self.stand_down()
                return False

            dx = target_x - self.pose_x
            dy = target_y - self.pose_y
            dist = math.hypot(dx, dy)

            if dist <= ARRIVAL_TOLERANCE_M:
                break

            bearing_to_target = math.atan2(dy, dx)
            heading_error = angle_diff_rad(bearing_to_target, self.pose_yaw)
            heading_error_deg = math.degrees(heading_error)

            vyaw = max(-MAX_YAW_RATE, min(MAX_YAW_RATE, HEADING_KP * heading_error))

            if abs(heading_error_deg) > TURN_IN_PLACE_THRESHOLD_DEG:
                # Facing badly wrong direction: turn in place before driving forward.
                vx = 0.0
            else:
                # Slow down as we approach, and while still correcting heading.
                vx = max_speed * min(1.0, dist / 0.5)

            self._move(vx, 0.0, vyaw)
            print(f"move command sent {vx}")
            time.sleep(period)

        self._stop()

        # Rotate to the requested final yaw, if the waypoint specifies one.
        if target_yaw_deg is not None:
            target_yaw_rad = math.radians(target_yaw_deg)
            while True:
                if _check_for_space_kill():
                    print("\\nKill switch pressed. Stopping.")
                    self._stop()
                    self.stand_down()
                    return False
                err = angle_diff_rad(target_yaw_rad, self.pose_yaw)
                if abs(math.degrees(err)) <= YAW_TOLERANCE_DEG:
                    break
                vyaw = max(-MAX_YAW_RATE, min(MAX_YAW_RATE, HEADING_KP * err))
                self._move(0.0, 0.0, vyaw)
                time.sleep(period)
            self._stop()

        return True

    def run_patrol(self, waypoints_data):
        try:
            waypoints = waypoints_data.get("waypoints", [])
            print(f"Executing live patrol sequence for {len(waypoints)} waypoints...")

            if not self.is_standing:
                print("Standing up...")
                self.stand_up()

            for wp in waypoints:
                print(f"Navigating to Node {wp['id']} "
                    f"({wp['x']:.2f}, {wp['y']:.2f}, yaw={wp.get('yaw_deg', 0):.1f}°)...")
                ok = self.navigate_to_waypoint(wp)
                if not ok:
                    return  # kill switch was hit

                wait_time = wp.get("wait_time_sec", 0.5)
                if wait_time > 0:
                    t0 = time.time()
                    while time.time() - t0 < wait_time:
                        if _check_for_space_kill():
                            print("\\nKill switch pressed. Stopping.")
                            self._stop()
                            self.stand_down()
                            return
                        time.sleep(1.0 / CONTROL_HZ)
            print("Patrol completed. Returning to idle pose...")
            self.stand_down()
        finally:
            try:
                print("Shutting down safely...")
                if self.is_standing:
                    self.stand_down()
            except Exception as e:
                # If the robot disconnects during shutdown, we just print it and exit cleanly
                print(f"\\nNote: Shutdown command interrupted ({e}). Robot may need manual sit")

        


def run_patrol_simulation(waypoints_data, speed_factor=1.0):
    print("\\n=======================================================")
    print("      UNITREE GO2 PATROL SIMULATION / DRY-RUN MODE     ")
    print("=======================================================")
    waypoints = waypoints_data.get("waypoints", [])
    meta = waypoints_data.get("metadata", {})
    print(f"Loaded {len(waypoints)} waypoints across {meta.get('total_patrol_distance_m', 0)} meters.")
    print("Starting simulated route execution...\\n")

    curr_x, curr_y = 0.0, 0.0

    for wp in waypoints:
        target_x = wp["x"]
        target_y = wp["y"]
        target_yaw = wp["yaw_deg"]
        dist = math.hypot(target_x - curr_x, target_y - curr_y)
        speed = wp.get("target_speed_m_s", 0.8) * speed_factor
        travel_time = dist / speed if speed > 0 else 0
        wait_time = wp.get("wait_time_sec", 0.5)

        print(f"[Waypoint {wp['seq']:02d} | {wp['id']}] -> Target: "
              f"(x: {target_x:6.2f}m, y: {target_y:6.2f}m, yaw: {target_yaw:6.1f}°)")
        print(f"   -> Moving {dist:.2f}m at {speed:.2f} m/s (Est. time: {travel_time:.1f}s)...")

        steps = 5
        for s in range(1, steps + 1):
            ratio = s / steps
            px = curr_x + ratio * (target_x - curr_x)
            py = curr_y + ratio * (target_y - curr_y)
            bar = "=" * (s * 4) + ">" + "." * ((steps - s) * 4)
            print(f"   [{bar}] Robot pos: ({px:6.2f}, {py:6.2f})", end="\\r")
            time.sleep(0.1 / speed_factor)
        print()

        curr_x, curr_y = target_x, target_y
        print(f"   [ARRIVED] Reached waypoint {wp['id']} (Type: {wp['type']}). Waiting {wait_time}s...")
        time.sleep(min(wait_time, 0.5) / speed_factor)
        print("-" * 55)

    print("\\n[SUCCESS] Unitree Go2 patrol mission simulation finished successfully!")


def main():
    parser = argparse.ArgumentParser(description="Unitree Go2 Waypoint Patrol Controller")
    default_json = os.path.join(os.path.dirname(__file__), "__DEFAULT_WAYPOINTS_JSON__")
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
        controller = Go2PatrolController(network_interface=args.net)
        controller.run_patrol(data)
    


if __name__ == "__main__":
    main()'''
        runner_code = runner_code.replace("__DEFAULT_WAYPOINTS_JSON__", json_filename)

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