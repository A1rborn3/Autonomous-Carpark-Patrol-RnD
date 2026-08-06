#!/usr/bin/env python3
import argparse
import importlib
import json
import math
import os
import random
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

import numpy as np

from scan_match_localizer import OccupancyScanMatcher, PoseEstimate

SDK_AVAILABLE = False
try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.go2.obstacles_avoid.obstacles_avoid_client import ObstaclesAvoidClient
    from unitree_sdk2py.go2.sport.sport_client import SportClient
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False


ARRIVAL_TOLERANCE_M = 0.15
YAW_TOLERANCE_DEG = 8.0
MAX_LINEAR_SPEED = 1.0
MAX_YAW_RATE = 1.0
CONTROL_HZ = 20.0
HEADING_KP = 2.0
TURN_IN_PLACE_THRESHOLD_DEG = 30.0


def _wrap_angle_rad(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def angle_diff_rad(target_rad: float, current_rad: float) -> float:
    return _wrap_angle_rad(target_rad - current_rad)


def _check_for_space_kill() -> bool:
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1) == " "
    except (termios.error, OSError, EOFError):
        return False
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return False


def load_json(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_map_contract(waypoints_data: dict, map_metadata_path: Optional[str]) -> Tuple[dict, float]:
    meta = waypoints_data.get("metadata", {})
    bounds = meta.get("map_bounds") or meta.get("bounds")
    resolution = meta.get("resolution")

    if map_metadata_path:
        metadata = load_json(map_metadata_path)
        bounds = bounds or metadata.get("bounds")
        resolution = resolution or metadata.get("resolution")

    if bounds is None:
        raise ValueError("Map bounds missing. Provide waypoints metadata.map_bounds or --map-metadata.")
    if resolution is None:
        raise ValueError("Map resolution missing. Provide waypoints metadata.resolution or --map-metadata.")
    return bounds, float(resolution)


class LidarScanProvider(Protocol):
    def get_scan_points_xy(self) -> np.ndarray:
        ...


class DDSLidarScanProvider:
    def __init__(
        self,
        topic: str,
        msg_type_path: str,
        wait_timeout_sec: float = 4.0,
        min_range_m: float = 0.2,
        max_range_m: float = 20.0,
    ):
        if not SDK_AVAILABLE:
            raise RuntimeError("unitree_sdk2py not installed; cannot subscribe to LiDAR topic")
        if not topic:
            raise ValueError("LiDAR topic cannot be empty")
        if not msg_type_path:
            raise ValueError("LiDAR message type cannot be empty")

        msg_cls = self._resolve_msg_class(msg_type_path)
        self._latest_msg = None
        self._sub = ChannelSubscriber(topic, msg_cls)
        self._sub.Init(self._on_msg, 10)

        self.min_range_m = min_range_m
        self.max_range_m = max_range_m

        t0 = time.time()
        while self._latest_msg is None and time.time() - t0 < wait_timeout_sec:
            time.sleep(0.05)
        if self._latest_msg is None:
            raise RuntimeError(
                f"No LiDAR data received on topic '{topic}' within {wait_timeout_sec:.1f}s. "
                f"Check --lidar-topic and --lidar-msg-type."
            )

    @staticmethod
    def _resolve_msg_class(msg_type_path: str):
        if "." not in msg_type_path:
            raise ValueError("LiDAR message type must be a dotted path, e.g. module.ClassName_")
        module_name, class_name = msg_type_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    def _on_msg(self, msg):
        self._latest_msg = msg

    def _points_from_ranges(self, msg) -> Optional[np.ndarray]:
        if not hasattr(msg, "ranges"):
            return None
        ranges = np.asarray(msg.ranges, dtype=np.float32).reshape(-1)
        if ranges.size == 0:
            return None

        angle_min = float(getattr(msg, "angle_min", -math.pi))
        angle_increment = getattr(msg, "angle_increment", None)
        if angle_increment is None:
            angle_max = float(getattr(msg, "angle_max", math.pi))
            angle_increment = (angle_max - angle_min) / max(1, ranges.size - 1)
        angle_increment = float(angle_increment)

        idx = np.arange(ranges.size, dtype=np.float32)
        angles = angle_min + idx * angle_increment
        finite = np.isfinite(ranges)
        in_range = (ranges >= self.min_range_m) & (ranges <= self.max_range_m)
        keep = finite & in_range
        if not np.any(keep):
            return None

        rr = ranges[keep]
        aa = angles[keep]
        x = rr * np.cos(aa)
        y = rr * np.sin(aa)
        return np.column_stack((x, y)).astype(np.float32)

    @staticmethod
    def _points_from_xy_iterable(points_obj) -> Optional[np.ndarray]:
        try:
            arr = np.asarray(points_obj)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return arr[:, :2].astype(np.float32)
        except Exception:
            pass

        if points_obj is None:
            return None
        rows = []
        try:
            iterator = iter(points_obj)
        except TypeError:
            return None
        for p in iterator:
            if hasattr(p, "x") and hasattr(p, "y"):
                rows.append((float(p.x), float(p.y)))
            elif isinstance(p, (tuple, list)) and len(p) >= 2:
                rows.append((float(p[0]), float(p[1])))
        if not rows:
            return None
        return np.asarray(rows, dtype=np.float32)

    def get_scan_points_xy(self) -> np.ndarray:
        msg = self._latest_msg
        if msg is None:
            raise RuntimeError("LiDAR message is not available yet.")

        pts = self._points_from_ranges(msg)
        if pts is not None:
            return pts

        for field_name in ("points", "point_cloud", "xyz", "data"):
            if hasattr(msg, field_name):
                pts = self._points_from_xy_iterable(getattr(msg, field_name))
                if pts is not None and pts.shape[0] > 0:
                    r = np.hypot(pts[:, 0], pts[:, 1])
                    keep = (r >= self.min_range_m) & (r <= self.max_range_m)
                    filtered = pts[keep]
                    if filtered.shape[0] > 0:
                        return filtered

        raise RuntimeError(
            "Unsupported LiDAR message layout. Expected LaserScan-like ranges or point list with x/y."
        )


@dataclass
class PoseState:
    x: float
    y: float
    yaw: float
    confidence: float


class Go2PatrolController:
    def __init__(
        self,
        network_interface: str,
        localizer: Optional[OccupancyScanMatcher],
        lidar_provider: Optional[LidarScanProvider],
        min_localization_confidence: float,
    ):
        if not SDK_AVAILABLE:
            raise RuntimeError("unitree_sdk2py not installed; cannot run live mode")

        ChannelFactoryInitialize(0, network_interface)

        self.sport_client = SportClient()
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()

        self.obstacle_client = ObstaclesAvoidClient()
        self.obstacle_client.Init()
        self.obstacle_avoidance_enabled = False

        self.localizer = localizer
        self.lidar_provider = lidar_provider
        self.min_localization_confidence = min_localization_confidence

        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0
        self._odom_ready = False

        self.pose = PoseState(0.0, 0.0, 0.0, 0.0)
        self.map_from_odom: Optional[Tuple[float, float, float]] = None
        self._state_sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        self._state_sub.Init(self._on_state, 10)
        self.is_standing = False

        t0 = time.time()
        while not self._odom_ready and time.time() - t0 < 3.0:
            time.sleep(0.05)
        if not self._odom_ready:
            raise RuntimeError("No SportModeState pose received within 3 seconds.")

    def _on_state(self, msg):
        self.odom_x = float(msg.position[0])
        self.odom_y = float(msg.position[1])
        self.odom_yaw = float(msg.imu_state.rpy[2])
        self._odom_ready = True

    @staticmethod
    def _compose_map_from_odom(odom_x: float, odom_y: float, odom_yaw: float, map_x: float, map_y: float, map_yaw: float):
        theta = _wrap_angle_rad(map_yaw - odom_yaw)
        c = math.cos(theta)
        s = math.sin(theta)
        tx = map_x - (c * odom_x - s * odom_y)
        ty = map_y - (s * odom_x + c * odom_y)
        return (tx, ty, theta)

    @staticmethod
    def _apply_map_from_odom(odom_x: float, odom_y: float, odom_yaw: float, map_from_odom: Tuple[float, float, float]):
        tx, ty, theta = map_from_odom
        c = math.cos(theta)
        s = math.sin(theta)
        map_x = tx + c * odom_x - s * odom_y
        map_y = ty + s * odom_x + c * odom_y
        map_yaw = _wrap_angle_rad(theta + odom_yaw)
        return map_x, map_y, map_yaw

    def _update_pose(self):
        if self.localizer is None:
            self.pose = PoseState(self.odom_x, self.odom_y, self.odom_yaw, 1.0)
            return
        if self.lidar_provider is None:
            raise RuntimeError("Scan-matching mode requires a LiDAR provider.")
        if self.map_from_odom is None:
            raise RuntimeError("Map-from-odom transform is not initialized. Run initial localization first.")

        scan_points = self.lidar_provider.get_scan_points_xy()
        pred_map_x, pred_map_y, pred_map_yaw = self._apply_map_from_odom(
            self.odom_x, self.odom_y, self.odom_yaw, self.map_from_odom
        )
        estimate = self.localizer.match_scan(
            scan_points_xy=scan_points,
            odom_pose=(pred_map_x, pred_map_y, pred_map_yaw),
        )
        self.pose = PoseState(estimate.x, estimate.y, estimate.yaw, estimate.confidence)
        if estimate.confidence >= 0.30:
            self.map_from_odom = self._compose_map_from_odom(
                self.odom_x, self.odom_y, self.odom_yaw, estimate.x, estimate.y, estimate.yaw
            )

    def enable_obstacle_avoidance(self):
        self.obstacle_client.UseRemoteCommandFromApi(True)
        self.obstacle_client.SwitchSet(True)
        self.obstacle_avoidance_enabled = True
        time.sleep(0.5)

    def disable_obstacle_avoidance(self):
        self.obstacle_client.SwitchSet(False)
        self.obstacle_client.UseRemoteCommandFromApi(False)
        self.obstacle_avoidance_enabled = False

    def _move(self, vx: float, vy: float, vyaw: float):
        if self.obstacle_avoidance_enabled:
            self.obstacle_client.Move(vx, vy, vyaw)
        else:
            self.sport_client.Move(vx=vx, vy=vy, vyaw=vyaw)

    def stand_up(self):
        self.sport_client.StopMove()
        time.sleep(0.2)
        self.sport_client.StandUp()
        self.is_standing = True
        time.sleep(2.5)
        self.sport_client.ClassicWalk(True)
        self.enable_obstacle_avoidance()

    def stand_down(self):
        self.sport_client.StopMove()
        self.disable_obstacle_avoidance()
        self.sport_client.Euler(0.0, 0.0, 0.0)
        time.sleep(0.2)
        self.sport_client.StandDown()
        self.is_standing = False
        time.sleep(1.5)

    def localize_initial_pose_by_spin(
        self,
        spin_seconds: float,
        spin_yaw_rate: float,
        sample_period_sec: float,
        required_confidence: float,
    ):
        if self.localizer is None or self.lidar_provider is None:
            raise RuntimeError("Initial spin localization requires scan_match mode and a LiDAR provider.")
        if not self.is_standing:
            self.stand_up()

        print("Starting initial global localization by spin...")
        best: Optional[PoseEstimate] = None
        t0 = time.time()
        last_sample_t = 0.0
        while time.time() - t0 < spin_seconds:
            if _check_for_space_kill():
                raise RuntimeError("Kill switch pressed during initial localization.")
            self._move(0.0, 0.0, max(-MAX_YAW_RATE, min(MAX_YAW_RATE, spin_yaw_rate)))
            now = time.time()
            if now - last_sample_t >= sample_period_sec:
                scan_points = self.lidar_provider.get_scan_points_xy()
                if scan_points.shape[0] < 20:
                    last_sample_t = now
                    continue
                estimate = self.localizer.global_match_scan(scan_points_xy=scan_points)
                if best is None or estimate.score > best.score:
                    best = estimate
                print(
                    f"  sample conf={estimate.confidence:.2f} score={estimate.score:.3f} "
                    f"pose=({estimate.x:.2f}, {estimate.y:.2f}, {math.degrees(estimate.yaw):.1f}deg)"
                )
                last_sample_t = now
            time.sleep(1.0 / CONTROL_HZ)

        self.sport_client.StopMove()
        if best is None:
            raise RuntimeError("Initial localization failed: no valid LiDAR scan match produced.")
        if best.confidence < required_confidence:
            raise RuntimeError(
                f"Initial localization confidence too low ({best.confidence:.2f} < {required_confidence:.2f})."
            )

        self.map_from_odom = self._compose_map_from_odom(
            self.odom_x, self.odom_y, self.odom_yaw, best.x, best.y, best.yaw
        )
        self.pose = PoseState(best.x, best.y, best.yaw, best.confidence)
        print(
            f"Initial localization fixed: x={best.x:.3f} m, y={best.y:.3f} m, "
            f"yaw={math.degrees(best.yaw):.2f} deg, conf={best.confidence:.2f}"
        )

    def navigate_to_waypoint(self, waypoint: dict) -> bool:
        target_x = float(waypoint["x"])
        target_y = float(waypoint["y"])
        target_yaw_deg = waypoint.get("yaw_deg")
        max_speed = min(float(waypoint.get("target_speed_m_s", 0.8)), MAX_LINEAR_SPEED)
        period = 1.0 / CONTROL_HZ

        while True:
            if _check_for_space_kill():
                print("\nKill switch pressed. Stopping.")
                self.sport_client.StopMove()
                self.stand_down()
                return False

            self._update_pose()
            if self.pose.confidence < self.min_localization_confidence:
                print(
                    f"Localization confidence {self.pose.confidence:.2f} below threshold "
                    f"{self.min_localization_confidence:.2f}. Stopping patrol."
                )
                self.sport_client.StopMove()
                self.stand_down()
                return False

            dx = target_x - self.pose.x
            dy = target_y - self.pose.y
            dist = math.hypot(dx, dy)
            if dist <= ARRIVAL_TOLERANCE_M:
                break

            bearing_to_target = math.atan2(dy, dx)
            heading_error = angle_diff_rad(bearing_to_target, self.pose.yaw)
            heading_error_deg = math.degrees(heading_error)
            vyaw = max(-MAX_YAW_RATE, min(MAX_YAW_RATE, HEADING_KP * heading_error))

            if abs(heading_error_deg) > TURN_IN_PLACE_THRESHOLD_DEG:
                vx = 0.0
            else:
                vx = max_speed * min(1.0, dist / 0.5)

            self._move(vx, 0.0, vyaw)
            time.sleep(period)

        self.sport_client.StopMove()

        if target_yaw_deg is not None:
            target_yaw_rad = math.radians(float(target_yaw_deg))
            while True:
                if _check_for_space_kill():
                    print("\nKill switch pressed. Stopping.")
                    self.sport_client.StopMove()
                    self.stand_down()
                    return False
                self._update_pose()
                err = angle_diff_rad(target_yaw_rad, self.pose.yaw)
                if abs(math.degrees(err)) <= YAW_TOLERANCE_DEG:
                    break
                vyaw = max(-MAX_YAW_RATE, min(MAX_YAW_RATE, HEADING_KP * err))
                self._move(0.0, 0.0, vyaw)
                time.sleep(period)
            self.sport_client.StopMove()

        return True

    def run_patrol(self, waypoints_data: dict):
        waypoints = waypoints_data.get("waypoints", [])
        print(f"Executing localized patrol for {len(waypoints)} waypoints...")
        if not self.is_standing:
            print("Standing up...")
            self.stand_up()

        for wp in waypoints:
            print(f"Navigating to {wp['id']} ({wp['x']:.2f}, {wp['y']:.2f})")
            ok = self.navigate_to_waypoint(wp)
            if not ok:
                return
            wait_time = float(wp.get("wait_time_sec", 0.5))
            if wait_time > 0:
                t0 = time.time()
                while time.time() - t0 < wait_time:
                    if _check_for_space_kill():
                        print("\nKill switch pressed. Stopping.")
                        self.sport_client.StopMove()
                        self.stand_down()
                        return
                    time.sleep(1.0 / CONTROL_HZ)

        print("Patrol completed. Returning to idle pose...")
        self.stand_down()


def _simulate_localized_dry_run(
    waypoints_data: dict,
    localizer: OccupancyScanMatcher,
    speed_factor: float,
    min_conf: float,
):
    print("\n=======================================================")
    print("   GO2 LOCALIZED PATROL DRY-RUN (SYNTHETIC LIDAR)      ")
    print("=======================================================")
    waypoints = waypoints_data.get("waypoints", [])
    if not waypoints:
        raise ValueError("Waypoints list is empty.")

    gt_x, gt_y, gt_yaw = waypoints[0]["x"], waypoints[0]["y"], math.radians(waypoints[0].get("yaw_deg", 0.0))
    odom_x, odom_y, odom_yaw = gt_x, gt_y, gt_yaw
    dt = (1.0 / CONTROL_HZ) / max(0.1, speed_factor)
    linear_speed = 0.6

    for wp in waypoints:
        target_x = float(wp["x"])
        target_y = float(wp["y"])
        print(f"\nTarget waypoint {wp['id']} -> ({target_x:.2f}, {target_y:.2f})")
        while True:
            dx = target_x - gt_x
            dy = target_y - gt_y
            dist = math.hypot(dx, dy)
            if dist <= ARRIVAL_TOLERANCE_M:
                break

            target_heading = math.atan2(dy, dx)
            heading_err = angle_diff_rad(target_heading, gt_yaw)
            gt_yaw = _wrap_angle_rad(gt_yaw + max(-0.6, min(0.6, heading_err * 1.8)) * dt)
            step_dist = min(linear_speed * dt, dist)
            gt_x += step_dist * math.cos(gt_yaw)
            gt_y += step_dist * math.sin(gt_yaw)

            odom_x += step_dist * math.cos(odom_yaw) + random.uniform(-0.005, 0.005)
            odom_y += step_dist * math.sin(odom_yaw) + random.uniform(-0.005, 0.005)
            odom_yaw = _wrap_angle_rad(odom_yaw + max(-0.5, min(0.5, heading_err * 1.2)) * dt + random.uniform(-0.002, 0.002))

            scan_points = localizer.raycast_scan_points((gt_x, gt_y, gt_yaw))
            if scan_points.shape[0] < 20:
                raise RuntimeError("Synthetic scan had too few obstacle returns; map may be too sparse.")
            estimate: PoseEstimate = localizer.match_scan(scan_points, (odom_x, odom_y, odom_yaw))
            pos_err = math.hypot(estimate.x - gt_x, estimate.y - gt_y)

            if estimate.confidence < min_conf:
                print(
                    f"[HALT] confidence={estimate.confidence:.2f} < {min_conf:.2f} "
                    f"at gt=({gt_x:.2f},{gt_y:.2f}), est=({estimate.x:.2f},{estimate.y:.2f})"
                )
                return

            print(
                f"gt=({gt_x:6.2f},{gt_y:6.2f}) odom=({odom_x:6.2f},{odom_y:6.2f}) "
                f"est=({estimate.x:6.2f},{estimate.y:6.2f}) err={pos_err:4.2f}m "
                f"conf={estimate.confidence:0.2f}",
                end="\r",
            )
            time.sleep(dt)

        print(" " * 130, end="\r")
        print(f"[ARRIVED] {wp['id']}")

    print("\n[SUCCESS] Dry-run localized patrol completed.")


def main():
    parser = argparse.ArgumentParser(description="Unitree Go2 localized waypoint patrol")
    parser.add_argument("--waypoints", required=True, help="Path to *_go2_waypoints.json")
    parser.add_argument("--occ-map", required=True, help="Path to occupancy map image")
    parser.add_argument("--map-metadata", help="Optional map_metadata.json if waypoints missing bounds/resolution")
    parser.add_argument("--net", default="eth0", help="Network interface for Unitree SDK 2")
    parser.add_argument("--dry-run", action="store_true", help="Run synthetic scan-matching simulation")
    parser.add_argument("--speed-factor", type=float, default=2.0, help="Dry-run speed multiplier")
    parser.add_argument(
        "--localization-mode",
        choices=("scan_match", "odom"),
        default="scan_match",
        help="scan_match: LiDAR-vs-map localization, odom: raw robot odometry only",
    )
    parser.add_argument(
        "--min-localization-confidence",
        type=float,
        default=0.55,
        help="Stop patrol if confidence drops below this threshold",
    )
    parser.add_argument(
        "--lidar-topic",
        default="",
        help="DDS topic for LiDAR data (required in live scan_match mode)",
    )
    parser.add_argument(
        "--lidar-msg-type",
        default="",
        help="Python dotted class path for LiDAR DDS message, e.g. unitree_sdk2py.idl.unitree_go.msg.dds_.YourMsg_",
    )
    parser.add_argument(
        "--initial-spin-seconds",
        type=float,
        default=10.0,
        help="Duration for startup spin-based global localization",
    )
    parser.add_argument(
        "--initial-spin-yaw-rate",
        type=float,
        default=0.6,
        help="Yaw rate (rad/s) during startup localization spin",
    )
    parser.add_argument(
        "--initial-sample-period",
        type=float,
        default=1.0,
        help="Seconds between global-match attempts during startup spin",
    )
    parser.add_argument(
        "--initial-min-confidence",
        type=float,
        default=0.45,
        help="Required confidence for accepting startup global localization",
    )
    parser.add_argument(
        "--localize-only",
        action="store_true",
        help="Perform startup localization and print pose, then exit without patrol",
    )
    args = parser.parse_args()

    waypoints_data = load_json(args.waypoints)
    bounds, resolution = resolve_map_contract(waypoints_data, args.map_metadata)

    localizer = None
    if args.localization_mode == "scan_match":
        localizer = OccupancyScanMatcher(
            occupancy_map_path=args.occ_map,
            bounds=bounds,
            resolution_px_per_m=resolution,
        )

    if args.dry_run or not SDK_AVAILABLE:
        if args.localization_mode == "odom":
            print("Dry-run in odom mode is not useful for map localization; choose --localization-mode scan_match.")
            return
        _simulate_localized_dry_run(
            waypoints_data=waypoints_data,
            localizer=localizer,
            speed_factor=args.speed_factor,
            min_conf=args.min_localization_confidence,
        )
        return

    lidar_provider = None
    if args.localization_mode == "scan_match":
        if not args.lidar_topic or not args.lidar_msg_type:
            raise ValueError(
                "In live scan_match mode, provide --lidar-topic and --lidar-msg-type."
            )
        lidar_provider = DDSLidarScanProvider(
            topic=args.lidar_topic,
            msg_type_path=args.lidar_msg_type,
        )

    controller = Go2PatrolController(
        network_interface=args.net,
        localizer=localizer,
        lidar_provider=lidar_provider,
        min_localization_confidence=args.min_localization_confidence,
    )
    if args.localization_mode == "scan_match":
        controller.localize_initial_pose_by_spin(
            spin_seconds=args.initial_spin_seconds,
            spin_yaw_rate=args.initial_spin_yaw_rate,
            sample_period_sec=args.initial_sample_period,
            required_confidence=args.initial_min_confidence,
        )
    if args.localize_only:
        return
    controller.run_patrol(waypoints_data)


if __name__ == "__main__":
    main()
