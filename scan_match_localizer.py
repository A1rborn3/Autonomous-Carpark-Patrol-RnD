import math
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


def _wrap_angle_rad(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class PoseEstimate:
    x: float
    y: float
    yaw: float
    confidence: float
    score: float


class OccupancyScanMatcher:
    """
    2D scan matcher against an occupancy map.

    The matcher assumes scan points are obstacle returns in robot frame (meters),
    where +x is forward and +y is left.
    """

    def __init__(
        self,
        occupancy_map_path: str,
        bounds: dict,
        resolution_px_per_m: float,
        obstacle_threshold: int = 50,
    ):
        self.bounds = bounds
        self.resolution = float(resolution_px_per_m)

        occ = cv2.imread(occupancy_map_path, cv2.IMREAD_UNCHANGED)
        if occ is None:
            raise FileNotFoundError(f"Could not read occupancy map: {occupancy_map_path}")
        if occ.ndim == 3:
            occ = cv2.cvtColor(occ, cv2.COLOR_BGR2GRAY)

        # Blocked cells are dark in this repo's occupancy output.
        self.obstacle_mask = occ <= obstacle_threshold
        self.height, self.width = self.obstacle_mask.shape[:2]

    def world_to_pixel(self, x_m: np.ndarray, y_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        u = np.rint((x_m - self.bounds["min_x"]) * self.resolution).astype(np.int32)
        v = np.rint((self.bounds["max_y"] - y_m) * self.resolution).astype(np.int32)
        return u, v

    def _score_pose(self, scan_xy: np.ndarray, pose_x: float, pose_y: float, pose_yaw: float) -> float:
        c = math.cos(pose_yaw)
        s = math.sin(pose_yaw)

        wx = pose_x + c * scan_xy[:, 0] - s * scan_xy[:, 1]
        wy = pose_y + s * scan_xy[:, 0] + c * scan_xy[:, 1]
        u, v = self.world_to_pixel(wx, wy)

        in_bounds = (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        valid_count = int(np.count_nonzero(in_bounds))
        if valid_count == 0:
            return 0.0

        uv_hits = self.obstacle_mask[v[in_bounds], u[in_bounds]]
        hit_ratio = float(np.count_nonzero(uv_hits)) / float(valid_count)
        coverage_ratio = float(valid_count) / float(scan_xy.shape[0])
        return 0.8 * hit_ratio + 0.2 * coverage_ratio

    def match_scan(
        self,
        scan_points_xy: np.ndarray,
        odom_pose: Tuple[float, float, float],
        search_xy_m: float = 0.8,
        search_yaw_deg: float = 25.0,
        step_xy_m: float = 0.05,
        step_yaw_deg: float = 2.0,
    ) -> PoseEstimate:
        if scan_points_xy.ndim != 2 or scan_points_xy.shape[1] != 2:
            raise ValueError("scan_points_xy must be shape [N, 2]")
        if scan_points_xy.shape[0] < 20:
            raise ValueError("scan_points_xy must contain at least 20 points")

        odom_x, odom_y, odom_yaw = odom_pose
        x_vals = np.arange(odom_x - search_xy_m, odom_x + search_xy_m + 1e-9, step_xy_m)
        y_vals = np.arange(odom_y - search_xy_m, odom_y + search_xy_m + 1e-9, step_xy_m)
        yaw_step = math.radians(step_yaw_deg)
        yaw_range = math.radians(search_yaw_deg)
        yaw_vals = np.arange(odom_yaw - yaw_range, odom_yaw + yaw_range + 1e-9, yaw_step)

        best_score = -1.0
        second_score = -1.0
        best_pose = (odom_x, odom_y, odom_yaw)

        for cyaw in yaw_vals:
            wrapped_yaw = _wrap_angle_rad(float(cyaw))
            for cx in x_vals:
                for cy in y_vals:
                    score = self._score_pose(scan_points_xy, float(cx), float(cy), wrapped_yaw)
                    if score > best_score:
                        second_score = best_score
                        best_score = score
                        best_pose = (float(cx), float(cy), wrapped_yaw)
                    elif score > second_score:
                        second_score = score

        margin = max(0.0, best_score - max(0.0, second_score))
        confidence = max(0.0, min(1.0, 0.75 * best_score + 0.25 * min(1.0, margin * 8.0)))
        return PoseEstimate(
            x=best_pose[0],
            y=best_pose[1],
            yaw=best_pose[2],
            confidence=confidence,
            score=best_score,
        )

    def global_match_scan(
        self,
        scan_points_xy: np.ndarray,
        coarse_xy_step_m: float = 1.5,
        coarse_yaw_step_deg: float = 20.0,
        top_k: int = 3,
    ) -> PoseEstimate:
        """
        Global localization against the full map (no odometry prior).
        Uses coarse search followed by local refinement around top candidates.
        """
        if scan_points_xy.ndim != 2 or scan_points_xy.shape[1] != 2:
            raise ValueError("scan_points_xy must be shape [N, 2]")
        if scan_points_xy.shape[0] < 20:
            raise ValueError("scan_points_xy must contain at least 20 points")
        if coarse_xy_step_m <= 0.0:
            raise ValueError("coarse_xy_step_m must be > 0")
        if coarse_yaw_step_deg <= 0.0:
            raise ValueError("coarse_yaw_step_deg must be > 0")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        x_vals = np.arange(self.bounds["min_x"], self.bounds["max_x"] + 1e-9, coarse_xy_step_m)
        y_vals = np.arange(self.bounds["min_y"], self.bounds["max_y"] + 1e-9, coarse_xy_step_m)
        yaw_step = math.radians(coarse_yaw_step_deg)
        yaw_vals = np.arange(-math.pi, math.pi + 1e-9, yaw_step)

        best_candidates = []
        for cyaw in yaw_vals:
            wrapped_yaw = _wrap_angle_rad(float(cyaw))
            for cx in x_vals:
                for cy in y_vals:
                    score = self._score_pose(scan_points_xy, float(cx), float(cy), wrapped_yaw)
                    if len(best_candidates) < top_k:
                        best_candidates.append((score, float(cx), float(cy), wrapped_yaw))
                        best_candidates.sort(key=lambda item: item[0], reverse=True)
                    elif score > best_candidates[-1][0]:
                        best_candidates[-1] = (score, float(cx), float(cy), wrapped_yaw)
                        best_candidates.sort(key=lambda item: item[0], reverse=True)

        if not best_candidates:
            raise RuntimeError("Global localization failed to produce any candidates.")

        refined = []
        for _, cx, cy, cyaw in best_candidates:
            local_est = self.match_scan(
                scan_points_xy=scan_points_xy,
                odom_pose=(cx, cy, cyaw),
                search_xy_m=max(0.6, coarse_xy_step_m),
                search_yaw_deg=max(12.0, coarse_yaw_step_deg),
                step_xy_m=0.08,
                step_yaw_deg=2.0,
            )
            refined.append(local_est)

        refined.sort(key=lambda e: e.score, reverse=True)
        return refined[0]

    def raycast_scan_points(
        self,
        world_pose: Tuple[float, float, float],
        max_range_m: float = 12.0,
        angle_min_deg: float = -135.0,
        angle_max_deg: float = 135.0,
        angle_step_deg: float = 2.0,
        ray_step_m: float = 0.08,
    ) -> np.ndarray:
        """
        Synthetic LiDAR helper for dry-run testing.
        Returns obstacle hit points in robot frame.
        """
        px, py, yaw = world_pose
        hits = []

        ang = angle_min_deg
        while ang <= angle_max_deg + 1e-9:
            ray_yaw = yaw + math.radians(ang)
            c = math.cos(ray_yaw)
            s = math.sin(ray_yaw)
            dist = ray_step_m
            found = False
            while dist <= max_range_m + 1e-9:
                wx = px + dist * c
                wy = py + dist * s
                u, v = self.world_to_pixel(np.array([wx]), np.array([wy]))
                uu = int(u[0])
                vv = int(v[0])
                if uu < 0 or uu >= self.width or vv < 0 or vv >= self.height:
                    break
                if self.obstacle_mask[vv, uu]:
                    lx = dist * math.cos(math.radians(ang))
                    ly = dist * math.sin(math.radians(ang))
                    hits.append((lx, ly))
                    found = True
                    break
                dist += ray_step_m
            if not found:
                # no return for this ray
                pass
            ang += angle_step_deg

        if not hits:
            return np.zeros((0, 2), dtype=np.float32)
        return np.asarray(hits, dtype=np.float32)
