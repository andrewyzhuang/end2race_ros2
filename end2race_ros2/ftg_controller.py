from typing import Tuple
import numpy as np


class FollowTheGapController:
    """
    High-speed Follow-the-Gap (FTG) reactive baseline controller.
    Operates on the 180-beam CCW scan (index 0 = forward, 0..45 = front-left, 135..179 = front-right).
    """

    def __init__(
        self,
        bubble_radius_m: float = 0.35,
        disparity_threshold_m: float = 0.40,
        max_lookahead_m: float = 5.0,
        max_speed_m_s: float = 3.5,
        min_speed_m_s: float = 1.2,
        max_steer_rad: float = 0.42,
        steer_gain: float = 0.85,
    ):
        self.bubble_radius = bubble_radius_m
        self.disparity_thresh = disparity_threshold_m
        self.max_lookahead = max_lookahead_m
        self.max_speed = max_speed_m_s
        self.min_speed = min_speed_m_s
        self.max_steer = max_steer_rad
        self.steer_gain = steer_gain

    def compute_action(
        self,
        forward_ccw_180_m: np.ndarray,
        beam_angles_rad: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Computes (steer_rad, speed_m_s) from the front 180-degree FOV (-90 deg to +90 deg).
        """
        # Extract front 180-deg FOV in contiguous order from -90 deg (right) to +90 deg (left)
        # In forward_ccw_180_m: indices 135..179 are [-90, -2 deg], indices 0..45 are [0, +90 deg]
        front_indices = np.concatenate([np.arange(135, 180), np.arange(0, 46)])
        ranges = np.clip(forward_ccw_180_m[front_indices].copy(), 0.0, self.max_lookahead)
        angles = beam_angles_rad[front_indices]

        # 1. Disparity Extender: mask beams adjacent to sharp depth discontinuities
        diffs = np.abs(np.diff(ranges))
        disparities = np.where(diffs > self.disparity_thresh)[0]
        beam_step_rad = np.deg2rad(2.0)
        for idx in disparities:
            closer_idx = idx if ranges[idx] < ranges[idx + 1] else idx + 1
            d = max(ranges[closer_idx], 0.1)
            num_pad = int(np.ceil(np.arctan2(self.bubble_radius, d) / beam_step_rad))
            start = max(0, closer_idx - num_pad)
            end = min(len(ranges), closer_idx + num_pad + 1)
            ranges[start:end] = np.minimum(ranges[start:end], d)

        # 2. Safety Bubble around the closest point
        min_idx = int(np.argmin(ranges))
        min_dist = max(float(ranges[min_idx]), 0.1)
        bubble_beams = int(np.ceil(np.arctan2(self.bubble_radius, min_dist) / beam_step_rad))
        b_start = max(0, min_idx - bubble_beams)
        b_end = min(len(ranges), min_idx + bubble_beams + 1)
        ranges[b_start:b_end] = 0.0

        # 3. Find largest contiguous non-zero gap above dynamic threshold
        free_thresh = max(1.2, 0.45 * float(np.max(ranges)))
        valid = ranges >= free_thresh
        if not np.any(valid):
            valid = ranges > 0.3

        # Identify longest contiguous segment of True in `valid`
        padded = np.concatenate(([False], valid, [False]))
        starts = np.where(padded[1:] & ~padded[:-1])[0]
        ends = np.where(~padded[1:] & padded[:-1])[0]

        if len(starts) > 0:
            lengths = ends - starts
            best_gap = int(np.argmax(lengths))
            g_start, g_end = starts[best_gap], ends[best_gap]
            # Weighted center of the deepest points in the best gap
            gap_slice = ranges[g_start:g_end]
            gap_angles = angles[g_start:g_end]
            weights = gap_slice ** 2
            target_angle = float(np.sum(gap_angles * weights) / max(np.sum(weights), 1e-5))
        else:
            target_angle = float(angles[int(np.argmax(ranges))])

        steer_rad = float(np.clip(self.steer_gain * target_angle, -self.max_steer, self.max_steer))

        # 4. Speed profile based on forward clearance and steering angle
        fwd_dist = float(np.mean(forward_ccw_180_m[[178, 179, 0, 1, 2]]))
        steer_ratio = abs(steer_rad) / self.max_steer
        speed_m_s = self.min_speed + (self.max_speed - self.min_speed) * (
            1.0 - 0.65 * steer_ratio
        ) * np.clip(fwd_dist / 3.5, 0.3, 1.0)

        return steer_rad, float(np.clip(speed_m_s, self.min_speed, self.max_speed))