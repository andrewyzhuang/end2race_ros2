import time
from typing import Tuple
import numpy as np


class End2RacePreprocessor:
    """
    Sanitizes NeoRacer LakiBeam1 LaserScan (~1440 CW samples with rear blind wedge),
    aligns angular convention to F1TENTH 360-deg CCW 180-beam scan, maintains short-term
    side-corridor obstacle memory across the blind wedge, and applies End2Race sigmoid
    normalization.
    """

    def __init__(
        self,
        num_output_beams: int = 180,
        max_lidar_range_m: float = 10.0,
        sigmoid_k: float = 0.3,
        lidar_distance_scale: float = 1.15,
        speed_input_scale: float = 1.25,
        lidar_scan_clockwise: bool = True,
        shift_index_zero_to_rear: bool = True,
        rear_blind_angle_min_deg: float = 135.0,
        rear_blind_angle_max_deg: float = 225.0,
        rear_blind_fill_distance_m: float = 10.0,
        side_corridor_hold_sec: float = 0.25,
        median_filter_window: int = 3,
    ):
        self.num_output_beams = num_output_beams
        self.max_range = max_lidar_range_m
        self.k = sigmoid_k
        self.lidar_distance_scale = lidar_distance_scale
        self.speed_input_scale = speed_input_scale
        self.lidar_scan_clockwise = lidar_scan_clockwise
        self.shift_index_zero_to_rear = shift_index_zero_to_rear
        self.rear_blind_min_deg = rear_blind_angle_min_deg
        self.rear_blind_max_deg = rear_blind_angle_max_deg
        self.rear_blind_fill = rear_blind_fill_distance_m
        self.side_corridor_hold_sec = side_corridor_hold_sec
        self.median_window = median_filter_window

        # sigmoid normalization
        # Normalization constant xi so sigma(max_range) == 1.0 and sigma(0.0) == 0.0
        # Using standard zero-anchored logistic curve: (2 / (1 + exp(-k*x)) - 1) * xi
        denom = (2.0 / (1.0 + np.exp(-self.k * self.max_range))) - 1.0
        self.xi = 1.0 / max(denom, 1e-6)

        # Temporal memory for side-corridor overtake protection (180 beams)
        self._held_side_ranges = np.full(self.num_output_beams, self.max_range, dtype=np.float32)
        self._last_update_time = time.monotonic()

    def reset(self) -> None:
        self._held_side_ranges.fill(self.max_range)
        self._last_update_time = time.monotonic()

    def process(
        self,
        raw_ranges: np.ndarray,
        ego_speed_m_s: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Processes a raw LakiBeam1 LaserScan range array and scalar ego speed.

        Returns:
            norm_lidar_180: (180,) float32 array in [0, 1] ordered for End2Race network
            forward_ccw_180_m: (180,) float32 metric distances (m) where index 0 is 0 deg (forward)
                               and angles increase CCW (0 -> 90 left -> 180 rear -> 270 right)
            beam_angles_rad: (180,) float32 angles in [-pi, pi] (0 = straight ahead, +pi/2 = left)
            scaled_ego_speed: float scaled ego speed for network input
        """
        now = time.monotonic()
        dt = max(1e-3, now - self._last_update_time)
        self._last_update_time = now

        scans = np.asarray(raw_ranges, dtype=np.float32).copy()
        n_raw = len(scans)
        if n_raw == 0:
            scans = np.full(1440, self.max_range, dtype=np.float32)
            n_raw = 1440

        # 1. Mask out rear chassis occlusion wedge (135 deg to 225 deg CW on LakiBeam1)
        raw_angles_deg = np.linspace(0.0, 360.0, n_raw, endpoint=False, dtype=np.float32)
        blind_mask = (raw_angles_deg >= self.rear_blind_min_deg) & (
            raw_angles_deg <= self.rear_blind_max_deg
        )
        scans[blind_mask] = self.rear_blind_fill

        # 2. Sanitize NaNs, infs, and invalid zeros (specular drop-outs or out-of-range)
        invalid_mask = ~np.isfinite(scans) | (scans <= 0.03)
        scans[invalid_mask] = self.max_range

        # Optional 3-tap circular median filter to strip single-beam multipath spikes
        if self.median_window == 3:
            scans = np.median(
                np.stack([np.roll(scans, 1), scans, np.roll(scans, -1)], axis=0),
                axis=0,
            )

        # 3. Convert scan orientation from LakiBeam1 CW (idx 0 = forward, idx N/4 = right)
        #    to standard robotics CCW (idx 0 = forward, idx N/4 = left, idx 3N/4 = right)
        if self.lidar_scan_clockwise:
            scans_ccw = np.empty_like(scans)
            scans_ccw[0] = scans[0]
            scans_ccw[1:] = scans[1:][::-1]
        else:
            scans_ccw = scans

        # 4. Uniformly downsample from ~1440 beams to 180 beams (2.0 deg spacing)
        idx_180 = np.linspace(0, n_raw, self.num_output_beams, endpoint=False).astype(np.int32)
        forward_ccw_180_m = np.clip(scans_ccw[idx_180], 0.0, self.max_range)

        # 5. Side-Corridor Overtake Hold (protects Stage 4->5 lane recovery when opponent enters blind spot)
        # Beams around 90-135 deg (left rear quarter, indices 45..67) and 225-270 deg (right rear quarter, 112..135)
        if self.side_corridor_hold_sec > 0.0:
            decay_rate = (self.max_range / self.side_corridor_hold_sec) * dt
            self._held_side_ranges = np.minimum(
                self.max_range, self._held_side_ranges + decay_rate
            )
            side_mask = np.zeros(self.num_output_beams, dtype=bool)
            side_mask[40:70] = True   # Left flank / rear-quarter
            side_mask[110:140] = True # Right flank / rear-quarter

            self._held_side_ranges[side_mask] = np.minimum(
                self._held_side_ranges[side_mask], forward_ccw_180_m[side_mask]
            )
            forward_ccw_180_m[side_mask] = np.minimum(
                forward_ccw_180_m[side_mask], self._held_side_ranges[side_mask]
            )

        # 6. Apply 1:12 -> 1:10 track geometry scaling for the neural policy
        scaled_ranges = np.clip(
            forward_ccw_180_m * self.lidar_distance_scale, 0.0, self.max_range
        )

        # 7. Align beam 0 to match End2Race training convention
        # In F1TENTH Gym 360-deg scans, beam 0 starts at -pi (rear) and sweeps CCW through 0 (front, beam 90) to +pi.
        if self.shift_index_zero_to_rear:
            policy_ranges_180 = np.roll(scaled_ranges, self.num_output_beams // 2)
        else:
            policy_ranges_180 = scaled_ranges

        # 8. Sigmoid Normalization (Eq. 2 in End2Race paper): maps [0, 10] -> [0, 1]
        norm_lidar_180 = (
            (2.0 / (1.0 + np.exp(-self.k * policy_ranges_180))) - 1.0
        ) * self.xi
        norm_lidar_180 = np.clip(norm_lidar_180, 0.0, 1.0).astype(np.float32)

        # Compute angles in [-pi, pi] corresponding to forward_ccw_180_m (for safety filter & FTG)
        ccw_angles_rad = np.linspace(
            0.0, 2.0 * np.pi, self.num_output_beams, endpoint=False, dtype=np.float32
        )
        beam_angles_rad = (ccw_angles_rad + np.pi) % (2.0 * np.pi) - np.pi

        scaled_ego_speed = float(ego_speed_m_s * self.speed_input_scale)
        return norm_lidar_180, forward_ccw_180_m, beam_angles_rad, scaled_ego_speed