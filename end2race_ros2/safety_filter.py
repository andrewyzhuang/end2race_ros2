import time
from typing import Tuple
import numpy as np


class RacingSafetyFilter:
    """
    High-speed autonomous racing safety & stability filter for NeoRacer:
      1. Steering-Aligned Cone iTTC:
         - Soft speed scaling when iTTC < ttc_warning_sec (preserves steering for overtaking)
         - Hard E-brake + GRU reset flag when iTTC < ttc_critical_sec
      2. Friction-Circle ("Traction Control") Modulator:
         - Modulates longitudinal speed command as lateral acceleration approaches tire grip limit,
           preventing high-speed understeer without restricting steering angle.
      3. IMU Yaw-Rate Drift Assist:
         - Permits sideslip/drift within max_drift_yaw_error_rad_s.
         - Applies counter-steering damping only when yaw rate exceeds the drift envelope (spin prevention).
      4. Servo Rate & EMA Smoothing:
         - Eliminates straight-line servo oscillation caused by actuator latency.
    """

    def __init__(
        self,
        wheelbase_m: float = 0.28,
        safety_margin_radius_m: float = 0.18,
        ttc_warning_sec: float = 0.85,
        ttc_critical_sec: float = 0.32,
        ttc_ebrake_speed_m_s: float = 0.0,
        enable_traction_circle: bool = True,
        max_lat_accel_m_s2: float = 5.5,
        traction_min_speed_ratio: float = 0.45,
        enable_yaw_damping: bool = True,
        max_drift_yaw_error_rad_s: float = 1.8,
        yaw_damping_kp: float = 0.08,
        steering_ema_alpha: float = 0.80,
        max_steering_rate_rad_s: float = 3.2,
        max_steer_rad: float = 0.42,
        max_speed_cap_m_s: float = 2.5,
        min_speed_floor_m_s: float = 0.0,
    ):
        self.wheelbase = wheelbase_m
        self.safety_margin = safety_margin_radius_m
        self.ttc_warning = ttc_warning_sec
        self.ttc_critical = ttc_critical_sec
        self.ttc_ebrake_speed = ttc_ebrake_speed_m_s

        self.enable_traction_circle = enable_traction_circle
        self.max_lat_accel = max_lat_accel_m_s2
        self.traction_min_ratio = traction_min_speed_ratio

        self.enable_yaw_damping = enable_yaw_damping
        self.max_drift_yaw_error = max_drift_yaw_error_rad_s
        self.yaw_damping_kp = yaw_damping_kp

        self.ema_alpha = steering_ema_alpha
        self.max_steer_rate = max_steering_rate_rad_s
        self.max_steer = max_steer_rad
        self.max_speed_cap = max_speed_cap_m_s
        self.min_speed_floor = min_speed_floor_m_s

        self._prev_steer_rad = 0.0
        self._last_time = time.monotonic()

    def reset(self) -> None:
        self._prev_steer_rad = 0.0
        self._last_time = time.monotonic()

    def filter_commands(
        self,
        cmd_steer_rad: float,
        cmd_speed_m_s: float,
        forward_ccw_180_m: np.ndarray,
        beam_angles_rad: np.ndarray,
        ego_speed_m_s: float,
        imu_yaw_rate_rad_s: float,
    ) -> Tuple[float, float, bool, float, str]:
        """
        Applies dynamic stability and collision filters to raw policy outputs.

        Returns:
            safe_steer_rad: filtered steering command in radians (CCW +left convention)
            safe_speed_m_s: filtered speed command in m/s
            reset_gru: bool flag indicating whether GRU hidden state should be reset
            min_ttc_sec: minimum instantaneous TTC along intended path
            status_code: short status string for LED dot-matrix ("OK", "TTC-W", "E-BRK", "DRIFT")
        """
        now = time.monotonic()
        dt = np.clip(now - self._last_time, 1e-3, 0.1)
        self._last_time = now

        status_code = "OK"
        reset_gru = False

        # Clamp initial raw inputs to physical bounds
        steer = float(np.clip(cmd_steer_rad, -self.max_steer, self.max_steer))
        speed = float(np.clip(cmd_speed_m_s, self.min_speed_floor, self.max_speed_cap))

        # ------------------------------------------------------------------
        # Stage 1: IMU Yaw-Rate Drift Assist (Spin Arrest without killing slip)
        # ------------------------------------------------------------------
        if self.enable_yaw_damping and abs(ego_speed_m_s) > 0.8:
            kinematic_yaw_rate = (ego_speed_m_s * np.tan(steer)) / self.wheelbase
            yaw_error = imu_yaw_rate_rad_s - kinematic_yaw_rate
            if abs(yaw_error) > self.max_drift_yaw_error:
                # Over-rotation beyond intentional drift window: apply counter-steer correction
                excess_error = np.sign(yaw_error) * (abs(yaw_error) - self.max_drift_yaw_error)
                steer -= self.yaw_damping_kp * excess_error
                steer = float(np.clip(steer, -self.max_steer, self.max_steer))
                status_code = "DRIFT"

        # ------------------------------------------------------------------
        # Stage 2: Steering Rate Limit & EMA Smoothing (Servo Wobble Prevention)
        # ------------------------------------------------------------------
        steer_ema = self.ema_alpha * steer + (1.0 - self.ema_alpha) * self._prev_steer_rad
        max_delta = self.max_steer_rate * dt
        steer_smooth = float(
            np.clip(steer_ema, self._prev_steer_rad - max_delta, self._prev_steer_rad + max_delta)
        )
        self._prev_steer_rad = steer_smooth

        # ------------------------------------------------------------------
        # Stage 3: Friction-Circle ("Traction Control") Speed Modulation
        # ------------------------------------------------------------------
        if self.enable_traction_circle and speed > 1.0:
            # Use max of current speed and commanded speed to anticipate corner-entry lateral g
            v_ref = max(ego_speed_m_s, 0.7 * speed)
            lat_accel_est = (v_ref ** 2) * abs(np.tan(steer_smooth)) / self.wheelbase
            grip_ratio = min(1.0, lat_accel_est / max(self.max_lat_accel, 0.5))
            speed_scale = np.sqrt(
                max(self.traction_min_ratio ** 2, 1.0 - (grip_ratio ** 2))
            )
            speed *= float(speed_scale)

        # ------------------------------------------------------------------
        # Stage 4: Steering-Aligned Cone iTTC (Soft Scaling + Hard E-Brake)
        # ------------------------------------------------------------------
        # Evaluate beams within +/- 14 degrees (0.24 rad) of the intended steering angle
        cone_half_angle = 0.24
        angle_diff = np.abs((beam_angles_rad - steer_smooth + np.pi) % (2.0 * np.pi) - np.pi)
        cone_mask = angle_diff <= cone_half_angle

        v_closing_base = max(ego_speed_m_s, speed, 0.1)
        if np.any(cone_mask):
            cone_ranges = forward_ccw_180_m[cone_mask]
            cone_diffs = angle_diff[cone_mask]
            closing_vels = np.maximum(0.05, v_closing_base * np.cos(cone_diffs))
            dist_clear = np.maximum(0.0, cone_ranges - self.safety_margin)
            ttcs = dist_clear / closing_vels
            min_ttc_sec = float(np.min(ttcs))
        else:
            min_ttc_sec = 99.0

        if min_ttc_sec < self.ttc_critical:
            # Critical imminent collision: Emergency brake, nudge steering toward wider side gap, reset GRU
            speed = self.ttc_ebrake_speed
            reset_gru = True
            status_code = "E-BRK"

            # Find whether left (0..60 deg) or right (-60..0 deg) has more clearance for escape
            left_clear = float(np.mean(forward_ccw_180_m[5:35]))
            right_clear = float(np.mean(forward_ccw_180_m[145:175]))
            steer_smooth = 0.30 if left_clear > right_clear else -0.30
            self._prev_steer_rad = steer_smooth

        elif min_ttc_sec < self.ttc_warning:
            # Soft TTC modulation: linearly scale speed down while keeping full steering authority for overtaking
            alpha_ttc = (min_ttc_sec - self.ttc_critical) / max(
                1e-3, (self.ttc_warning - self.ttc_critical)
            )
            speed *= float(np.clip(alpha_ttc, 0.20, 1.0))
            if status_code == "OK":
                status_code = "TTC-W"

        return steer_smooth, speed, reset_gru, min_ttc_sec, status_code