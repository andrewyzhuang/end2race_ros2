#!/usr/bin/env python3
import math
import time
import numpy as np
from scipy.ndimage import median_filter
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Float32, String
from sensor_msgs.msg import LaserScan, Imu, Joy
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped


class End2RacePolicyNode(Node):
    def __init__(self):
        super().__init__('end2race_policy_node')

        # --- 1. Declare All Parameters from policy_params.yaml ---
        self.declare_parameter('mode', 'end2race')
        self.declare_parameter('inference_backend', 'torch')
        self.declare_parameter('model_path', 'models/end2race_policy_jit_fp16.pt')
        self.declare_parameter('use_fp16', True)
        self.declare_parameter('control_rate_hz', 50.0)

        self.declare_parameter('num_beams', 180)
        self.declare_parameter('gru_hidden_dim', 420)
        self.declare_parameter('use_dummy_speed', False)
        self.declare_parameter('actions_are_normalized', False)
        self.declare_parameter('action_steer_index', 0)
        self.declare_parameter('action_speed_index', 1)
        self.declare_parameter('gru_stale_timeout_sec', 0.25)

        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('speed_topic', '/encoder/speed')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('imu_topic', '/imu/fused')
        self.declare_parameter('joy_topic', '/joy')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('dotmatrix_topic', '/dotmatrix/text')

        self.declare_parameter('normalize_drive_output', True)
        self.declare_parameter('neoracer_max_physical_speed_m_s', 6.0)
        self.declare_parameter('neoracer_max_physical_steer_rad', 0.42)
        self.declare_parameter('max_steer_rad', 0.4189)
        self.declare_parameter('steer_sign', -1.0)
        self.declare_parameter('steer_trim_normalized', 0.0)

        self.declare_parameter('lidar_distance_scale', 1.15)
        self.declare_parameter('speed_input_scale', 1.25)
        self.declare_parameter('speed_output_scale', 0.65)
        self.declare_parameter('max_speed_cap_m_s', 2.5)
        self.declare_parameter('min_speed_floor_m_s', 0.0)

        self.declare_parameter('lidar_scan_clockwise', True)
        self.declare_parameter('shift_index_zero_to_rear', True)
        self.declare_parameter('rear_blind_angle_min_deg', 135.0)
        self.declare_parameter('rear_blind_angle_max_deg', 225.0)
        self.declare_parameter('rear_blind_fill_distance_m', 10.0)
        self.declare_parameter('max_lidar_range_m', 10.0)
        self.declare_parameter('min_lidar_range_m', 0.05)
        self.declare_parameter('side_corridor_hold_sec', 0.25)
        self.declare_parameter('median_filter_window', 3)

        self.declare_parameter('wheelbase_m', 0.26)
        self.declare_parameter('vehicle_width_m', 0.22)
        self.declare_parameter('safety_margin_radius_m', 0.18)
        self.declare_parameter('ttc_warning_sec', 0.85)
        self.declare_parameter('ttc_critical_sec', 0.32)
        self.declare_parameter('ttc_ebrake_speed_m_s', 0.0)

        self.declare_parameter('enable_traction_circle', True)
        self.declare_parameter('max_lat_accel_m_s2', 5.5)
        self.declare_parameter('traction_min_speed_ratio', 0.45)

        self.declare_parameter('enable_yaw_damping', True)
        self.declare_parameter('max_drift_yaw_error_rad_s', 1.8)
        self.declare_parameter('yaw_damping_kp', 0.08)

        self.declare_parameter('steering_ema_alpha', 0.80)
        self.declare_parameter('max_steering_rate_rad_s', 3.2)

        # --- 2. Read Parameters ---
        self.mode = self.get_parameter('mode').value
        self.backend = self.get_parameter('inference_backend').value
        self.model_path = self.get_parameter('model_path').value
        self.use_fp16 = self.get_parameter('use_fp16').value
        self.control_rate = float(self.get_parameter('control_rate_hz').value)
        self.dt = 1.0 / max(self.control_rate, 1.0)

        self.num_beams = int(self.get_parameter('num_beams').value)
        self.gru_hidden_dim = int(self.get_parameter('gru_hidden_dim').value)
        self.actions_are_normalized = bool(self.get_parameter('actions_are_normalized').value)
        self.steer_idx = int(self.get_parameter('action_steer_index').value)
        self.speed_idx = int(self.get_parameter('action_speed_index').value)
        self.stale_timeout = float(self.get_parameter('gru_stale_timeout_sec').value)

        self.normalize_drive_output = bool(self.get_parameter('normalize_drive_output').value)
        self.neoracer_max_speed = float(self.get_parameter('neoracer_max_physical_speed_m_s').value)
        self.neoracer_max_steer = float(self.get_parameter('neoracer_max_physical_steer_rad').value)
        self.max_steer_rad = float(self.get_parameter('max_steer_rad').value)
        self.steer_sign = float(self.get_parameter('steer_sign').value)
        self.steer_trim = float(self.get_parameter('steer_trim_normalized').value)

        self.lidar_scale = float(self.get_parameter('lidar_distance_scale').value)
        self.speed_in_scale = float(self.get_parameter('speed_input_scale').value)
        self.speed_out_scale = float(self.get_parameter('speed_output_scale').value)
        self.max_speed_cap = float(self.get_parameter('max_speed_cap_m_s').value)
        self.min_speed_floor = float(self.get_parameter('min_speed_floor_m_s').value)

        self.scan_cw = bool(self.get_parameter('lidar_scan_clockwise').value)
        self.shift_zero_rear = bool(self.get_parameter('shift_index_zero_to_rear').value)
        self.blind_min_deg = float(self.get_parameter('rear_blind_angle_min_deg').value)
        self.blind_max_deg = float(self.get_parameter('rear_blind_angle_max_deg').value)
        self.blind_fill_m = float(self.get_parameter('rear_blind_fill_distance_m').value)
        self.max_range = float(self.get_parameter('max_lidar_range_m').value)
        self.min_range = float(self.get_parameter('min_lidar_range_m').value)
        self.corridor_hold_sec = float(self.get_parameter('side_corridor_hold_sec').value)
        self.median_win = int(self.get_parameter('median_filter_window').value)

        self.wheelbase = float(self.get_parameter('wheelbase_m').value)
        self.ttc_warn = float(self.get_parameter('ttc_warning_sec').value)
        self.ttc_crit = float(self.get_parameter('ttc_critical_sec').value)
        self.ttc_ebrake_speed = float(self.get_parameter('ttc_ebrake_speed_m_s').value)

        self.enable_traction = bool(self.get_parameter('enable_traction_circle').value)
        self.max_lat_accel = float(self.get_parameter('max_lat_accel_m_s2').value)
        self.traction_min_ratio = float(self.get_parameter('traction_min_speed_ratio').value)

        self.enable_yaw_damping = bool(self.get_parameter('enable_yaw_damping').value)
        self.max_drift_yaw_err = float(self.get_parameter('max_drift_yaw_error_rad_s').value)
        self.yaw_kp = float(self.get_parameter('yaw_damping_kp').value)

        self.ema_alpha = float(self.get_parameter('steering_ema_alpha').value)
        self.max_steer_rate = float(self.get_parameter('max_steering_rate_rad_s').value)

        # --- 3. Torch Device & State Initialization ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = (
            torch.float16
            if (self.use_fp16 and self.device.type == 'cuda')
            else torch.float32
        )

        self.latest_scan = None
        self.last_scan_time = 0.0
        self.measured_speed_mps = 0.0
        self.measured_yaw_rate = 0.0
        self.prev_steer_rad = 0.0

        # Side-corridor obstacle hold state
        self.left_corridor_min_m = self.max_range
        self.right_corridor_min_m = self.max_range
        self.left_corridor_ts = 0.0
        self.right_corridor_ts = 0.0

        # Recurrent GRU hidden state: (1, 1, 420)
        self.h_state = torch.zeros((1, 1, self.gru_hidden_dim), dtype=self.dtype, device=self.device)
        self.jit_model = None
        if self.mode == 'end2race':
            self._load_model()

        # --- 4. ROS 2 Subscriptions & Publishers ---
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        control_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.create_subscription(LaserScan, self.get_parameter('scan_topic').value, self.scan_cb, sensor_qos)
        self.create_subscription(Float32, self.get_parameter('speed_topic').value, self.speed_cb, sensor_qos)
        self.create_subscription(Odometry, self.get_parameter('odom_topic').value, self.odom_cb, sensor_qos)
        self.create_subscription(Imu, self.get_parameter('imu_topic').value, self.imu_cb, sensor_qos)
        self.create_subscription(Joy, self.get_parameter('joy_topic').value, self.joy_cb, sensor_qos)

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, self.get_parameter('drive_topic').value, control_qos
        )
        self.dotmatrix_pub = self.create_publisher(
            String, self.get_parameter('dotmatrix_topic').value, 10
        )

        self.timer = self.create_timer(self.dt, self.control_loop)
        self.get_logger().info(
            f"End2RacePolicyNode started | mode={self.mode} | device={self.device} ({self.dtype}) | "
            f"max_speed_cap={self.max_speed_cap} m/s"
        )

    def _load_model(self):
        try:
            self.jit_model = torch.jit.load(self.model_path, map_location=self.device)
            self.jit_model = self.jit_model.half() if self.dtype == torch.float16 else self.jit_model.float()
            self.jit_model.eval()

            # Warmup CUDA kernels
            dummy_lidar = torch.full((1, self.num_beams), 5.0, dtype=self.dtype, device=self.device)
            dummy_speed = torch.zeros((1, 1), dtype=self.dtype, device=self.device)
            dummy_h = torch.zeros((1, 1, self.gru_hidden_dim), dtype=self.dtype, device=self.device)
            with torch.inference_mode():
                _, _ = self.jit_model(dummy_lidar, dummy_speed, dummy_h)

            self.get_logger().info(f"Loaded TorchScript GRU policy: {self.model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load model '{self.model_path}': {e}")
            self.jit_model = None

    def scan_cb(self, msg: LaserScan):
        self.latest_scan = msg
        self.last_scan_time = time.monotonic()

    def speed_cb(self, msg: Float32):
        self.measured_speed_mps = float(msg.data)

    def odom_cb(self, msg: Odometry):
        self.measured_speed_mps = float(msg.twist.twist.linear.x)

    def imu_cb(self, msg: Imu):
        self.measured_yaw_rate = float(msg.angular_velocity.z)

    def joy_cb(self, msg: Joy):
        pass

    def reset_gru_state(self):
        self.h_state.zero_()

    def preprocess_lakibeam_scan(self, scan_msg: LaserScan, now: float) -> tuple[np.ndarray, float]:
        """
        Converts raw LakiBeam1 scan (or sim scan) to 180-beam CCW End2Race input
        and returns (beams_180_scaled, forward_min_dist_m).
        """
        raw = np.array(scan_msg.ranges, dtype=np.float32)
        n_raw = len(raw)
        raw = np.nan_to_num(raw, nan=self.max_range, posinf=self.max_range, neginf=self.min_range)
        raw = np.clip(raw, self.min_range, self.max_range)

        if self.median_win > 1:
            raw = median_filter(raw, size=self.median_win, mode='wrap')

        # Construct angle array in degrees [0, 360) matching raw index order
        angles_deg = np.linspace(0.0, 360.0, n_raw, endpoint=False)

        # 1. Fill LakiBeam1 rear blind wedge (e.g., 135 deg to 225 deg)
        blind_mask = (angles_deg >= self.blind_min_deg) & (angles_deg <= self.blind_max_deg)
        raw[blind_mask] = self.blind_fill_m

        # 2. Side-corridor temporal persistence (90-135 deg & 225-270 deg) for safe overtaking
        if self.corridor_hold_sec > 0.0:
            q1_mask = (angles_deg >= 90.0) & (angles_deg < self.blind_min_deg)
            q2_mask = (angles_deg > self.blind_max_deg) & (angles_deg <= 270.0)
            if np.any(q1_mask):
                cur_q1 = float(np.min(raw[q1_mask]))
                if cur_q1 < self.right_corridor_min_m or (now - self.right_corridor_ts > self.corridor_hold_sec):
                    self.right_corridor_min_m = cur_q1
                    self.right_corridor_ts = now
                else:
                    raw[q1_mask] = np.minimum(raw[q1_mask], self.right_corridor_min_m)
            if np.any(q2_mask):
                cur_q2 = float(np.min(raw[q2_mask]))
                if cur_q2 < self.left_corridor_min_m or (now - self.left_corridor_ts > self.corridor_hold_sec):
                    self.left_corridor_min_m = cur_q2
                    self.left_corridor_ts = now
                else:
                    raw[q2_mask] = np.minimum(raw[q2_mask], self.left_corridor_min_m)

        # Forward +/- 12 deg cone minimum distance for TTC calculation (around index 0 if 0=forward)
        fwd_mask = (angles_deg <= 12.0) | (angles_deg >= 348.0)
        fwd_min_dist = float(np.min(raw[fwd_mask])) if np.any(fwd_mask) else self.max_range

        # 3. Convert Clockwise (CW) LakiBeam1 to Counter-Clockwise (CCW) F1TENTH convention
        if self.scan_cw:
            # Keep index 0 (0 deg) fixed and reverse remaining elements so angle increases CCW
            raw = np.concatenate([raw[:1], raw[1:][::-1]])

        # 4. Shift index 0 from 0 deg (front) to -180 deg (rear) so beam 90 is straight ahead
        if self.shift_zero_rear:
            raw = np.roll(raw, shift=n_raw // 2)

        # 5. Downsample to 180 beams
        if n_raw != self.num_beams:
            idx = np.linspace(0, n_raw - 1, self.num_beams, dtype=int)
            beams_180 = raw[idx]
        else:
            beams_180 = raw

        # 6. Sim-to-real track-width distance scaling
        beams_180_scaled = np.clip(beams_180 * self.lidar_scale, self.min_range, self.max_range)

        return beams_180_scaled.astype(np.float32), fwd_min_dist

    def compute_ftg_command(self, beams_180: np.ndarray) -> tuple[float, float]:
        """Simple Follow-the-Gap baseline on front 180 deg of the 180-beam CCW array."""
        front_slice = beams_180[45:135]  # -90 deg to +90 deg around beam 90
        best_idx = int(np.argmax(front_slice))
        angle_rad = ((best_idx - 45) / 90.0) * (math.pi / 2.0)
        steer_rad = float(np.clip(angle_rad, -self.max_steer_rad, self.max_steer_rad))
        speed_mps = float(min(self.max_speed_cap, max(1.0, front_slice[best_idx] * 0.5)))
        return steer_rad, speed_mps

    def publish_drive(self, physical_steer_rad: float, physical_speed_mps: float):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        if self.normalize_drive_output:
            norm_steer = (physical_steer_rad / max(self.neoracer_max_steer, 1e-4)) * self.steer_sign + self.steer_trim
            norm_speed = physical_speed_mps / max(self.neoracer_max_speed, 1e-4)
            msg.drive.steering_angle = float(np.clip(norm_steer, -1.0, 1.0))
            msg.drive.speed = float(np.clip(norm_speed, -1.0, 1.0))
        else:
            msg.drive.steering_angle = float(physical_steer_rad * self.steer_sign)
            msg.drive.speed = float(physical_speed_mps)

        self.drive_pub.publish(msg)

    def control_loop(self):
        now = time.monotonic()

        # 1. Sensor Watchdog & Stale GRU Reset
        if self.latest_scan is None or (now - self.last_scan_time > self.stale_timeout):
            self.reset_gru_state()
            self.publish_drive(0.0, 0.0)
            return

        # 2. Preprocess Scan
        beams_180, fwd_min_dist = self.preprocess_lakibeam_scan(self.latest_scan, now)

        # 3. Compute Policy or FTG Action (in physical units: rad, m/s)
        if self.mode == 'ftg' or self.jit_model is None:
            target_steer_rad, target_speed_mps = self.compute_ftg_command(beams_180)
        else:
            scaled_input_speed = self.measured_speed_mps * self.speed_in_scale
            with torch.inference_mode():
                lidar_t = torch.from_numpy(beams_180).unsqueeze(0).to(device=self.device, dtype=self.dtype)
                speed_t = torch.tensor([[scaled_input_speed]], dtype=self.dtype, device=self.device)
                action_t, self.h_state = self.jit_model(lidar_t, speed_t, self.h_state)
                raw_action = action_t[0].float().cpu().numpy()

            raw_steer = float(raw_action[self.steer_idx])
            raw_speed = float(raw_action[self.speed_idx])

            if self.actions_are_normalized:
                target_steer_rad = np.clip(raw_steer, -1.0, 1.0) * self.max_steer_rad
                target_speed_mps = (
                    self.min_speed_floor
                    + 0.5 * (np.clip(raw_speed, -1.0, 1.0) + 1.0) * (self.max_speed_cap - self.min_speed_floor)
                )
            else:
                target_steer_rad = float(np.clip(raw_steer, -self.max_steer_rad, self.max_steer_rad))
                target_speed_mps = float(raw_speed * self.speed_out_scale)

        # 4. Actuator Rate Limiting & EMA Smoothing on Steering
        max_delta_steer = self.max_steer_rate * self.dt
        rate_limited_steer = float(
            np.clip(target_steer_rad, self.prev_steer_rad - max_delta_steer, self.prev_steer_rad + max_delta_steer)
        )
        cmd_steer_rad = self.ema_alpha * rate_limited_steer + (1.0 - self.ema_alpha) * self.prev_steer_rad

        # 5. IMU Yaw-Rate Drift Assist
        if self.enable_yaw_damping and abs(self.measured_speed_mps) > 0.5:
            kinematic_yaw_rate = (self.measured_speed_mps / max(self.wheelbase, 0.1)) * math.tan(cmd_steer_rad)
            yaw_err = self.measured_yaw_rate - kinematic_yaw_rate
            if abs(yaw_err) > self.max_drift_yaw_err:
                excess_err = yaw_err - math.copysign(self.max_drift_yaw_err, yaw_err)
                cmd_steer_rad -= self.yaw_kp * excess_err

        cmd_steer_rad = float(np.clip(cmd_steer_rad, -self.max_steer_rad, self.max_steer_rad))
        self.prev_steer_rad = cmd_steer_rad

        # 6. Friction-Circle Cornering Speed Limit
        cmd_speed_mps = float(np.clip(target_speed_mps, self.min_speed_floor, self.max_speed_cap))
        if self.enable_traction and abs(cmd_steer_rad) > 0.03:
            curvature = abs(math.tan(cmd_steer_rad)) / max(self.wheelbase, 0.1)
            v_lat_max = math.sqrt(max(self.max_lat_accel / max(curvature, 1e-3), 0.0))
            min_allowed = self.traction_min_speed_ratio * cmd_speed_mps
            cmd_speed_mps = float(max(min_allowed, min(cmd_speed_mps, v_lat_max)))

        # 7. Forward Time-To-Collision (TTC) Safety Override
        eff_speed = max(self.measured_speed_mps, cmd_speed_mps, 0.1)
        ttc = fwd_min_dist / eff_speed
        if ttc <= self.ttc_crit:
            cmd_speed_mps = self.ttc_ebrake_speed
        elif ttc <= self.ttc_warn:
            ratio = (ttc - self.ttc_crit) / max(self.ttc_warn - self.ttc_crit, 1e-3)
            cmd_speed_mps *= float(np.clip(ratio, 0.0, 1.0))

        self.publish_drive(cmd_steer_rad, cmd_speed_mps)


def main(args=None):
    rclpy.init(args=args)
    node = End2RacePolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_drive(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()