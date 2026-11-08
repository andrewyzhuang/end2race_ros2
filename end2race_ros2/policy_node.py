#!/usr/bin/env python3
import time
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Float32, String
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

from end2race_ros2.preprocessor import End2RacePreprocessor
from end2race_ros2.ftg_controller import FollowTheGapController
from end2race_ros2.safety_filter import RacingSafetyFilter


class End2RacePolicyNode(Node):
    def __init__(self):
        super().__init__('end2race_policy_node')

        # --- 1. Declare Parameters from policy_params.yaml ---
        self.declare_parameter('mode', 'end2race')
        self.declare_parameter('model_path', 'models/end2race_policy_jit_fp16.pt')
        self.declare_parameter('use_fp16', True)
        self.declare_parameter('control_rate_hz', 30.0)

        self.declare_parameter('num_beams', 180)
        self.declare_parameter('gru_hidden_dim', 420)
        self.declare_parameter('actions_are_normalized', False)
        self.declare_parameter('action_steer_index', 0)
        self.declare_parameter('action_speed_index', 1)
        self.declare_parameter('gru_stale_timeout_sec', 0.25)

        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('speed_topic', '/encoder/speed')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('imu_topic', '/imu/fused')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('dotmatrix_topic', '/dotmatrix/text')

        self.declare_parameter('normalize_drive_output', True)
        self.declare_parameter('neoracer_max_physical_speed_m_s', 6.0)
        self.declare_parameter('neoracer_max_physical_steer_rad', 0.5236)
        self.declare_parameter('max_steer_rad', 0.42)
        self.declare_parameter('steer_sign', -1.0)
        self.declare_parameter('steer_trim_normalized', 0.0)

        self.declare_parameter('lidar_distance_scale', 1.15)
        self.declare_parameter('speed_input_scale', 1.25)
        self.declare_parameter('speed_output_scale', 0.65)
        self.declare_parameter('max_speed_cap_m_s', 2.5)
        self.declare_parameter('min_speed_floor_m_s', 0.0)

        self.declare_parameter('lidar_scan_clockwise', True)
        self.declare_parameter('shift_index_zero_to_rear', True)
        self.declare_parameter('scan_index0_at_rear', False)
        self.declare_parameter('rear_blind_angle_min_deg', 135.0)
        self.declare_parameter('rear_blind_angle_max_deg', 225.0)
        self.declare_parameter('rear_blind_fill_distance_m', 10.0)
        self.declare_parameter('max_lidar_range_m', 10.0)
        self.declare_parameter('min_lidar_range_m', 0.05)
        self.declare_parameter('side_corridor_hold_sec', 0.25)
        self.declare_parameter('median_filter_window', 3)

        self.declare_parameter('wheelbase_m', 0.28)
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
        self.model_path = self.get_parameter('model_path').value
        self.use_fp16 = bool(self.get_parameter('use_fp16').value)
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
        self.scan_index0_at_rear = bool(self.get_parameter('scan_index0_at_rear').value)
        self.blind_min_deg = float(self.get_parameter('rear_blind_angle_min_deg').value)
        self.blind_max_deg = float(self.get_parameter('rear_blind_angle_max_deg').value)
        self.blind_fill_m = float(self.get_parameter('rear_blind_fill_distance_m').value)
        self.max_range = float(self.get_parameter('max_lidar_range_m').value)
        self.min_range = float(self.get_parameter('min_lidar_range_m').value)
        self.corridor_hold_sec = float(self.get_parameter('side_corridor_hold_sec').value)
        self.median_win = int(self.get_parameter('median_filter_window').value)

        self.wheelbase = float(self.get_parameter('wheelbase_m').value)
        self.safety_margin_radius = float(self.get_parameter('safety_margin_radius_m').value)
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

        # --- 3. Instantiate Submodules to Eliminate Code Redundancies ---
        self.preprocessor = End2RacePreprocessor(
            num_output_beams=self.num_beams,
            max_lidar_range_m=self.max_range,
            min_lidar_range_m=self.min_range,
            lidar_distance_scale=self.lidar_scale,
            speed_input_scale=self.speed_in_scale,
            lidar_scan_clockwise=self.scan_cw,
            shift_index_zero_to_rear=self.shift_zero_rear,
            scan_index0_at_rear=self.scan_index0_at_rear,
            rear_blind_angle_min_deg=self.blind_min_deg,
            rear_blind_angle_max_deg=self.blind_max_deg,
            rear_blind_fill_distance_m=self.blind_fill_m,
            side_corridor_hold_sec=self.corridor_hold_sec,
            median_filter_window=self.median_win,
        )

        self.ftg_controller = FollowTheGapController(
            max_steer_rad=self.max_steer_rad,
            max_speed_m_s=self.max_speed_cap,
            min_speed_m_s=self.min_speed_floor,
        )

        self.safety_filter = RacingSafetyFilter(
            wheelbase_m=self.wheelbase,
            safety_margin_radius_m=self.safety_margin_radius,
            ttc_warning_sec=self.ttc_warn,
            ttc_critical_sec=self.ttc_crit,
            ttc_ebrake_speed_m_s=self.ttc_ebrake_speed,
            enable_traction_circle=self.enable_traction,
            max_lat_accel_m_s2=self.max_lat_accel,
            traction_min_speed_ratio=self.traction_min_ratio,
            enable_yaw_damping=self.enable_yaw_damping,
            max_drift_yaw_error_rad_s=self.max_drift_yaw_err,
            yaw_damping_kp=self.yaw_kp,
            steering_ema_alpha=self.ema_alpha,
            max_steering_rate_rad_s=self.max_steer_rate,
            max_steer_rad=self.max_steer_rad,
            max_speed_cap_m_s=self.max_speed_cap,
            min_speed_floor_m_s=self.min_speed_floor,
        )

        # --- 4. Torch Device & State Initialization ---
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

        # Recurrent GRU hidden state: (1, 1, 420)
        self.h_state = torch.zeros((1, 1, self.gru_hidden_dim), dtype=self.dtype, device=self.device)
        self.jit_model = None
        if self.mode == 'end2race':
            self._load_model()

        # --- 5. ROS 2 Subscriptions & Publishers ---
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

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, self.get_parameter('drive_topic').value, control_qos
        )
        self.dotmatrix_pub = self.create_publisher(
            String, self.get_parameter('dotmatrix_topic').value, 10
        )

        self.timer = self.create_timer(self.dt, self.control_loop)
        self.get_logger().info(
            f"End2RacePolicyNode started | mode={self.mode} | control_rate={self.control_rate} Hz | "
            f"device={self.device} ({self.dtype}) | max_speed_cap={self.max_speed_cap} m/s"
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

    def reset_gru_state(self):
        self.h_state.zero_()

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

        # 1. Sensor Watchdog & Stale State Reset
        if self.latest_scan is None or (now - self.last_scan_time > self.stale_timeout):
            self.reset_gru_state()
            self.preprocessor.reset()
            self.safety_filter.reset()
            self.publish_drive(0.0, 0.0)
            return

        # 2. Preprocess Scan & Speed via End2RacePreprocessor
        policy_ranges_180, forward_ccw_180_m, beam_angles_rad, scaled_ego_speed = (
            self.preprocessor.process(self.latest_scan.ranges, self.measured_speed_mps)
        )

        # 3. Compute Actions (Follow-The-Gap or End2Race Neural Policy)
        if self.mode == 'ftg' or self.jit_model is None:
            target_steer_rad, target_speed_mps = self.ftg_controller.compute_action(
                forward_ccw_180_m, beam_angles_rad
            )
        else:
            with torch.inference_mode():
                lidar_t = torch.from_numpy(policy_ranges_180).unsqueeze(0).to(device=self.device, dtype=self.dtype)
                speed_t = torch.tensor([[scaled_ego_speed]], dtype=self.dtype, device=self.device)
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

        # 4. Apply Multi-Stage Dynamic Stability & Safety Filter
        # (Yaw damping drift assist, servo rate/EMA smoothing, friction circle, and cone iTTC override)
        cmd_steer_rad, cmd_speed_mps, reset_gru, min_ttc_sec, status_code = (
            self.safety_filter.filter_commands(
                cmd_steer_rad=target_steer_rad,
                cmd_speed_m_s=target_speed_mps,
                forward_ccw_180_m=forward_ccw_180_m,
                beam_angles_rad=beam_angles_rad,
                ego_speed_m_s=self.measured_speed_mps,
                imu_yaw_rate_rad_s=self.measured_yaw_rate,
            )
        )

        if reset_gru:
            self.reset_gru_state()

        # 5. Publish LED Dot-Matrix Status
        if status_code:
            dot_msg = String()
            dot_msg.data = status_code
            self.dotmatrix_pub.publish(dot_msg)

        # 6. Publish Actuation Command
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
