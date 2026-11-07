#!/usr/bin/env python3
import time
from collections import deque
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False


class End2RacePolicyNode(Node):
    def __init__(self):
        super().__init__('end2race_policy_node')

        # Declare parameters
        self.declare_parameter('model_path', 'models/end2race_policy.onnx')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('control_rate_hz', 50.0)
        self.declare_parameter('watchdog_timeout_sec', 0.25)

        # LiDAR preprocessing parameters
        self.declare_parameter('num_beams', 108)
        self.declare_parameter('max_range', 30.0)
        self.declare_parameter('min_range', 0.05)
        self.declare_parameter('fov_deg', 270.0)
        self.declare_parameter('history_len', 4)

        # Action scaling / safety limits
        self.declare_parameter('max_steer_rad', 0.4189)  # ~24 deg
        self.declare_parameter('max_speed_mps', 8.0)
        self.declare_parameter('min_speed_mps', 0.0)
        self.declare_parameter('steer_ema_alpha', 0.7)   # Low-pass filter on steering

        # Read parameters
        self.model_path = self.get_parameter('model_path').value
        scan_topic = self.get_parameter('scan_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        drive_topic = self.get_parameter('drive_topic').value
        control_rate = self.get_parameter('control_rate_hz').value
        self.watchdog_timeout = self.get_parameter('watchdog_timeout_sec').value

        self.num_beams = self.get_parameter('num_beams').value
        self.max_range = self.get_parameter('max_range').value
        self.min_range = self.get_parameter('min_range').value
        self.history_len = self.get_parameter('history_len').value

        self.max_steer = self.get_parameter('max_steer_rad').value
        self.max_speed = self.get_parameter('max_speed_mps').value
        self.min_speed = self.get_parameter('min_speed_mps').value
        self.alpha = self.get_parameter('steer_ema_alpha').value

        # State buffers
        self.latest_scan = None
        self.latest_odom = None
        self.last_scan_time = 0.0
        self.last_odom_time = 0.0
        self.prev_steer = 0.0
        self.prev_speed = 0.0

        # Single-step feature dimension: num_beams + [vx, vy, yaw_rate, prev_steer, prev_speed]
        self.single_obs_dim = self.num_beams + 5
        self.obs_history = deque(maxlen=self.history_len)

        # Load ONNX model
        self.ort_session = None
        self.input_name = None
        self._load_model()

        # QoS profiles
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

        # Subscribers & Publisher
        self.scan_sub = self.create_subscription(
            LaserScan, scan_topic, self.scan_callback, sensor_qos
        )
        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, sensor_qos
        )
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, drive_topic, control_qos
        )

        # Control loop timer
        self.timer = self.create_timer(1.0 / control_rate, self.control_loop)
        self.get_logger().info(
            f"End2RacePolicyNode initialized at {control_rate} Hz | Model: {self.model_path}"
        )

    def _load_model(self):
        if not HAS_ORT:
            self.get_logger().error("onnxruntime is not installed! Running in dummy zero-output mode.")
            return
        try:
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 2
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.ort_session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.input_name = self.ort_session.get_inputs()[0].name
            self.get_logger().info(
                f"Loaded ONNX model successfully. Input: '{self.input_name}' "
                f"shape={self.ort_session.get_inputs()[0].shape}"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to load ONNX model at '{self.model_path}': {e}")
            self.ort_session = None

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg
        self.last_scan_time = time.monotonic()

    def odom_callback(self, msg: Odometry):
        self.latest_odom = msg
        self.last_odom_time = time.monotonic()

    def preprocess_scan(self, scan_msg: LaserScan) -> np.ndarray:
        ranges = np.array(scan_msg.ranges, dtype=np.float32)

        # Replace NaNs with min_range (obstacle collision precaution) and Infs with max_range
        ranges = np.nan_to_num(
            ranges,
            nan=self.min_range,
            posinf=self.max_range,
            neginf=self.min_range
        )
        ranges = np.clip(ranges, self.min_range, self.max_range)

        # Uniform index downsampling to target num_beams
        if len(ranges) != self.num_beams:
            indices = np.linspace(0, len(ranges) - 1, self.num_beams, dtype=int)
            ranges = ranges[indices]

        # Normalize to [0, 1]
        norm_ranges = ranges / self.max_range
        return norm_ranges

    def extract_ego_state(self, odom_msg: Odometry) -> np.ndarray:
        vx = odom_msg.twist.twist.linear.x / max(self.max_speed, 1.0)
        vy = odom_msg.twist.twist.linear.y / max(self.max_speed, 1.0)
        yaw_rate = odom_msg.twist.twist.angular.z / 5.0  # Normalized by nominal max yaw rate
        norm_prev_steer = self.prev_steer / max(self.max_steer, 1e-4)
        norm_prev_speed = self.prev_speed / max(self.max_speed, 1.0)
        return np.array([vx, vy, yaw_rate, norm_prev_steer, norm_prev_speed], dtype=np.float32)

    def publish_stop(self):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.steering_angle = 0.0
        msg.drive.speed = 0.0
        self.drive_pub.publish(msg)

    def control_loop(self):
        now = time.monotonic()

        # Sensor freshness watchdog
        if self.latest_scan is None or self.latest_odom is None:
            return
        if (now - self.last_scan_time > self.watchdog_timeout) or (
            now - self.last_odom_time > self.watchdog_timeout
        ):
            self.get_logger().warn(
                "Sensor timeout exceeded watchdog limit! Publishing zero-velocity stop command.",
                throttle_duration_sec=1.0
            )
            self.publish_stop()
            return

        # Build current step observation
        scan_feat = self.preprocess_scan(self.latest_scan)
        ego_feat = self.extract_ego_state(self.latest_odom)
        step_obs = np.concatenate([scan_feat, ego_feat], axis=0)

        # Populate history buffer on first frame
        if len(self.obs_history) == 0:
            for _ in range(self.history_len):
                self.obs_history.append(step_obs)
        else:
            self.obs_history.append(step_obs)

        # Flatten stacked history: shape (1, history_len * single_obs_dim)
        stacked_obs = np.concatenate(list(self.obs_history), axis=0).astype(np.float32)
        obs_tensor = np.expand_dims(stacked_obs, axis=0)

        # Inference
        if self.ort_session is not None:
            outputs = self.ort_session.run(None, {self.input_name: obs_tensor})
            raw_action = outputs[0][0]  # Expected in [-1, 1] for [steer, speed]
            raw_steer = float(np.clip(raw_action[0], -1.0, 1.0))
            raw_speed = float(np.clip(raw_action[1], -1.0, 1.0))

            target_steer = raw_steer * self.max_steer
            # Map [-1, 1] -> [min_speed, max_speed]
            target_speed = self.min_speed + 0.5 * (raw_speed + 1.0) * (self.max_speed - self.min_speed)
        else:
            target_steer = 0.0
            target_speed = 0.0

        # Exponential moving average filter on steering to suppress high-frequency chattering
        cmd_steer = self.alpha * target_steer + (1.0 - self.alpha) * self.prev_steer
        cmd_speed = float(np.clip(target_speed, self.min_speed, self.max_speed))

        self.prev_steer = cmd_steer
        self.prev_speed = cmd_speed

        # Publish Ackermann command
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.steering_angle = float(cmd_steer)
        drive_msg.drive.speed = float(cmd_speed)
        self.drive_pub.publish(drive_msg)


def main(args=None):
    rclpy.init(args=args)
    node = End2RacePolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()