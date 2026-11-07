#!/usr/bin/env python3
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import LaserScan
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue


class LidarDiagnosticsNode(Node):
    def __init__(self):
        super().__init__('lidar_diagnostics_node')

        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('expected_hz', 30.0)
        self.declare_parameter('hz_tolerance_ratio', 0.2)
        self.declare_parameter('max_invalid_beam_ratio', 0.15)
        self.declare_parameter('min_safe_distance_m', 0.25)

        scan_topic = self.get_parameter('scan_topic').value
        self.expected_hz = self.get_parameter('expected_hz').value
        self.hz_tol = self.get_parameter('hz_tolerance_ratio').value
        self.max_invalid_ratio = self.get_parameter('max_invalid_beam_ratio').value
        self.min_safe_dist = self.get_parameter('min_safe_distance_m').value

        self.frame_timestamps = []
        self.latest_scan = None

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.scan_sub = self.create_subscription(
            LaserScan, scan_topic, self.scan_callback, sensor_qos
        )
        self.diag_pub = self.create_publisher(DiagnosticArray, '/diagnostics', 10)
        self.timer = self.create_timer(0.5, self.publish_diagnostics)

        self.get_logger().info(f"LidarDiagnosticsNode monitoring {scan_topic}")

    def scan_callback(self, msg: LaserScan):
        now = time.monotonic()
        self.latest_scan = msg
        self.frame_timestamps.append(now)
        # Keep only timestamps within the last 2.0 seconds
        self.frame_timestamps = [t for t in self.frame_timestamps if now - t <= 2.0]

    def publish_diagnostics(self):
        now = time.monotonic()
        diag_array = DiagnosticArray()
        diag_array.header.stamp = self.get_clock().now().to_msg()

        status = DiagnosticStatus()
        status.name = 'end2race_ros2: LiDAR Health'
        status.hardware_id = 'planar_lidar_2d'

        if self.latest_scan is None or len(self.frame_timestamps) == 0 or (
            now - self.frame_timestamps[-1] > 0.5
        ):
            status.level = DiagnosticStatus.ERROR
            status.message = 'No recent LaserScan messages received'
            diag_array.status.append(status)
            self.diag_pub.publish(diag_array)
            return

        # Compute scan rate
        if len(self.frame_timestamps) >= 2:
            dt = self.frame_timestamps[-1] - self.frame_timestamps[0]
            measured_hz = (len(self.frame_timestamps) - 1) / max(dt, 1e-4)
        else:
            measured_hz = 0.0

        # Analyze range validity
        ranges = np.array(self.latest_scan.ranges, dtype=np.float32)
        total_beams = len(ranges)
        invalid_mask = ~np.isfinite(ranges) | (ranges < self.latest_scan.range_min)
        invalid_ratio = float(np.sum(invalid_mask)) / max(total_beams, 1)

        valid_ranges = ranges[~invalid_mask]
        min_dist = float(np.min(valid_ranges)) if len(valid_ranges) > 0 else 0.0

        # Determine health level
        level = DiagnosticStatus.OK
        messages = []

        if abs(measured_hz - self.expected_hz) > self.expected_hz * self.hz_tol:
            level = max(level, DiagnosticStatus.WARN)
            messages.append(f"Rate {measured_hz:.1f}Hz deviates from {self.expected_hz:.1f}Hz")

        if invalid_ratio > self.max_invalid_ratio:
            level = max(level, DiagnosticStatus.WARN)
            messages.append(f"High invalid beam ratio ({invalid_ratio * 100:.1f}%)")

        if min_dist < self.min_safe_dist:
            level = max(level, DiagnosticStatus.WARN)
            messages.append(f"Obstacle proximity warning ({min_dist:.2f}m)")

        status.level = level
        status.message = '; '.join(messages) if messages else 'OK'
        status.values = [
            KeyValue(key='measured_rate_hz', value=f"{measured_hz:.2f}"),
            KeyValue(key='total_beams', value=str(total_beams)),
            KeyValue(key='invalid_beam_ratio', value=f"{invalid_ratio:.4f}"),
            KeyValue(key='min_valid_range_m', value=f"{min_dist:.3f}"),
        ]

        diag_array.status.append(status)
        self.diag_pub.publish(diag_array)


def main(args=None):
    rclpy.init(args=args)
    node = LidarDiagnosticsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()