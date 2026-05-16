#!/usr/bin/env python3
"""
Verify cameras are publishing image + camera_info on expected ROS topics.

Topic-driven, drone-model-agnostic. The list of cameras to verify is a
ROS parameter, so the same node works for Phase 1 (1 stereo unit + RGB)
and Phase 11 (multiple OAK-D Pro W units in 360 layout).

Default: 3 lenses on the forward OAK-D Pro W (left mono, right mono, RGB).
Override: -p 'cameras:=["/cam_front/left", "/cam_back/left", ...]'
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo


DEFAULT_CAMERAS = [
    '/cam_front/left',
    '/cam_front/right',
    '/cam_front/rgb',
]


class CameraVerifier(Node):
    def __init__(self):
        super().__init__('camera_verifier')

        self.declare_parameter('cameras', DEFAULT_CAMERAS)
        self.declare_parameter('report_period_s', 3.0)

        camera_prefixes = self.get_parameter('cameras').value
        report_period = self.get_parameter('report_period_s').value

        self.cameras = {p: {'image_ok': False, 'info_ok': False, 'info': None}
                        for p in camera_prefixes}

        for prefix in camera_prefixes:
            self.create_subscription(
                Image, f'{prefix}/image_raw',
                lambda msg, p=prefix: self._on_image(p), 10)
            self.create_subscription(
                CameraInfo, f'{prefix}/camera_info',
                lambda msg, p=prefix: self._on_info(p, msg), 10)

        self.create_timer(report_period, self._report)

        self.get_logger().info(
            f"Camera Verifier started. Watching {len(camera_prefixes)} camera(s):")
        for p in camera_prefixes:
            self.get_logger().info(f"  - {p}")

    def _on_image(self, prefix):
        self.cameras[prefix]['image_ok'] = True

    def _on_info(self, prefix, msg):
        self.cameras[prefix]['info_ok'] = True
        self.cameras[prefix]['info'] = msg

    def _report(self):
        self.get_logger().info("=" * 70)
        self.get_logger().info("CAMERA STATUS REPORT")
        self.get_logger().info("=" * 70)

        all_ok = True
        for prefix, status in self.cameras.items():
            img = "✓" if status['image_ok'] else "✗"
            inf = "✓" if status['info_ok'] else "✗"
            ok = status['image_ok'] and status['info_ok']
            if not ok:
                all_ok = False
            self.get_logger().info(
                f"{prefix:30s}: Image={img} Info={inf} [{'OK' if ok else 'FAIL'}]")
            if status['info']:
                k = status['info'].k
                self.get_logger().info(
                    f"  {' ' * 28}  {status['info'].width}x{status['info'].height}, "
                    f"fx={k[0]:.2f} fy={k[4]:.2f} cx={k[2]:.2f} cy={k[5]:.2f}")

        self.get_logger().info("=" * 70)
        if all_ok:
            self.get_logger().info("✓ ALL CAMERAS WORKING")
        else:
            self.get_logger().warn("⚠ Some cameras not responding")
        self.get_logger().info("=" * 70 + "\n")


def main(args=None):
    rclpy.init(args=args)
    node = CameraVerifier()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
