#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from collections import deque


class FeatureTracker(Node):
    """
    Per-camera ORB feature detection + frame-to-frame matching.
    Phase 1 exit metric: median detection count per stereo camera.
    """

    def __init__(self):
        super().__init__('feature_tracker')

        self.declare_parameter('cameras', ['cam_front/left', 'cam_front/right'])
        self.declare_parameter('show_visualization', False)

        cameras = list(self.get_parameter('cameras').get_parameter_value().string_array_value)
        self.show_visualization = bool(self.get_parameter('show_visualization').value)

        self.bridge = CvBridge()
        self.detector = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Per-camera state and rolling stats.
        self.state = {
            cam: {
                'prev_gray': None,
                'prev_kp': None,
                'prev_desc': None,
                'prev_img': None,
                'detect_counts': deque(maxlen=100),
                'match_counts': deque(maxlen=100),
                'match_rates': deque(maxlen=100),
            }
            for cam in cameras
        }

        for cam in cameras:
            topic = f'/{cam}/image_raw'
            # c=cam binds the loop variable into the lambda's default arg.
            self.create_subscription(
                Image, topic,
                lambda msg, c=cam: self.track_features(msg, c),
                10,
            )
            self.get_logger().info(f"Subscribed to {topic}")

        self.create_timer(10.0, self.print_periodic_stats)

        self.get_logger().info("Feature tracker ready — Ctrl+C for final stats.")

    def decode_gray(self, msg):
        if msg.encoding in ('mono8', '8UC1'):
            return self.bridge.imgmsg_to_cv2(msg, 'mono8')
        bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    def track_features(self, msg, cam):
        s = self.state[cam]

        gray = self.decode_gray(msg)
        kp, desc = self.detector.detectAndCompute(gray, None)

        if desc is None or len(kp) < 10:
            self.get_logger().warn(f"[{cam}] only {len(kp)} features detected")
            return

        s['detect_counts'].append(len(kp))

        if s['prev_desc'] is not None and len(s['prev_desc']) > 0:
            try:
                matches = self.matcher.match(desc, s['prev_desc'])
                matches = sorted(matches, key=lambda m: m.distance)
                good = [m for m in matches if m.distance < 50]

                match_count = len(good)
                match_rate = (match_count / len(kp)) * 100.0
                s['match_counts'].append(match_count)
                s['match_rates'].append(match_rate)

                status = "GOOD" if match_count > 100 else "LOW" if match_count > 50 else "POOR"
                self.get_logger().info(
                    f"[{cam}] detect={len(kp):4d} match={match_count:4d} "
                    f"({match_rate:5.1f}%) {status}"
                )

                if self.show_visualization and good:
                    self.visualize_matches(cam, msg, gray, kp, s, good)

            except cv2.error as e:
                self.get_logger().error(f"[{cam}] matching error: {e}")

        s['prev_gray'] = gray
        s['prev_img'] = self.bridge.imgmsg_to_cv2(msg, 'bgr8') if msg.encoding not in ('mono8', '8UC1') else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        s['prev_kp'] = kp
        s['prev_desc'] = desc

    def visualize_matches(self, cam, msg, gray, kp, s, matches):
        cur_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if s['prev_img'] is None:
            return

        match_img = cv2.drawMatches(
            s['prev_img'], s['prev_kp'],
            cur_img, kp,
            matches[:50],
            None,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        cv2.putText(
            match_img, f"{cam}  matches={len(matches)}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
        )
        cv2.imshow(f'Feature Tracking — {cam}', match_img)
        cv2.waitKey(1)

    def print_periodic_stats(self):
        for cam, s in self.state.items():
            if not s['detect_counts']:
                continue
            d_med = float(np.median(s['detect_counts']))
            m_med = float(np.median(s['match_counts'])) if s['match_counts'] else 0.0
            r_med = float(np.median(s['match_rates'])) if s['match_rates'] else 0.0
            self.get_logger().info(
                f"[{cam:<16}] detect_med={d_med:6.1f}  match_med={m_med:6.1f}  "
                f"({r_med:5.1f}%)  frames={len(s['detect_counts'])}"
            )

    def print_statistics(self):
        any_data = any(s['detect_counts'] for s in self.state.values())
        if not any_data:
            self.get_logger().warn("No tracking data collected")
            return

        print("\n" + "=" * 70)
        print("FEATURE TRACKING — PHASE 1 EXIT METRIC (per camera)")
        print("=" * 70)

        for cam, s in self.state.items():
            if not s['detect_counts']:
                print(f"\n=== {cam} ===  (no frames)")
                continue

            d = np.array(s['detect_counts'])
            print(f"\n=== {cam} ===")
            print(f"Frames: {len(d)}")
            print(f"Detections: median={np.median(d):.1f}, mean={np.mean(d):.1f} "
                  f"± {np.std(d):.1f}, range {np.min(d):.0f}–{np.max(d):.0f}")

            if s['match_counts']:
                m = np.array(s['match_counts'])
                r = np.array(s['match_rates'])
                print(f"Matches:    median={np.median(m):.1f}, mean={np.mean(m):.1f} "
                      f"± {np.std(m):.1f}, match rate {np.mean(r):.1f}%")
                avg_m = float(np.mean(m))
                if avg_m > 150:
                    rating = "EXCELLENT — strong VIO prospect"
                elif avg_m > 100:
                    rating = "GOOD — sufficient for VIO"
                elif avg_m > 50:
                    rating = "ACCEPTABLE — VIO marginal"
                else:
                    rating = "POOR — VIO will struggle"
                print(f"VIO quality: {rating}")
            else:
                print("Matches:    (only one frame received)")

        print("=" * 70 + "\n")


def main(args=None):
    rclpy.init(args=args)
    node = FeatureTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nStopping feature tracker...")
    finally:
        node.print_statistics()
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
