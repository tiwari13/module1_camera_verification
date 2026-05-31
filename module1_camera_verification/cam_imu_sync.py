#!/usr/bin/env python3
"""
STEP 4: Stereo and Camera-IMU Header-Stamp Alignment (Phase 1 exit metrics)

Measures, per Phase 1 exit criteria:
  - Stereo-stereo header-stamp offset (target |median| < 1 ms)
  - Cam-IMU   header-stamp offset (target |median| < 5 ms)
  - Sensor publishing rates
  - IMU pre-integration segments anchored to left-camera header times

NOTE on naming: this tool measures *header-stamp alignment* across streams,
not true exposure-vs-sample td. True td estimation (motion correlation,
Kalibr) is Phase 2 work. For sim-with-shared-sim-clock, header alignment is
both necessary and sufficient.

Sign conventions (consistent across code, log, CSV, and report):
  stereo  offset_ms = left_header_time - right_header_time
                       (positive ⇒ left  timestamped after right)
  cam-IMU offset_ms = left_header_time - nearest_imu_header_time
                       (positive ⇒ left  timestamped after IMU sample)

Pairing is computed retrospectively in generate_report() from header-time
buffers, NOT in the per-message callbacks. This prevents off-by-one bias
when the right or IMU message for a given sim tick arrives after the left.

Rejection thresholds prevent startup/loss-of-stream garbage from polluting
the report:
  STEREO_MAX_OFFSET_MS  = 50.0
  CAM_IMU_MAX_OFFSET_MS = 20.0

Run:
  ros2 run module1_camera_verification cam_imu_sync
  # all topics overridable via --ros-args -p left_image_topic:=... etc.

Hold the drone hovering (or gently moving) ~30 s, then Ctrl+C for the full
report. Report + CSVs are written in finally{}, so SIGTERM is safe too.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
import numpy as np
from collections import deque
import csv
import time


STEREO_MAX_OFFSET_MS = 50.0
CAM_IMU_MAX_OFFSET_MS = 20.0


def stamp_to_float(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class CamIMUSync(Node):
    """
    Stereo + camera-IMU temporal characterization for VIO bring-up.
    Left camera is the canonical reference frame (anchors IMU pairing and
    preintegration). Right camera participates only in stereo pairing.
    Pairing is retrospective — callbacks only stamp into buffers.
    """

    def __init__(self):
        super().__init__('cam_imu_sync')

        self.declare_parameter('left_image_topic', '/cam_front/left/image_raw')
        self.declare_parameter('right_image_topic', '/cam_front/right/image_raw')
        self.declare_parameter('imu_topic', '/cam_front/imu')
        self.declare_parameter('report_period_s', 5.0)

        left_topic = self.get_parameter('left_image_topic').get_parameter_value().string_value
        right_topic = self.get_parameter('right_image_topic').get_parameter_value().string_value
        imu_topic = self.get_parameter('imu_topic').get_parameter_value().string_value
        report_period = float(self.get_parameter('report_period_s').value)

        # ── Per-side camera state (bounded) ────────────────────
        self.cam = {
            'left':  {'arrival_times': deque(maxlen=4000),
                      'header_times':  deque(maxlen=4000)},
            'right': {'arrival_times': deque(maxlen=4000),
                      'header_times':  deque(maxlen=4000)},
        }

        # ── IMU state (bounded) ────────────────────────────────
        self.imu_arrival_times = deque(maxlen=40000)
        self.imu_header_times  = deque(maxlen=40000)
        # Full IMU sample history keyed by header time, used for
        # header-time bucketing during preintegration.
        self.imu_history = deque(maxlen=40000)

        # Pairings + preintegration are recomputed from buffers at
        # report time; no on-the-fly accumulation.
        self.stereo_pairs = []
        self.cam_imu_pairs = []
        self.preintegrated_segments = []

        # ── Subscribers ────────────────────────────────────────
        self.create_subscription(Image, left_topic,  self.left_cb,  10)
        self.create_subscription(Image, right_topic, self.right_cb, 10)
        self.create_subscription(Imu,   imu_topic,   self.imu_cb,   50)

        # ── Status timer ───────────────────────────────────────
        self.create_timer(report_period, self.print_live_status)
        self.start_time = time.time()

        self.get_logger().info("=" * 60)
        self.get_logger().info("STEP 4: Stereo + Cam-IMU Header-Stamp Alignment")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"  left : {left_topic}")
        self.get_logger().info(f"  right: {right_topic}")
        self.get_logger().info(f"  imu  : {imu_topic}")
        self.get_logger().info("Hover (or gently move) the drone ~30 s, then Ctrl+C")
        self.get_logger().info("")

    # ── Callbacks (stamp-only, no pairing) ─────────────────────

    def left_cb(self, msg):
        ht = stamp_to_float(msg.header.stamp)
        self.cam['left']['arrival_times'].append(time.time())
        self.cam['left']['header_times'].append(ht)

    def right_cb(self, msg):
        ht = stamp_to_float(msg.header.stamp)
        self.cam['right']['arrival_times'].append(time.time())
        self.cam['right']['header_times'].append(ht)

    def imu_cb(self, msg):
        ht = stamp_to_float(msg.header.stamp)
        self.imu_arrival_times.append(time.time())
        self.imu_header_times.append(ht)
        self.imu_history.append({
            'time': ht,
            'accel': [msg.linear_acceleration.x,
                      msg.linear_acceleration.y,
                      msg.linear_acceleration.z],
            'gyro':  [msg.angular_velocity.x,
                      msg.angular_velocity.y,
                      msg.angular_velocity.z],
        })

    # ── Retrospective Pairing ──────────────────────────────────

    def _compute_pairings(self):
        """Build stereo_pairs, cam_imu_pairs, preintegrated_segments from
        header-time buffers. Apply rejection thresholds. Idempotent —
        safe to call multiple times (overwrites results)."""
        left_h  = list(self.cam['left']['header_times'])
        right_h = list(self.cam['right']['header_times'])
        imu_h   = list(self.imu_header_times)

        # Stereo pairs: for each left, find nearest right within tolerance.
        self.stereo_pairs = []
        if left_h and right_h:
            right_arr = np.array(right_h)
            for lt in left_h:
                idx = int(np.argmin(np.abs(right_arr - lt)))
                rt = float(right_arr[idx])
                off = (lt - rt) * 1000.0  # left - right
                if abs(off) < STEREO_MAX_OFFSET_MS:
                    self.stereo_pairs.append({
                        'left_time': lt, 'right_time': rt, 'offset_ms': off,
                    })

        # Cam-IMU pairs: for each left, find nearest IMU within tolerance.
        self.cam_imu_pairs = []
        if left_h and imu_h:
            imu_arr = np.array(imu_h)
            for lt in left_h:
                idx = int(np.argmin(np.abs(imu_arr - lt)))
                it = float(imu_arr[idx])
                off = (lt - it) * 1000.0  # left - imu
                if abs(off) < CAM_IMU_MAX_OFFSET_MS:
                    self.cam_imu_pairs.append({
                        'left_time': lt, 'imu_time': it, 'offset_ms': off,
                    })

        # Preintegration: bucket IMU samples between consecutive left
        # header timestamps BY HEADER TIME (not arrival order).
        self.preintegrated_segments = []
        imu_hist = list(self.imu_history)
        if len(left_h) >= 2 and imu_hist:
            for prev, cur in zip(left_h[:-1], left_h[1:]):
                samples = [s for s in imu_hist if prev <= s['time'] <= cur]
                if not samples:
                    continue
                self.preintegrated_segments.append({
                    'cam_time': cur,
                    'prev_cam_time': prev,
                    'imu_count': len(samples),
                    'imu_samples': samples,
                    'dt': cur - prev,
                })

    # ── Live Status ────────────────────────────────────────────

    def print_live_status(self):
        elapsed = time.time() - self.start_time

        left_rate  = self._compute_rate(self.cam['left']['arrival_times'])
        right_rate = self._compute_rate(self.cam['right']['arrival_times'])
        imu_rate   = self._compute_rate(self.imu_arrival_times)

        # Quick approximate latest offsets (NOT the full retrospective
        # pairing — that runs once at report time).
        stereo_latest = self._latest_offset(
            self.cam['left']['header_times'],
            self.cam['right']['header_times'],
            STEREO_MAX_OFFSET_MS,
        )
        cam_imu_latest = self._latest_offset(
            self.cam['left']['header_times'],
            self.imu_header_times,
            CAM_IMU_MAX_OFFSET_MS,
        )

        # Most-recent inter-left-frame IMU count (header-time bucketed).
        imu_in_last = "N/A"
        if len(self.cam['left']['header_times']) >= 2 and self.imu_history:
            prev = self.cam['left']['header_times'][-2]
            cur  = self.cam['left']['header_times'][-1]
            count = sum(1 for s in self.imu_history if prev <= s['time'] <= cur)
            imu_in_last = f"{count}"

        self.get_logger().info(
            f"[{elapsed:5.1f}s] "
            f"L:{left_rate:5.1f}Hz R:{right_rate:5.1f}Hz IMU:{imu_rate:6.1f}Hz | "
            f"stereo(L-R):{stereo_latest:>10} | "
            f"cam-imu(L-I):{cam_imu_latest:>10} | "
            f"imu/last-left:{imu_in_last:>4}"
        )

    @staticmethod
    def _latest_offset(left_buf, other_buf, max_ms):
        """Quick approximate offset for the most recent left frame against
        the nearest entry in other_buf, gated by rejection threshold."""
        if not left_buf or not other_buf:
            return "N/A"
        lt = left_buf[-1]
        arr = np.array(other_buf)
        idx = int(np.argmin(np.abs(arr - lt)))
        off = (lt - float(arr[idx])) * 1000.0
        if abs(off) >= max_ms:
            return "rej"
        return f"{off:+.2f} ms"

    def _compute_rate(self, timestamps):
        if len(timestamps) < 2:
            return 0.0
        recent = list(timestamps)[-100:]
        dt = recent[-1] - recent[0]
        if dt <= 0:
            return 0.0
        return (len(recent) - 1) / dt

    # ── Full Report ────────────────────────────────────────────

    def generate_report(self):
        # Single retrospective pass to build all pairings.
        self._compute_pairings()

        print("\n" + "=" * 70)
        print("STEREO + CAM-IMU HEADER-STAMP ALIGNMENT REPORT")
        print("=" * 70)

        # ── 1. Sensor Rates ────────────────────────────────────
        print("\n┌─ 1. SENSOR RATES ─────────────────────────────────┐")
        left_rate  = self._compute_rate(self.cam['left']['arrival_times'])
        right_rate = self._compute_rate(self.cam['right']['arrival_times'])
        imu_rate   = self._compute_rate(self.imu_arrival_times)

        print(f"  cam_left : {left_rate:6.2f} Hz  ({len(self.cam['left']['arrival_times'])} frames)")
        print(f"  cam_right: {right_rate:6.2f} Hz  ({len(self.cam['right']['arrival_times'])} frames)")
        print(f"  imu      : {imu_rate:6.2f} Hz  ({len(self.imu_arrival_times)} samples)")
        if left_rate > 0:
            print(f"  IMU per left frame: {imu_rate/left_rate:.1f}x")

        for side in ('left', 'right'):
            headers = self.cam[side]['header_times']
            if len(headers) > 1:
                dts = np.diff(list(headers)) * 1000.0
                print(f"\n  {side} frame intervals (ms):")
                print(f"    mean={np.mean(dts):6.2f}  std={np.std(dts):6.2f}"
                      f"  min={np.min(dts):6.2f}  max={np.max(dts):6.2f}")

        if len(self.imu_header_times) > 1:
            dts = np.diff(list(self.imu_header_times)) * 1000.0
            print(f"\n  imu sample intervals (ms):")
            print(f"    mean={np.mean(dts):6.2f}  std={np.std(dts):6.2f}"
                  f"  min={np.min(dts):6.2f}  max={np.max(dts):6.2f}")
        print("└──────────────────────────────────────────────────┘")

        # ── 2a. Stereo Header-Stamp Offset ─────────────────────
        print("\n┌─ 2a. STEREO HEADER-STAMP OFFSET (left - right) ───┐")
        n_left  = len(self.cam['left']['header_times'])
        if self.stereo_pairs:
            offs = np.array([p['offset_ms'] for p in self.stereo_pairs])
            rejected = n_left - len(offs)
            print(f"  Pairs accepted: {len(offs)}   rejected (>|{STEREO_MAX_OFFSET_MS:.0f}|ms): {rejected}")
            print(f"  mean   = {np.mean(offs):+7.3f} ms")
            print(f"  std    =  {np.std(offs):7.3f} ms")
            print(f"  median = {np.median(offs):+7.3f} ms")
            print(f"  min    = {np.min(offs):+7.3f} ms")
            print(f"  max    = {np.max(offs):+7.3f} ms")
            stereo_abs = float(np.median(np.abs(offs)))
            print(f"\n  ► |stereo offset| median = {stereo_abs:.3f} ms (Phase 1 target < 1 ms)")
        else:
            stereo_abs = None
            print("  (no stereo pairs survived rejection)")
        print("└──────────────────────────────────────────────────┘")

        # ── 2b. Cam-IMU Header-Stamp Offset ────────────────────
        print("\n┌─ 2b. CAM-IMU HEADER-STAMP OFFSET (left - imu) ────┐")
        if self.cam_imu_pairs:
            offs = np.array([p['offset_ms'] for p in self.cam_imu_pairs])
            rejected = n_left - len(offs)
            print(f"  Pairs accepted: {len(offs)}   rejected (>|{CAM_IMU_MAX_OFFSET_MS:.0f}|ms): {rejected}")
            print(f"  mean   = {np.mean(offs):+7.3f} ms")
            print(f"  std    =  {np.std(offs):7.3f} ms")
            print(f"  median = {np.median(offs):+7.3f} ms")
            print(f"  min    = {np.min(offs):+7.3f} ms")
            print(f"  max    = {np.max(offs):+7.3f} ms")
            ts_off = float(np.median(offs))
            cam_imu_abs = abs(ts_off)
            print(f"\n  ► header-stamp offset (median) = {ts_off:+.3f} ms  (Phase 1 target |·| < 5 ms)")
            print(f"    (true td estimation deferred to Phase 2 — Kalibr)")
        else:
            cam_imu_abs = None
            print("  (no cam-IMU pairs survived rejection)")
        print("└──────────────────────────────────────────────────┘")

        # ── 3. Preintegration Segments (anchored to LEFT) ──────
        print("\n┌─ 3. IMU PREINTEGRATION SEGMENTS (left-anchored) ──┐")
        if self.preintegrated_segments:
            seg_counts = [s['imu_count'] for s in self.preintegrated_segments]
            seg_dts    = [s['dt'] * 1000.0 for s in self.preintegrated_segments]
            print(f"  segments: {len(self.preintegrated_segments)} (header-time bucketed)")
            print(f"  IMU samples / left frame: "
                  f"mean={np.mean(seg_counts):5.1f}  std={np.std(seg_counts):5.1f}  "
                  f"min={np.min(seg_counts):.0f}  max={np.max(seg_counts):.0f}")
            print(f"  segment duration (ms):    "
                  f"mean={np.mean(seg_dts):6.2f}  std={np.std(seg_dts):6.2f}")

            if len(self.preintegrated_segments) > 5:
                seg = self.preintegrated_segments[5]
                dv, dp, da = self._preintegrate_segment(seg)
                print(f"\n  Example preintegration (segment #5):")
                print(f"    Δt = {seg['dt']*1000:.2f} ms over {seg['imu_count']} IMU samples")
                print(f"    Δv = [{dv[0]:+.4f}, {dv[1]:+.4f}, {dv[2]:+.4f}] m/s")
                print(f"    Δp = [{dp[0]:+.6f}, {dp[1]:+.6f}, {dp[2]:+.6f}] m")
                print(f"    Δθ = [{da[0]:+.6f}, {da[1]:+.6f}, {da[2]:+.6f}] rad")
        else:
            print("  (no segments collected)")
        print("└──────────────────────────────────────────────────┘")

        # ── 4. VIO Readiness ───────────────────────────────────
        print("\n┌─ 4. VIO READINESS  (Phase 1 exit gates) ──────────┐")
        issues, recs = [], []

        if left_rate < 10 or right_rate < 10:
            issues.append(f"Camera rate too low (L:{left_rate:.1f} R:{right_rate:.1f} Hz, < 10)")
        elif left_rate < 20 or right_rate < 20:
            recs.append("Camera rate OK but 20+ Hz is better for VIO")

        if imu_rate < 100:
            issues.append(f"IMU rate too low ({imu_rate:.1f} Hz, < 100)")
        elif imu_rate < 200:
            recs.append(f"IMU rate {imu_rate:.1f} Hz; 200+ Hz is ideal (C-rtf may apply)")

        if stereo_abs is not None:
            if stereo_abs > 1.0:
                issues.append(f"Stereo alignment above gate: |L-R| median {stereo_abs:.3f} ms (gate < 1 ms)")
            else:
                print(f"  ✔ Stereo alignment within gate (|L-R| median {stereo_abs:.3f} ms < 1 ms)")

        if cam_imu_abs is not None:
            if cam_imu_abs > 5.0:
                issues.append(f"Cam-IMU offset above gate: |·| {cam_imu_abs:.3f} ms (gate < 5 ms)")
            else:
                print(f"  ✔ Cam-IMU alignment within gate (|·| {cam_imu_abs:.3f} ms < 5 ms)")

        if self.preintegrated_segments:
            avg_imu = np.mean([s['imu_count'] for s in self.preintegrated_segments])
            if avg_imu < 3:
                issues.append(f"Too few IMU samples per left frame ({avg_imu:.1f})")

        if not issues:
            print("  ✔ ALL CHECKS PASSED — Phase 1 sync gates satisfied")
        for i in issues:
            print(f"  ✗ {i}")
        for r in recs:
            print(f"  △ {r}")
        print("└──────────────────────────────────────────────────┘")

        # ── 5. Why This Matters ────────────────────────────────
        print("\n┌─ 5. WHY THIS MATTERS FOR VIO ────────────────────┐")
        print("                                                    ")
        print("  Camera frame k          Camera frame k+1          ")
        print("       │                        │                   ")
        print("       ▼                        ▼                   ")
        print("  ─────●────────────────────────●─────── time       ")
        print("       │  ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑  │                  ")
        print("       │  IMU measurements      │                   ")
        print("       │  (header-time bucketed)│                   ")
        print("       └── These IMU samples ───┘                   ")
        print("           predict how features                     ")
        print("           should move between                      ")
        print("           frames k and k+1                         ")
        print("                                                    ")
        print("  If header timestamps are misaligned across streams,")
        print("  the bucketing and the prediction are both wrong.   ")
        print("└──────────────────────────────────────────────────┘")
        print("\n" + "=" * 70)

    def _preintegrate_segment(self, segment):
        """
        Simplified IMU preintegration for diagnostic display.
        Assumes near-level orientation (gravity along world Z) — full
        treatment is Phase 2/3 work.
        """
        delta_v = np.zeros(3)
        delta_p = np.zeros(3)
        delta_angle = np.zeros(3)
        gravity = np.array([0.0, 0.0, 9.81])

        samples = segment['imu_samples']
        for i in range(1, len(samples)):
            dt = samples[i]['time'] - samples[i-1]['time']
            if dt <= 0 or dt > 0.1:
                continue
            accel = np.array(samples[i]['accel']) - gravity
            gyro  = np.array(samples[i]['gyro'])
            delta_p += delta_v * dt + 0.5 * accel * dt * dt
            delta_v += accel * dt
            delta_angle += gyro * dt
        return delta_v, delta_p, delta_angle

    # ── CSV Output ─────────────────────────────────────────────

    def save_sync_data(self):
        # Recompute is cheap and ensures save_sync_data is callable
        # independently of generate_report.
        if not (self.stereo_pairs or self.cam_imu_pairs or self.preintegrated_segments):
            self._compute_pairings()

        if self.cam_imu_pairs:
            fn = 'cam_imu_sync_data_step4.csv'
            with open(fn, 'w', newline='') as f:
                w = csv.writer(f)
                # offset_ms = left_header_time - nearest_imu_header_time
                w.writerow(['left_time', 'imu_time', 'offset_ms_left_minus_imu'])
                for p in self.cam_imu_pairs:
                    w.writerow([p['left_time'], p['imu_time'], p['offset_ms']])
            print(f"✔ Saved {fn}")

        if self.stereo_pairs:
            fn = 'stereo_sync_data_step4.csv'
            with open(fn, 'w', newline='') as f:
                w = csv.writer(f)
                # offset_ms = left_header_time - right_header_time
                w.writerow(['left_time', 'right_time', 'offset_ms_left_minus_right'])
                for p in self.stereo_pairs:
                    w.writerow([p['left_time'], p['right_time'], p['offset_ms']])
            print(f"✔ Saved {fn}")

        if self.preintegrated_segments:
            fn = 'preintegration_segments_step4.csv'
            with open(fn, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['cam_time', 'prev_cam_time', 'dt_ms', 'imu_count'])
                for s in self.preintegrated_segments:
                    w.writerow([s['cam_time'], s['prev_cam_time'],
                                s['dt'] * 1000.0, s['imu_count']])
            print(f"✔ Saved {fn}")


def main(args=None):
    rclpy.init(args=args)
    node = CamIMUSync()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nStopping sensor analysis...")
    finally:
        print("Generating report...\n")
        node.generate_report()
        node.save_sync_data()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
