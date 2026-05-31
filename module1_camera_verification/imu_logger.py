#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import numpy as np
import csv
from collections import deque

class IMULogger(Node):
    """
    Log IMU data and compute statistics
    Goal: Understand IMU noise characteristics
    """
    
    def __init__(self):
        super().__init__('imu_logger')
        
        # Storage for statistics
        self.accel_data = {'x': deque(maxlen=1000), 
                          'y': deque(maxlen=1000), 
                          'z': deque(maxlen=1000)}
        self.gyro_data = {'x': deque(maxlen=1000), 
                         'y': deque(maxlen=1000), 
                         'z': deque(maxlen=1000)}
        
        # CSV output (param-controlled so two runs can be labelled separately)
        self.declare_parameter('output_csv', 'imu_log.csv')
        csv_path = self.get_parameter('output_csv').get_parameter_value().string_value
        self.log_file = open(csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow([
            'timestamp',
            'accel_x', 'accel_y', 'accel_z',
            'gyro_x', 'gyro_y', 'gyro_z'
        ])

        # IMU topic: FCU IMU on X500 base by default;
        # pass -p imu_topic:=/cam_front/imu to log the OAK-D's onboard IMU instead.
        self.declare_parameter('imu_topic', '/imu/data')
        imu_topic = self.get_parameter('imu_topic').get_parameter_value().string_value

        self.imu_sub = self.create_subscription(
            Imu, imu_topic,
            self.imu_callback, 10
        )

        self.get_logger().info(f"Logging {imu_topic} -> {csv_path}")
        
        # Statistics timer
        self.create_timer(5.0, self.print_statistics)
        
        self.sample_count = 0
        
        self.get_logger().info("IMU Logger started!")
        self.get_logger().info("Keep drone STATIONARY for 30 seconds")
        self.get_logger().info("This measures noise characteristics")
    
    def imu_callback(self, msg):
        # Extract data
        accel = msg.linear_acceleration
        gyro = msg.angular_velocity
        
        # Store for statistics
        self.accel_data['x'].append(accel.x)
        self.accel_data['y'].append(accel.y)
        self.accel_data['z'].append(accel.z)
        
        self.gyro_data['x'].append(gyro.x)
        self.gyro_data['y'].append(gyro.y)
        self.gyro_data['z'].append(gyro.z)
        
        # Log to CSV
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.csv_writer.writerow([
            timestamp,
            accel.x, accel.y, accel.z,
            gyro.x, gyro.y, gyro.z
        ])
        
        self.sample_count += 1
    
    def print_statistics(self):
        if len(self.accel_data['x']) < 10:
            return
        
        self.get_logger().info("\n" + "=" * 70)
        self.get_logger().info(f"IMU STATISTICS (from {self.sample_count} samples)")
        self.get_logger().info("=" * 70)
        
        # Accelerometer statistics. Per-axis means depend on orientation;
        # use |a| for the gravity check (C-tilt: sim drone rests at ~11.5° pitch).
        self.get_logger().info("\nAccelerometer (m/s²):")
        for axis in ['x', 'y', 'z']:
            data = np.array(self.accel_data[axis])
            mean = np.mean(data)
            std = np.std(data)
            self.get_logger().info(
                f"  {axis.upper()}: mean={mean:7.4f}, std={std:7.4f}"
            )

        ax = float(np.mean(self.accel_data['x']))
        ay = float(np.mean(self.accel_data['y']))
        az = float(np.mean(self.accel_data['z']))
        g_measured = float(np.sqrt(ax*ax + ay*ay + az*az))
        self.get_logger().info(
            f"  |a| = {g_measured:.4f} m/s²  (expected 9.81, error {abs(g_measured - 9.81):.4f})"
        )
        
        # Gyroscope statistics (stationary → should be ~0)
        self.get_logger().info("\nGyroscope (rad/s):")
        for axis in ['x', 'y', 'z']:
            data = np.array(self.gyro_data[axis])
            mean = np.mean(data)
            std = np.std(data)
            
            self.get_logger().info(
                f"  {axis.upper()}: mean={mean:7.4f}, std={std:7.4f}"
            )
            
            # Bias check
            if abs(mean) > 0.01:
                self.get_logger().warn(
                    f"       ⚠ Bias detected! (Should be ~0 when stationary)"
                )
        
        self.get_logger().info("=" * 70 + "\n")
    
    def shutdown(self):
        self.get_logger().info("✓ Saving IMU data to imu_data_module3.csv")
        self.log_file.close()


def main(args=None):
    rclpy.init(args=args)
    node = IMULogger()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nStopping IMU logger...")
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
