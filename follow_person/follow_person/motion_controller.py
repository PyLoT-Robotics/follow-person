#!/usr/bin/env python3
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from rclpy.node import Node
import math

class MotionController:
    def __init__(self, node: Node):
        self.node = node
        
        # Parameters
        self.max_linear_speed = 0.5  # m/s
        self.max_angular_speed = 1.0  # rad/s
        self.min_obstacle_distance = 0.5  # meters
        self.target_following_distance = 1.2  # meters
        self.distance_tolerance = 0.1  # meters
        self.angle_tolerance = 0.1  # radians
        self.obstacle_memory_time = 2.0  # seconds
        
        # PIDパラメータ
        self.linear_pid = {
            'Kp': 0.5,
            'Ki': 0.1,
            'Kd': 0.2,
            'integral': 0.0,
            'prev_error': 0.0,
            'integral_limit': 1.0
        }
        
        self.angular_pid = {
            'Kp': 2.0,
            'Ki': 0.1,
            'Kd': 0.4,
            'integral': 0.0,
            'prev_error': 0.0,
            'integral_limit': 1.0
        }
        
        # タイムスタンプ
        self.prev_time = self.node.get_clock().now()
        
        # Obstacle memory
        self.obstacle_points = []
        self.obstacle_timestamps = []

    def update_obstacles_pc(self, cloud_msg: PointCloud2):
        """Update obstacle memory from pointcloud data."""
        current_time = self.node.get_clock().now().to_msg().sec
        
        # Remove old obstacles
        valid_indices = []
        for i, timestamp in enumerate(self.obstacle_timestamps):
            if current_time - timestamp < self.obstacle_memory_time:
                valid_indices.append(i)
        
        self.obstacle_points = [self.obstacle_points[i] for i in valid_indices]
        self.obstacle_timestamps = [self.obstacle_timestamps[i] for i in valid_indices]
        
        # Add new obstacles from pointcloud
        for point in pc2.read_points(cloud_msg, skip_nans=True):
            x, y, z = point[:3]
            # Only consider points at robot height and within range
            if 0.1 < z < 0.6 and math.sqrt(x*x + y*y) < 2.0:
                self.obstacle_points.append(np.array([x, y]))
                self.obstacle_timestamps.append(current_time)

    def update_obstacles_scan(self, scan_msg: LaserScan):
        """Update obstacle memory from laser scan data."""
        current_time = self.node.get_clock().now().to_msg().sec
        
        # Convert scan to points
        angles = np.arange(
            scan_msg.angle_min,
            scan_msg.angle_max + scan_msg.angle_increment,
            scan_msg.angle_increment
        )
        ranges = np.array(scan_msg.ranges)
        
        # Filter invalid measurements
        valid_mask = (ranges >= scan_msg.range_min) & (ranges <= scan_msg.range_max)
        valid_angles = angles[valid_mask]
        valid_ranges = ranges[valid_mask]
        
        # Convert to cartesian coordinates
        x = valid_ranges * np.cos(valid_angles)
        y = valid_ranges * np.sin(valid_angles)
        
        # Add new obstacles
        for i in range(len(x)):
            if math.sqrt(x[i]*x[i] + y[i]*y[i]) < 2.0:  # Only consider nearby points
                self.obstacle_points.append(np.array([x[i], y[i]]))
                self.obstacle_timestamps.append(current_time)

    def compute_velocity(self, person_position: np.ndarray, person_confidence: float) -> Twist:
        """Compute safe velocity command to follow person while avoiding obstacles."""
        cmd_vel = Twist()
        
        if person_position is None or person_confidence < 0.3:
            # If person not detected or low confidence, stop
            return cmd_vel
            
        # Calculate distance and angle to person
        distance = np.linalg.norm(person_position[:2])
        angle = math.atan2(person_position[1], person_position[0])
        
        # Check if path to person is clear
        path_clear = self._check_path_to_target(person_position[:2])
        
        # Compute speeds based on distance and angle
        if path_clear:
            linear_speed = self._compute_linear_speed(distance)
            angular_speed = self._compute_angular_speed(angle)
        else:
            # If path blocked, rotate in place to find clear path
            linear_speed = 0.0
            angular_speed = self.max_angular_speed * np.sign(angle)
            
        # Apply confidence scaling
        linear_speed *= person_confidence
        angular_speed *= person_confidence
        
        # Set velocity commands
        cmd_vel.linear.x = linear_speed
        cmd_vel.angular.z = angular_speed
        
        return cmd_vel

    def _check_path_to_target(self, target_pos: np.ndarray) -> bool:
        """Check if path to target is clear of obstacles."""
        if not self.obstacle_points:
            return True
            
        # Create path corridor
        path_length = np.linalg.norm(target_pos)
        path_direction = target_pos / path_length
        corridor_width = self.min_obstacle_distance
        
        # Check each obstacle
        for obstacle in self.obstacle_points:
            # Project obstacle onto path direction
            proj_length = np.dot(obstacle, path_direction)
            
            # Only consider obstacles between robot and target
            if 0 < proj_length < path_length:
                # Calculate perpendicular distance to path
                proj_point = proj_length * path_direction
                perp_dist = np.linalg.norm(obstacle - proj_point)
                
                if perp_dist < corridor_width:
                    return False
                    
        return True

    def _compute_linear_speed(self, distance: float) -> float:
        """PIDによる線形速度の計算"""
        # 目標距離に到達している場合は停止
        if abs(distance - self.target_following_distance) < self.distance_tolerance:
            # PID状態をリセット
            self.linear_pid['integral'] = 0.0
            self.linear_pid['prev_error'] = 0.0
            return 0.0
            
        # 現在時刻とdt（経過時間）の計算
        current_time = self.node.get_clock().now()
        dt = (current_time - self.prev_time).nanoseconds / 1e9  # 秒に変換
        
        # 誤差計算
        error = distance - self.target_following_distance
        
        # 積分項の更新（制限付き）
        self.linear_pid['integral'] += error * dt
        self.linear_pid['integral'] = max(
            -self.linear_pid['integral_limit'],
            min(self.linear_pid['integral_limit'], self.linear_pid['integral'])
        )
        
        # 微分項の計算
        derivative = (error - self.linear_pid['prev_error']) / dt if dt > 0 else 0.0
        
        # PID制御の計算
        speed = (self.linear_pid['Kp'] * error +
                self.linear_pid['Ki'] * self.linear_pid['integral'] +
                self.linear_pid['Kd'] * derivative)
        
        # 状態の更新
        self.linear_pid['prev_error'] = error
        self.prev_time = current_time
        
        # 速度制限の適用
        speed = max(-self.max_linear_speed, min(self.max_linear_speed, speed))
        
        return speed

    def _compute_angular_speed(self, angle: float) -> float:
        """PIDによる角速度の計算"""
        # 目標角度に到達している場合は停止
        if abs(angle) < self.angle_tolerance:
            # PID状態をリセット
            self.angular_pid['integral'] = 0.0
            self.angular_pid['prev_error'] = 0.0
            return 0.0
            
        # 現在時刻とdt（経過時間）の計算
        current_time = self.node.get_clock().now()
        dt = (current_time - self.prev_time).nanoseconds / 1e9  # 秒に変換
        
        # 誤差計算
        error = angle  # 目標角度は0（ターゲットに正面を向ける）
        
        # 積分項の更新（制限付き）
        self.angular_pid['integral'] += error * dt
        self.angular_pid['integral'] = max(
            -self.angular_pid['integral_limit'],
            min(self.angular_pid['integral_limit'], self.angular_pid['integral'])
        )
        
        # 微分項の計算
        derivative = (error - self.angular_pid['prev_error']) / dt if dt > 0 else 0.0
        
        # PID制御の計算
        speed = (self.angular_pid['Kp'] * error +
                self.angular_pid['Ki'] * self.angular_pid['integral'] +
                self.angular_pid['Kd'] * derivative)
        
        # 状態の更新
        self.angular_pid['prev_error'] = error
        self.prev_time = current_time
        
        # 速度制限の適用
        speed = max(-self.max_angular_speed, min(self.max_angular_speed, speed))
        
        return speed
