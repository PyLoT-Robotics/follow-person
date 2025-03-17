import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, CameraInfo
from geometry_msgs.msg import Twist
from realsense2_camera_msgs.msg import RGBD
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO

#!/usr/bin/env python3

class FollowPerson(Node):
    def __init__(self):
        super().__init__('follow_person_node')

        # Create publishers/subscribers.
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        # Create subscribers for camera and lidar data.
        self.create_subscription(Image, '/d455/camera/color/image_raw', self.image_callback, 10)
        self.create_subscription(Image, '/d455/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(CameraInfo, '/d455/camera/color/camera_info', self.camera_info_callback, 10)
        self.bridge = CvBridge()
        self.latest_color = None
        self.latest_depth = None
        self.obstacle_detected = False
        self.camera_info_received = False
        self.obstacle_distance = float('inf')
        self.latest_scan = None

        # Load YOLO model (adjust the weight file as needed)
        self.model = YOLO("yolov8n-pose.pt")

        # Camera intrinsic parameters (example hard-coded values, adjust these)
        self.fx = 645.0
        self.fy = 643.0
        self.cx = 320.0
        self.cy = 240.0

        # Control parameters
        self.follow_distance = 1.0  # desired distance in meters to the person
        self.linear_gain = 0.8
        self.angular_gain = 1.2
        self.obstacle_threshold = 2.5  # stop if obstacle within 0.5m

        self.retangle_width = 0.5
        self.retangle_height = 0.5
        self.safe_paths = [0,5,-5,10,-10,15,-15,20,-20,25,-25,30,-30,35,-35,40,-40,45,-45,50,-50,55,-55,60,-60,65,-65,70,-70,75,-75,80,-80,85,-85,90,-90]
        self.avoid_speed = 0.2
        self.current_avoid_angle = 0
        self.is_avoiding = False
        self.avoid_timeout = 0

        self.create_timer(0.1, self.timer_callback)

    # 新たにカメラ情報コールバックを追加
    def camera_info_callback(self, msg: CameraInfo):
        if not self.camera_info_received:
            # カメラ行列からパラメータを抽出
            # camera_info.Kは3x3行列で、[fx, 0, cx, 0, fy, cy, 0, 0, 1]の形式で格納されている
            self.fx = msg.k[0]  # fx
            self.fy = msg.k[4]  # fy
            #self.cx = msg.k[2]  # cx
            #self.cy = msg.k[5]  # cy
            self.camera_info_received = True
            self.get_logger().info(f'カメラパラメータを更新しました: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}')

    def image_callback(self, msg: Image):
        try:
            self.latest_color = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error('Failed to convert color image: ' + str(e))

    def depth_callback(self, msg: Image):
        try:
            # Assuming the depth image is encoded in 32FC1
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except Exception as e:
            self.get_logger().error('Failed to convert depth image: ' + str(e))

    def scan_callback(self, msg: LaserScan):

        self.latest_scan = msg
        # Check if there is any obstacle in front
        # Assume that the scan angle 0 corresponds to the front.
        # Using a window around 0 degrees if available.
        front_ranges = []
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment
        for i, r in enumerate(msg.ranges):
            angle = angle_min + i * angle_increment
            if abs(angle) > np.radians(155):  # within +/- 10 degrees
                if r > msg.range_min and r < msg.range_max:
                    front_ranges.append(r)
        if front_ranges:
            self.obstacle_distance = min(front_ranges)
            if self.obstacle_distance < self.obstacle_threshold:
                self.obstacle_detected = True
            else:
                self.obstacle_detected = False

        else:
            self.obstacle_detected = False
            self.obstacle_distance = float('inf')

    def check_rectangle_safety(self, person_position_x, person_position_z, yaw_angle=0.0):
        """
        指定された位置を中心とする矩形領域の安全性をチェック
        person_position_x: カメラ座標系でのx座標(m)
        person_position_z: カメラ座標系でのz座標(m)
        yaw_angle: ロボットから見た方向角度(ラジアン)
        return: 安全ならTrue、障害物があればFalse
        """
        if self.latest_scan is None:
            return True
        
        #parametor of scan data
        angle_min = self.latest_scan.angle_min
        angle_increment = self.latest_scan.angle_increment
        ranges = self.latest_scan.ranges

        # Calculate the corner points of the rectangle
        half_width = self.retangle_width / 2.0
        rect_points = [
            [-half_width, 0],
            [half_width, 0],
            [half_width, self.retangle_height],
            [-half_width, self.retangle_height]
        ]

        # Rotate the rectangle points by the yaw angle
        rect_center = np.array([person_position_x, person_position_z-self.follow_distance])
        rot_matrix = np.array([[np.cos(yaw_angle), -np.sin(yaw_angle)], 
                               [np.sin(yaw_angle), np.cos(yaw_angle)]])
        rotated_points = [np.dot(rot_matrix, p) + rect_center for p in rect_points]

        # Check if any of the points are in the obstacle area
        for p in rotated_points:
            angle = np.arctan2(p[1], p[0])
            index = int((angle - angle_min) / angle_increment)
            if index >= 0 and index < len(ranges):
                if ranges[index] < self.retangle_height:
                    return False
        return True
    
    def find_safe_path(self, person_x, person_z):
        """
        指定された位置周辺の安全な進行方向を探す
        person_x: カメラ座標系でのx座標(m)
        person_z: カメラ座標系でのz座標(m)
        return: 安全な進行方向(ラジアン)
        """
        if self.check_rectangle_safety(person_x, person_z):
            self.current_avoid_angle = 0
            self.is_avoiding = False
            return True
        
        # Try different angles to find a safe path
        for angle in self.safe_paths:
            if self.check_rectangle_safety(person_x, person_z, np.radians(angle)):
                self.current_avoid_angle = np.radians(angle)
                self.is_avoiding = True
                return True
        return False


    def timer_callback(self):
        if self.latest_color is None or self.latest_depth is None:
            self.get_logger().warn('No image or depth data.')
            return

        # Run YOLO detection on the latest color image.
        # stream=Trueの場合はジェネレーターが返されるので、forループで処理する
        results_gen = self.model(self.latest_color, stream=True)
        
        # ジェネレーターから最初の結果を取得
        try:
            result = next(results_gen)
            annotated_image = result.plot()
            
            person_box = None
            # Loop through detected objects to find the first person
            if hasattr(result, 'boxes'):
                for r in result.boxes:
                    cls = int(r.cls[0])
                    if cls == 0:  # class 0 = person in COCO
                        # Get bounding box (x1, y1, x2, y2)
                        box = r.xyxy[0].cpu().numpy()  # format: [x1, y1, x2, y2]
                        person_box = box
                        break
            
            if person_box is None:
                # No person detected; stop robot.
                self.publish_cmd(0.0, 0.0)
                return
            
            # Calculate the center of the bounding box.
            x1, y1, x2, y2 = person_box
            center_u = int((x1 + x2) / 2)
            center_v = int((y1 + y2) / 2)

            # Get depth value from the corresponding pixel in the depth image.
            Z = self.latest_depth[center_v, center_u]/1000.0  # convert to meters
            if np.isnan(Z) or Z <= 0.0:
                self.get_logger().warn('Invalid depth value.')
                self.publish_cmd(0.0, 0.0)
                return

            # Convert image coordinates to camera frame coordinates (assuming pinhole camera model).
            X = (center_u - self.cx) * Z / self.fx   # convert to meters
            self.get_logger().info(f'X: {X}, Z: {Z}')
            # Y = (center_v - self.cy) * Z / self.fy  # not used here (vertical direction)

            # Determine control commands.
            # The error is the lateral displacement (X) plus the distance error along Z.
            if self.is_avoiding and self.avoid_timeout > 0:
                angular_z = self.current_avoid_angle
                linear_x = self.avoid_speed
                self.avoid_timeout -= 1

                if self.avoid_timeout <= 0:
                    self.is_avoiding = False
                    self.get_logger().info('Avoidance completed.')
            else:
                linear_x, angular_z = self.calculate_control_commands(X, Z)
            
            if self.obstacle_detected:
                self.get_logger().info('Obstacle detected! Stopping the robot.')
                slow_down_factor = max(0.0, self.obstacle_distance / self.obstacle_threshold - 0.2)
                slow_down_factor = min(1.0, slow_down_factor)
                linear_x = linear_x * slow_down_factor
                
                if self.obstacle_distance < 0.5:
                    linear_x = 0.0

            # 表示用に矩形を描画
            if not self.is_avoiding:
                rect_color = (0, 255, 0)  # 安全なら緑
            else:
                rect_color = (0, 0, 255)  # 回避中は赤
                
            # 画像上に矩形を描画（簡易版）
            rect_width_px = int(self.rectangle_width * self.fx / Z)
            rect_center_u = center_u
            rect_top_v = center_v + 50  # 適当な位置
            cv2.rectangle(annotated_image, 
                          (rect_center_u - rect_width_px//2, rect_top_v), 
                          (rect_center_u + rect_width_px//2, rect_top_v + 100),
                          rect_color, 2)

            self.publish_cmd(linear_x, angular_z)
            self.get_logger().info(f'Linear: {linear_x}, Angular: {angular_z}')
            self.get_logger().info(f'Person detected at ({center_u}, {center_v}), depth: {Z:.2f}m')
            
            if self.is_avoiding:
                self.get_logger().info(f'回避中: 角度 {np.degrees(self.current_avoid_angle):.1f}度, 残り {self.avoid_timeout/10:.1f}秒')
    

        except StopIteration:
            self.get_logger().warn('No detection results.')
        cv2.imshow('YOLO', annotated_image)
        cv2.waitKey(1)

    def calculate_control_commands(self, X, Z):
        is_safe_path_found = self.find_safe_path(X, Z)
        if is_safe_path_found and not self.is_avoiding:
            angular_z = -self.angular_gain * X
            distance_error = Z - self.follow_distance
            linear_x = self.linear_gain * distance_error
        elif self.is_avoiding:
            angular_z = self.current_avoid_angle
            linear_x = self.avoid_speed
        else:
            # No safe path found and not currently avoiding obstacles
            angular_z = 0
            linear_x = 0
        return linear_x, angular_z

    def publish_cmd(self, linear_x, angular_z):
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = FollowPerson()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()