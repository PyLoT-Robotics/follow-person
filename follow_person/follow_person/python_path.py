import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, CameraInfo
from geometry_msgs.msg import Twist, Pose
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
from trajectory_msgs.msg import PersonTrajectory
from nav2_simple_commander.robot_navigator import BasicNavigator

class PersonTracker(Node):
    def __init__(self):
        super().__init__('person_tracker_node')

        # パブリッシャーとサブスクライバーの設定
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.trajectory_pub = self.create_publisher(PersonTrajectory, 'person_trajectory', 10)
        
        # カメラとLiDARのサブスクライバー
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

        self.nav = BasicNavigator()
        

        # YOLOモデルのロード
        self.model = YOLO("yolov8n-pose.pt")

        # カメラ内部パラメータ
        self.fx = 645.0
        self.fy = 643.0
        self.cx = 320.0
        self.cy = 240.0

        # 制御パラメータ
        self.follow_distance = 1.0  # 人物との目標距離 (m)
        self.linear_gain = 0.8
        self.angular_gain = 1.2
        self.obstacle_threshold = 2.5  # 障害物検知距離しきい値

        # 軌跡記録用パラメータ
        self.trajectory = []
        self.max_trajectory_points = 1000  # 記録する最大点数
        self.min_distance_between_points = 0.1  # 記録する点間の最小距離 (m)
        self.last_recorded_position = None
        self.is_recording = False
        self.record_timer = self.create_timer(0.5, self.publish_trajectory)

        # メイン処理タイマー
        self.create_timer(0.1, self.timer_callback)

    def camera_info_callback(self, msg: CameraInfo):
        if not self.camera_info_received:
            self.fx = msg.k[0]  # fx
            self.fy = msg.k[4]  # fy
            self.cx = msg.k[2]  # cx
            self.cy = msg.k[5]  # cy
            self.camera_info_received = True
            self.get_logger().info(f'カメラパラメータを更新: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}')

    def image_callback(self, msg: Image):
        try:
            self.latest_color = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error('色画像の変換に失敗: ' + str(e))

    def depth_callback(self, msg: Image):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except Exception as e:
            self.get_logger().error('深度画像の変換に失敗: ' + str(e))

    def scan_callback(self, msg: LaserScan):
        # 前方の障害物検知処理
        front_ranges = []
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment
        for i, r in enumerate(msg.ranges):
            angle = angle_min + i * angle_increment
            if abs(angle) > np.radians(155):  # ±10度の範囲
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

    def timer_callback(self):
        if self.latest_color is None or self.latest_depth is None:
            self.get_logger().warn('画像または深度データがありません')
            return

        # YOLOによる人物検出
        results_gen = self.model(self.latest_color, stream=True)
        
        try:
            result = next(results_gen)
            annotated_image = result.plot()
            
            person_box = None
            # 検出結果から最初の人物を見つける
            if hasattr(result, 'boxes'):
                for r in result.boxes:
                    cls = int(r.cls[0])
                    if cls == 0:  # COCO データセットでは class 0 = 人
                        box = r.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2] 形式
                        person_box = box
                        break
            
            if person_box is None:
                # 人物が検出されなければ停止
                self.publish_cmd(0.0, 0.0)
                return
            
            # バウンディングボックスの中心を計算
            x1, y1, x2, y2 = person_box
            center_u = int((x1 + x2) / 2)
            center_v = int((y1 + y2) / 2)

            # 深度値を取得
            Z = self.latest_depth[center_v, center_u] / 1000.0  # mに変換
            if np.isnan(Z) or Z <= 0.0:
                self.get_logger().warn('無効な深度値')
                self.publish_cmd(0.0, 0.0)
                return

            # 画像座標からカメラ座標系に変換
            X = (center_u - self.cx) * Z / self.fx  # m単位
            Y = (center_v - self.cy) * Z / self.fy  # m単位
            
            self.get_logger().info(f'人物検出: X={X:.2f}m, Y={Y:.2f}m, Z={Z:.2f}m')
            
            # 軌跡に点を追加
            self.update_trajectory(X, Y, Z)
            
            # 制御コマンドを生成
            angular_z = -self.angular_gain * X
            distance_error = Z - self.follow_distance
            linear_x = self.linear_gain * distance_error

            # 障害物があれば速度を調整
            if self.obstacle_detected:
                self.get_logger().info(f'障害物検出: 距離={self.obstacle_distance:.2f}m')
                slow_down_factor = max(0.0, self.obstacle_distance / self.obstacle_threshold - 0.2)
                slow_down_factor = min(1.0, slow_down_factor)
                linear_x = linear_x * slow_down_factor
                
                if self.obstacle_distance < 0.5:
                    linear_x = 0.0

            # 速度指令を発行
            self.publish_cmd(linear_x, angular_z)
            
            # 画像に軌跡を表示
            self.draw_trajectory(annotated_image)
            
            cv2.imshow('Person Tracking', annotated_image)
            cv2.waitKey(1)
            
        except StopIteration:
            self.get_logger().warn('検出結果なし')
    
    def update_trajectory(self, x, y, z):
        """人物の位置を軌跡に追加"""
        current_position = (x, y, z)
        
        # 軌跡記録中でなければ開始
        if not self.is_recording:
            self.is_recording = True
            self.trajectory = [current_position]
            self.last_recorded_position = current_position
            self.get_logger().info('軌跡の記録を開始')
            return
        
        # 前回記録した位置との距離をチェック
        if self.last_recorded_position:
            dx = current_position[0] - self.last_recorded_position[0]
            dy = current_position[1] - self.last_recorded_position[1]
            dz = current_position[2] - self.last_recorded_position[2]
            distance = np.sqrt(dx*dx + dy*dy + dz*dz)
            
            # 最小距離以上離れていれば記録
            if distance >= self.min_distance_between_points:
                self.trajectory.append(current_position)
                self.last_recorded_position = current_position
                
                # 最大点数を超えたら古い点を削除
                if len(self.trajectory) > self.max_trajectory_points:
                    self.trajectory.pop(0)
                
                self.get_logger().debug(f'軌跡に点を追加: ({x:.2f}, {y:.2f}, {z:.2f}), 合計点数: {len(self.trajectory)}')

    def publish_trajectory(self):
        """記録した軌跡をパブリッシュ"""
        if not self.trajectory:
            return
            
        trajectory_msg = PersonTrajectory()
        trajectory_msg.header.stamp = self.get_clock().now().to_msg()
        trajectory_msg.header.frame_id = 'camera_link'
        
        for pos in self.trajectory:
            pose = Pose()
            pose.position.x = pos[0]
            pose.position.y = pos[1]
            pose.position.z = pos[2]
            trajectory_msg.poses.append(pose)
            
        self.trajectory_pub.publish(trajectory_msg)
        self.get_logger().info(f'軌跡をパブリッシュ: {len(self.trajectory)}点')

    def draw_trajectory(self, image):
        """画像上に軌跡を描画"""
        if not self.trajectory or len(self.trajectory) < 2:
            return
            
        points = []
        for pos in self.trajectory:
            # 3D座標から画像座標に逆変換
            u = int(self.cx + pos[0] * self.fx / max(0.1, pos[2]))
            v = int(self.cy + pos[1] * self.fy / max(0.1, pos[2]))
            points.append((u, v))
            
        # 軌跡を線で結ぶ
        for i in range(1, len(points)):
            cv2.line(image, points[i-1], points[i], (0, 255, 255), 2)

    def publish_cmd(self, linear_x, angular_z):
        """速度指令をパブリッシュ"""
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = PersonTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()