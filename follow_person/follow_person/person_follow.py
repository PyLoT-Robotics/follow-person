import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, CameraInfo
from geometry_msgs.msg import Twist
from realsense2_camera_msgs.msg import RGBD
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO


#multithread
import threading
import queue
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

        # Initialize the track variables
        self.target_person_id = None
        self.tracking_lost_counter = 0
        self.max_tracking_lost = 10  # Number of frames to consider tracking lost
        self.tracking_interval = 5

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

        self.create_timer(0.1, self.timer_callback)

        self.debug_image_pub = self.create_publisher(Image, '~/debug_image', 10)

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

    # タイマーコールバック
    # 画像と深度データを使用して人物を追跡

    def timer_callback(self):
        if self.latest_color is None or self.latest_depth is None:
            self.get_logger().warn('No image or depth data.')
            return

        # Run YOLO detection with tracking enabled
        results_gen = self.model.track(self.latest_color, persist=True, classes=[0], stream=True)
        
        try:
            result = next(results_gen)
            annotated_image = result.plot()

            if 'annotated_image' in locals():
                try:
                    debug_image_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
                    debug_image_msg.header.stamp = self.get_clock().now().to_msg()
                    self.debug_image_pub.publish(debug_image_msg)
                except Exception as e:
                    self.get_logger().error('Failed to publish debug image: ' + str(e))
            
            # 人物を検出できたか確認
            person_detected = False
            
            # 検出した人物の情報を格納するリスト
            detected_persons = []
            
            # 追跡IDが存在するか確認
            has_tracking_ids = hasattr(result.boxes, 'id') and result.boxes.id is not None
            
            if hasattr(result, 'boxes') and len(result.boxes) > 0:
                for i, box in enumerate(result.boxes):
                    cls = int(box.cls[0])
                    if cls == 0:  # class 0 = person in COCO
                        person_detected = True
                        
                        # バウンディングボックスを取得
                        xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                        
                        # 追跡IDを取得（存在する場合）
                        person_id = int(box.id[0]) if has_tracking_ids else i
                        
                        # バウンディングボックスの中心座標を計算
                        center_u = int((xyxy[0] + xyxy[2]) / 2)
                        center_v = int((xyxy[1] + xyxy[3]) / 2)
                        
                        # 対応する深度値を取得
                        try:
                            depth = self.latest_depth[center_v, center_u]/1000.0  # メートルに変換
                            if np.isnan(depth) or depth <= 0.0:
                                continue  # 無効な深度値はスキップ
                        except IndexError:
                            continue  # 画像の範囲外ならスキップ
                        
                        # 人物情報をリストに追加
                        person_info = {
                            'id': person_id,
                            'box': xyxy,
                            'center': (center_u, center_v),
                            'depth': depth
                        }
                        detected_persons.append(person_info)
            
            # 追跡対象が見つからなかった場合のカウンター処理
            if not person_detected or len(detected_persons) == 0:
                self.tracking_lost_counter += 1
                if self.tracking_lost_counter >= self.max_tracking_lost:
                    # 長時間見つからなければターゲットIDをリセット
                    self.target_person_id = None
                    self.get_logger().info('Target lost. Resetting tracking.')
                
                # 人物が見つからない場合はロボットを停止
                self.publish_cmd(0.0, 0.0)
                return
            
            # 検出した人物をソート（距離が近い順）
            detected_persons.sort(key=lambda p: p['depth'])
            
            # ターゲット選択ロジック
            target_person = None
            
            if self.target_person_id is not None:
                # 既存のターゲットIDを持つ人物を探す
                for person in detected_persons:
                    if person['id'] == self.target_person_id:
                        target_person = person
                        self.tracking_lost_counter = 0  # カウンターリセット
                        break
            
            # ターゲットが見つからないか未設定の場合、最も近い人物を選択
            if target_person is None:
                if len(detected_persons) > 0:
                    target_person = detected_persons[0]  # 最も近い人物
                    self.target_person_id = target_person['id']
                    self.tracking_lost_counter = 0
                    self.get_logger().info(f'New target selected: ID {self.target_person_id}')
            
            # ターゲットに対する制御コマンドを計算
            if target_person:
                center_u, center_v = target_person['center']
                Z = target_person['depth']  # メートル単位
                
                # バウンディングボックスに矩形を描画（追跡対象は赤色）
                x1, y1, x2, y2 = map(int, target_person['box'])
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated_image, f"ID: {target_person['id']} ({Z:.2f}m)", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # カメラ座標系への変換
                X = (center_u - self.cx) * Z / self.fx
                
                # 制御コマンドの計算
                angular_z = -self.angular_gain * X
                distance_error = Z - self.follow_distance
                linear_x = self.linear_gain * distance_error
                
                # 障害物検出時の減速処理
                if self.obstacle_detected:
                    slow_down_factor = max(0.0, self.obstacle_distance / self.obstacle_threshold - 0.2)
                    slow_down_factor = min(1.0, slow_down_factor)
                    linear_x = linear_x * slow_down_factor
                    
                    if self.obstacle_distance < 0.5:
                        linear_x = 0.0
                
                # 制御コマンドの発行
                self.publish_cmd(linear_x, angular_z)
                
                # ログ出力
                self.get_logger().info(f'Following ID: {self.target_person_id}, Depth: {Z:.2f}m')
                self.get_logger().info(f'Linear: {linear_x:.2f}, Angular: {angular_z:.2f}')
                
            else:
                self.get_logger().warn('No valid target person found.')
                self.publish_cmd(0.0, 0.0)
            # 画像表示（デバッグ用）
            # cv2.imshow('Person Tracking', annotated_image)
            # key = cv2.waitKey(1)
            # if key==27:
            #     self.get_logger().info('ESC key pressed, exiting...')
            #     cv2.destroyAllWindows()
            #     rclpy.shutdown( )
            
        except StopIteration:
            self.get_logger().warn('No detection results.')
            self.tracking_lost_counter += 1
            if self.tracking_lost_counter >= self.max_tracking_lost:
                self.target_person_id = None
            self.publish_cmd(0.0, 0.0)
        except Exception as e:
            self.get_logger().error(f'Error in timer callback: {str(e)}')
            self.publish_cmd(0.0, 0.0)

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