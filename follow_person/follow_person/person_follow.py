import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, CameraInfo
from geometry_msgs.msg import Twist
from realsense2_camera_msgs.msg import RGBD
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
from pylot_msgs.srv import PersonFollow
import time

#!/usr/bin/env python3

class FollowPerson(Node):
    def __init__(self):
        super().__init__('follow_person_node')

        # Create publishers/subscribers.
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        # デバッグ画像用のパブリッシャーを追加
        self.debug_image_pub = self.create_publisher(Image, '/follow_person/debug_image', 10)
        # Create subscribers for camera and lidar data.
        self.create_subscription(Image, '/d455/camera/color/image_raw', self.image_callback, 10)
        self.create_subscription(Image, '/d455/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.create_subscription(CameraInfo, '/d455/camera/color/camera_info', self.camera_info_callback, 10)
        self.srv = self.create_service(PersonFollow, 'person_follow', self.person_follow_callback)

        self.bridge = CvBridge()
        self.latest_color = None
        self.latest_depth = None
        self.obstacle_detected = False
        self.camera_info_received = False
        self.obstacle_distance = float('inf')

        # Load YOLO model
        self.model = YOLO("yolov8n-pose.pt")

        # Camera intrinsic parameters
        self.fx = 645.0
        self.fy = 643.0
        self.cx = 320.0
        self.cy = 240.0

        # Control parameters
        self.follow_distance = 1.0  # desired distance in meters to the person
        self.linear_gain = 0.8
        self.angular_gain = 1.2
        self.obstacle_threshold = 2.5  # stop if obstacle within 0.5m

        # 追跡関連のパラメータ
        self.target_person_id = None
        self.tracking_lost_counter = 0
        self.max_tracking_lost = 30  # 追跡を失ったと判断するフレーム数
        self.tracking_interval = 100

        # タイマーの作成
        self.create_timer(0.1, self.timer_callback)

    def person_follow_callback(self, request, response):
        # 追従を開始
        self.get_logger().info("追従を開始します")
        response.success = True
        return response

    def camera_info_callback(self, msg: CameraInfo):
        if not self.camera_info_received:
            self.fx = msg.k[0]  # fx
            self.fy = msg.k[4]  # fy
            self.camera_info_received = True
            self.get_logger().info(f'カメラパラメータを更新しました: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}')

    def image_callback(self, msg: Image):
        try:
            self.latest_color = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error('Failed to convert color image: ' + str(e))

    def depth_callback(self, msg: Image):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except Exception as e:
            self.get_logger().error('Failed to convert depth image: ' + str(e))

    def timer_callback(self):
        if self.latest_color is None or self.latest_depth is None:
            self.get_logger().warn('No image or depth data.')
            return

        # Run YOLO detection with tracking
        results_gen = self.model.track(self.latest_color, persist=True, classes=[0], stream=True)
        
        try:
            result = next(results_gen)
            annotated_image = result.plot()
            
            # 人物を検出できたか確認
            person_detected = False
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
                        
                        # ROI内の深度値を取得
                        try:
                            x1, y1, x2, y2 = map(int, xyxy)
                            roi_depth = self.latest_depth[y1:y2, x1:x2]
                            
                            # 有効な深度値のみを抽出
                            valid_depths = roi_depth[~np.isnan(roi_depth) & (roi_depth > 0)]
                            
                            if len(valid_depths) > 0:
                                # 有効な深度値を昇順にソート
                                sorted_depths = np.sort(valid_depths)
                                
                                # 最も近い値から30%の位置の深度を選択
                                percentile_idx = int(len(sorted_depths) * 0.3)
                                depth_value = sorted_depths[percentile_idx]
                                
                                # メートルに変換
                                depth = depth_value / 1000.0
                                
                                # 極端な値をフィルタリング
                                if depth <= 0.1 or depth > 10.0:
                                    self.get_logger().warn(f'Extreme depth value: {depth}m at percentile 30%')
                                    continue
                            else:
                                # 有効な深度値がない場合は中心点を使用
                                depth = self.latest_depth[center_v, center_u] / 1000.0
                                if np.isnan(depth) or depth <= 0.0:
                                    self.get_logger().warn(f'Invalid depth value at center: {depth}')
                                    continue
                                    
                            # 深度デバッグ情報
                            self.get_logger().debug(f'Person ID {person_id}: depth={depth:.2f}m (from ROI)')
                            
                        except IndexError as ie:
                            self.get_logger().warn(f'Depth IndexError: {ie}')
                            continue
                        except Exception as e:
                            self.get_logger().warn(f'Depth error: {e}')
                            continue
                        
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
                # デバッグ画像をパブリッシュ
                try:
                    debug_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
                    self.debug_image_pub.publish(debug_msg)
                except Exception as e:
                    self.get_logger().error(f'Failed to publish debug image: {str(e)}')
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
                X = (center_u - self.cx)  / self.fx
                
                # 制御コマンドの計算
                angular_z = -self.angular_gain * X
                distance_error = Z - self.follow_distance
                linear_x = self.linear_gain * distance_error

                self.publish_cmd(linear_x, angular_z)
                self.get_logger().info(f'Linear: {linear_x}, Angular: {angular_z}')
                self.get_logger().info(f'Person detected at ({center_u}, {center_v}), depth: {Z:.2f}m')

                # デバッグ画像をパブリッシュ
                try:
                    debug_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
                    self.debug_image_pub.publish(debug_msg)
                except Exception as e:
                    self.get_logger().error(f'Failed to publish debug image: {str(e)}')

        except StopIteration:
            self.get_logger().warn('No detection results.')
            self.tracking_lost_counter += 1
            if self.tracking_lost_counter >= self.max_tracking_lost:
                self.target_person_id = None
            self.publish_cmd(0.0, 0.0)
            # デバッグ画像をパブリッシュ
            try:
                debug_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
                self.debug_image_pub.publish(debug_msg)
            except Exception as e:
                self.get_logger().error(f'Failed to publish debug image: {str(e)}')
        except Exception as e:
            self.get_logger().error(f'Error in timer callback: {str(e)}')
            self.publish_cmd(0.0, 0.0)

    def publish_cmd(self, linear_x, angular_z):
        twist = Twist()
        max_speed = 0.6
        max_angular_speed = 1.2
        if abs(linear_x) >= max_speed:
            if linear_x > 0:
                linear_x = max_speed
            else:
                linear_x = -max_speed
        if abs(angular_z) >= max_angular_speed:
            if angular_z > 0:
                angular_z = max_angular_speed
            else:
                angular_z = -max_angular_speed

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
