import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, CameraInfo
from geometry_msgs.msg import Twist, TransformStamped, Pose
from nav_msgs.msg import Odometry, OccupancyGrid
from tf2_ros import TransformListener, TransformBroadcaster, Buffer
import tf2_geometry_msgs
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import math

class FollowPersonWithObstacleAvoidance(Node):
    def __init__(self):
        super().__init__('follow_person_obstacle_avoid_node')
        
        # パブリッシャー
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.grid_pub = self.create_publisher(OccupancyGrid, '/laser_map', 10)
        
        # サブスクライバーの設定
        self.create_subscription(Image, '/d455/camera/color/image_raw', self.image_callback, 10)
        self.create_subscription(Image, '/d455/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.create_subscription(LaserScan, '/scan_filtered', self.scan_callback, 10)
        self.create_subscription(CameraInfo, '/d455/camera/color/camera_info', self.camera_info_callback, 10)
        # Odometryサブスクライバーを追加
        self.create_subscription(Odometry, '/odometry/filtered', self.odom_callback, 10)
        
        # 各種データの初期化
        self.bridge = CvBridge()
        self.latest_color = None
        self.latest_depth = None
        self.obstacle_detected = False
        self.camera_info_received = False
        
        # Odometryデータの初期化
        self.current_pose = Pose()
        self.last_map_update_pose = Pose()
        self.odometry_initialized = False
        
        # マップ更新の閾値
        self.map_update_threshold_distance = 0.2  # メートル
        self.map_update_threshold_angle = 0.1  # ラジアン（約5.7度）
        
        # 蓄積マップのデータ
        self.accumulated_map = None
        
        # YOLOモデルの読み込み
        self.model = YOLO("yolov8n-pose.pt")
        
        # カメラパラメータの初期値
        self.fx = 645.0
        self.fy = 643.0
        self.cx = 320.0
        self.cy = 240.0
        
        # OccupancyGridの設定
        self.grid_width = 400
        self.grid_height = 400
        self.grid_resolution = 0.05  # 5cm per cell
        self.latest_grid = None
        
        # TFの設定
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # LiDARの位置（base_linkからの相対位置）
        self.lidar_x = 0.0  # LiDARのx座標オフセット
        self.lidar_y = 0.0  # LiDARのy座標オフセット
        
        # 制御パラメータ
        self.follow_distance = 1.0  # 人物との目標距離（メートル）
        self.linear_gain = 0.8
        self.angular_gain = 1.0
        self.obstacle_threshold = 1.5  # 障害物検知距離閾値（メートル）
        
        # 障害物回避パラメータ
        self.obstacle_avoid_enabled = True
        self.avoid_gain = 0.5
        self.path_clear = True  # 経路が障害物なしかどうか
        self.avoid_direction = 0.0  # 回避方向 (-1:左, 1:右)
        
        # タイマーの設定
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

    def odom_callback(self, msg: Odometry):
        # 現在位置を更新
        self.current_pose = msg.pose.pose
        
        if not self.odometry_initialized:
            # 最初のOdometryメッセージを受信したら初期位置を記録
            self.last_map_update_pose = self.current_pose
            self.odometry_initialized = True
            self.get_logger().info("Odometry initialized")
    
    def scan_callback(self, msg: LaserScan):
        # 前方の障害物検出（従来の処理）
        front_ranges = []
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment
        for i, r in enumerate(msg.ranges):
            angle = angle_min + i * angle_increment
            if abs(angle) < np.radians(10):  # 前方10度以内
                if r > msg.range_min and r < msg.range_max:
                    front_ranges.append(r)
        
        if front_ranges and min(front_ranges) < self.obstacle_threshold:
            self.obstacle_detected = True
        else:
            self.obstacle_detected = False
        
        # LaserScanデータをOccupancyGridに変換
        msg_converted = msg
        self.create_occupancy_grid(msg)

    def create_occupancy_grid(self, scan_msg: LaserScan):
        # OccupancyGridメッセージ作成
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = "odom"  # odometryフレームを基準に変更
        
        # グリッドの基本情報設定
        grid_msg.info.resolution = self.grid_resolution
        grid_msg.info.width = self.grid_width
        grid_msg.info.height = self.grid_height
        
        # 原点設定：グリッドの中心がロボットの位置になるように
        origin_x = -self.grid_width * self.grid_resolution / 2.0
        origin_y = -self.grid_height * self.grid_resolution / 2.0
        
        # Odometryを考慮したグリッド原点の設定
        grid_msg.info.origin.position.x = origin_x + self.current_pose.position.x
        grid_msg.info.origin.position.y = origin_y + self.current_pose.position.y
        grid_msg.info.origin.position.z = 0.0
        grid_msg.info.origin.orientation = self.current_pose.orientation
        
        # グリッドデータ初期化（-1 = 未知）
        grid_data = np.full(self.grid_width * self.grid_height, -1, dtype=np.int8)
        
        # グリッドの中心座標
        center_x = self.grid_width // 2
        center_y = self.grid_height // 2
        
        # LiDARの位置オフセットを考慮
        lidar_offset_cells_x = int(self.lidar_x / self.grid_resolution)
        lidar_offset_cells_y = int(self.lidar_y / self.grid_resolution)
        
        # 現在のロボットの向き（クォータニオンからオイラー角に変換）
        _, _, yaw = self.quaternion_to_euler(
            self.current_pose.orientation.x,
            self.current_pose.orientation.y,
            self.current_pose.orientation.z,
            self.current_pose.orientation.w
        )
        
        # LaserScanデータをグリッドに変換
        angle = scan_msg.angle_min
        for i, r in enumerate(scan_msg.ranges):
            # 有効な測定値のみ処理
            if not math.isnan(r) and scan_msg.range_min <= r <= scan_msg.range_max:
                # LiDARの反転を考慮した座標変換
                scan_x = r * math.cos(angle)
                scan_y = r * math.sin(angle)
                
                # ロボット座標系での位置に変換（LiDARが180度回転している場合）
                robot_x = -scan_x  # ロボット座標系ではx軸反転
                robot_y = -scan_y  # ロボット座標系ではy軸反転
                
                # ロボットの向きを考慮して点を回転
                rotated_x = robot_x * math.cos(yaw) - robot_y * math.sin(yaw)
                rotated_y = robot_x * math.sin(yaw) + robot_y * math.cos(yaw)
                
                # グリッドの座標に変換
                grid_x = int(center_x + rotated_x / self.grid_resolution)
                grid_y = int(center_y + rotated_y / self.grid_resolution)
                
                # グリッド内かチェック
                if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                    # インデックス計算
                    idx = grid_y * self.grid_width + grid_x
                    
                    # 障害物を100で表示
                    grid_data[idx] = 100
                    
                    # 膨張処理：障害物の周囲にバッファを設ける
                    for dy in range(-3, 4):
                        for dx in range(-3, 4):
                            nx = grid_x + dx
                            ny = grid_y + dy
                            if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                                nidx = ny * self.grid_width + nx
                                # 膨張の強さを距離に応じて変える
                                dist = abs(dx) + abs(dy)  # マンハッタン距離
                                if grid_data[nidx] != 100:  # まだ障害物でない場合
                                    if dist <= 1:
                                        grid_data[nidx] = 90  # 近い周辺は強い障害物
                                    elif dist <= 2:
                                        grid_data[nidx] = 85  # 少し離れた周辺
                                    else:
                                        grid_data[nidx] = 80  # 遠い周辺
            
            angle += scan_msg.angle_increment
        
        # ロボット自身の位置（グリッドの中心）を空き(0)としてマーク
        robot_radius_cells = max(1, int(0.2 / self.grid_resolution))
        for y in range(center_y - robot_radius_cells, center_y + robot_radius_cells + 1):
            for x in range(center_x - robot_radius_cells, center_x + robot_radius_cells + 1):
                if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                    idx = y * self.grid_width + x
                    grid_data[idx] = 0
        
        # マップの更新が必要か確認
        if self.should_update_map():
            # 蓄積マップを更新
            self.update_accumulated_map(grid_data)
            # 更新位置を記録
            self.last_map_update_pose = self.current_pose
        
        # 最終的なグリッドデータを設定
        if self.accumulated_map is not None:
            # 蓄積マップと現在のスキャンを組み合わせる
            combined_grid = self.accumulated_map.copy()
            # 現在のロボット位置周辺は最新データで上書き
            # (簡易実装のため、全体を更新)
            for i in range(self.grid_width * self.grid_height):
                if grid_data[i] == 100:  # 障害物があれば上書き
                    combined_grid[i] = grid_data[i]
                elif grid_data[i] == 0:  # 空き領域も上書き
                    combined_grid[i] = grid_data[i]
            grid_data = combined_grid
        
        # メッセージにグリッドデータをセット
        grid_msg.data = grid_data.tolist()
        
        # 可視化用にグリッドデータを保存
        self.latest_grid = grid_data.reshape((self.grid_height, self.grid_width))
        
        # グリッドをパブリッシュ
        self.grid_pub.publish(grid_msg)
        
        # OpenCVで可視化
        self.visualize_grid()
    
    def should_update_map(self):
        """マップを更新すべきかどうかを判断する"""
        if not self.odometry_initialized:
            return False
        
        # 前回の更新からの移動距離を計算
        dx = self.current_pose.position.x - self.last_map_update_pose.position.x
        dy = self.current_pose.position.y - self.last_map_update_pose.position.y
        distance_moved = math.sqrt(dx*dx + dy*dy)
        
        # 前回の更新からの回転角度を計算
        current_yaw = self.get_yaw_from_quaternion(self.current_pose.orientation)
        last_yaw = self.get_yaw_from_quaternion(self.last_map_update_pose.orientation)
        angle_diff = abs(current_yaw - last_yaw)
        # 角度差が180度を超える場合の対応
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff
        
        # 距離または角度が閾値を超えたら更新
        return (distance_moved > self.map_update_threshold_distance or 
                angle_diff > self.map_update_threshold_angle)
    
    def update_accumulated_map(self, current_grid):
        """蓄積マップを更新する"""
        if self.accumulated_map is None:
            # 初めての更新
            self.accumulated_map = current_grid.copy()
        else:
            # マップの統合（単純な上書き - より洗練された方法も可能）
            for i in range(len(current_grid)):
                if current_grid[i] == 100:  # 障害物があれば上書き
                    self.accumulated_map[i] = current_grid[i]
                elif current_grid[i] == 0 and self.accumulated_map[i] == -1:  # 空き領域も更新
                    self.accumulated_map[i] = 0

    def quaternion_to_euler(self, x, y, z, w):
        """四元数からオイラー角（ロール、ピッチ、ヨー）に変換"""
        # ロール（x軸周りの回転）
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)
        
        # ピッチ（y軸周りの回転）
        t2 = 2.0 * (w * y - z * x)
        t2 = 1.0 if t2 > 1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.asin(t2)
        
        # ヨー（z軸周りの回転）
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)
        
        return roll, pitch, yaw

    def get_yaw_from_quaternion(self, quaternion):
        """四元数からヨー角のみ取得"""
        _, _, yaw = self.quaternion_to_euler(
            quaternion.x, quaternion.y, quaternion.z, quaternion.w
        )
        return yaw


    def visualize_grid(self):
        if self.latest_grid is None:
            return
        
        # 可視化用の画像を作成
        grid_viz = np.zeros((self.grid_height, self.grid_width, 3), dtype=np.uint8)
        
        # 値に基づいて色付け
        grid_viz[self.latest_grid == -1] = [128, 128, 128]  # 未知: 灰色
        grid_viz[self.latest_grid == 0] = [255, 255, 255]   # 空き: 白
        grid_viz[self.latest_grid == 80] = [0, 140, 255]    # 遠い障害物周辺: 薄いオレンジ
        grid_viz[self.latest_grid == 85] = [0, 165, 255]    # やや近い障害物周辺: オレンジ
        grid_viz[self.latest_grid == 90] = [0, 69, 255]     # 近い障害物周辺: 濃いオレンジ
        grid_viz[self.latest_grid == 100] = [0, 0, 0]       # 障害物: 黒
        
        # グリッドの中心座標（現在のロボット位置）
        center_x = self.grid_width // 2
        center_y = self.grid_height // 2
        
        # ロボットの位置を青い円で表示
        cv2.circle(grid_viz, (center_x, center_y), 5, (255, 0, 0), -1)
        
        # 現在の向きを示す矢印
        arrow_length = 20
        if self.odometry_initialized:
            yaw = self.get_yaw_from_quaternion(self.current_pose.orientation)
            end_x = int(center_x + arrow_length * math.cos(yaw))
            end_y = int(center_y + arrow_length * math.sin(yaw))
            cv2.arrowedLine(grid_viz, (center_x, center_y), (end_x, end_y), (0, 255, 0), 2)
        
        # 経路計画の可視化（もし人物が検出されている場合）
        if hasattr(self, 'target_x') and hasattr(self, 'target_y'):
            # Odometryを考慮した目標位置の計算
            target_grid_x = int(center_x + self.target_x / self.grid_resolution)
            target_grid_y = int(center_y + self.target_y / self.grid_resolution)
            
            if 0 <= target_grid_x < self.grid_width and 0 <= target_grid_y < self.grid_height:
                cv2.circle(grid_viz, (target_grid_x, target_grid_y), 8, (0, 0, 255), -1)
                cv2.line(grid_viz, (center_x, center_y), (target_grid_x, target_grid_y), (0, 255, 0), 2)
        
        # 障害物検出状態と経路情報を表示
        status = "ob" if self.obstacle_detected else "clear"
        color = (0, 0, 255) if self.obstacle_detected else (0, 255, 0)
        cv2.putText(grid_viz, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        path_status = "avoid" if not self.path_clear else "clear road"
        path_color = (0, 0, 255) if not self.path_clear else (0, 255, 0)
        cv2.putText(grid_viz, path_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, path_color, 2)
        
        # Odometry情報を表示
        if self.odometry_initialized:
            pos_text = f"x:{self.current_pose.position.x:.1f} y:{self.current_pose.position.y:.1f}"
            cv2.putText(grid_viz, pos_text, (10, self.grid_height - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 障害物回避方向を表示（回避中の場合）
        if not self.path_clear:
            direction = "left" if self.avoid_direction < 0 else "right"
            cv2.putText(grid_viz, direction, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        # 画像を表示
        cv2.imshow('Occupancy Grid', grid_viz)
        cv2.waitKey(1)

    def timer_callback(self):
        if self.latest_color is None or self.latest_depth is None:
            self.get_logger().warn('No image or depth data.')
            return
    
        # YOLO検出の実行
        results_gen = self.model(self.latest_color, stream=True)
        
        try:
            result = next(results_gen)
            annotated_image = result.plot()
            
            person_box = None
            keypoints = None
            use_keypoints = False
            
            # 人物のバウンディングボックス検出
            if hasattr(result, 'boxes'):
                for r in result.boxes:
                    cls = int(r.cls[0])
                    if cls == 0:  # COCO dataset: class 0 = person
                        box = r.xyxy[0].cpu().numpy()
                        person_box = box
                        break
            
            # キーポイント検出
            if hasattr(result, 'keypoints') and len(result.keypoints) > 0:
                keypoints = result.keypoints.xy[0].cpu().numpy()
                if len(keypoints) >= 7 and np.all(keypoints[5] > 0) and np.all(keypoints[6] > 0):
                    use_keypoints = True
            
            # 人物が検出されなかった場合
            if person_box is None:
                self.get_logger().warn('No person detected.')
                self.publish_cmd(0.0, 0.0)
                return
            
            # 追跡に使用する座標の決定
            center_u, center_v = 0, 0
            
            if use_keypoints:
                # 肩のキーポイント使用
                left_shoulder = keypoints[5]
                right_shoulder = keypoints[6]
                center_u = int((left_shoulder[0] + right_shoulder[0]) / 2)
                center_v = int((left_shoulder[1] + right_shoulder[1]) / 2)
                cv2.circle(annotated_image, (center_u, center_v), 8, (0, 255, 0), -1)
            else:
                # バウンディングボックス中心使用
                x1, y1, x2, y2 = person_box
                center_u = int((x1 + x2) / 2)
                center_v = int((y1 + y2) / 2)
                cv2.circle(annotated_image, (center_u, center_v), 8, (0, 0, 255), -1)
            
            # 深度値の取得
            Z = self.latest_depth[center_v, center_u] / 1000.0  # メートル単位に変換
            if np.isnan(Z) or Z <= 0.0:
                self.get_logger().warn('Invalid depth value.')
                self.publish_cmd(0.0, 0.0)
                return
            
            # 人物の3D位置計算
            X = (center_u - self.cx) * Z / self.fx
            
            # ターゲット位置を保存（可視化用）
            self.target_x = X
            self.target_y = Z
            
            # 経路上の障害物チェック
            self.check_path_obstacles(X, Z)
            
            # 制御コマンド計算
            angular_z = -self.angular_gain * X
            distance_error = Z - self.follow_distance
            linear_x = self.linear_gain * distance_error
            
            # 障害物回避ロジック
            if self.obstacle_avoid_enabled:
                if self.obstacle_detected:
                    # 前方に障害物がある場合は前進停止
                    self.get_logger().info("Obstacle detected in front. Stopping forward motion.")
                    linear_x = 0.0
                
                if not self.path_clear:
                    # 経路上に障害物がある場合、回避方向に旋回
                    avoid_angular = self.avoid_gain * self.avoid_direction
                    self.get_logger().info(f"Obstacle in path. Avoiding direction: {self.avoid_direction}")
                    angular_z += avoid_angular
            
            # 速度コマンド発行
            self.publish_cmd(linear_x, angular_z)
            
            # ログ出力
            self.get_logger().info(f'Linear: {linear_x}, Angular: {angular_z}')
            self.get_logger().info(f'Person at X: {X:.2f}m, Z: {Z:.2f}m')
            
            # 画像表示
            cv2.imshow('Detection', annotated_image)
            cv2.waitKey(1)
            
        except StopIteration:
            self.get_logger().warn('No detection results.')
            self.publish_cmd(0.0, 0.0)
        except Exception as e:
            self.get_logger().error(f'Error in timer_callback: {str(e)}')
            self.publish_cmd(0.0, 0.0)
    
    def check_path_obstacles(self, target_x, target_z):
        """
        人物までの経路上に障害物があるかチェックし、回避方向を決定する
        """
        if self.latest_grid is None:
            self.path_clear = True
            return
        
        # グリッド中心（ロボット位置）
        center_x = self.grid_width // 2
        center_y = self.grid_height // 2
        
        # ターゲット（人物）のグリッド上の位置
        # Odometryの向きを考慮して回転
        if self.odometry_initialized:
            yaw = self.get_yaw_from_quaternion(self.current_pose.orientation)
            
            # ロボット座標系での目標位置
            rotated_x = target_x * math.cos(-yaw) - target_z * math.sin(-yaw)
            rotated_z = target_x * math.sin(-yaw) + target_z * math.cos(-yaw)
            
            target_grid_x = int(center_x + rotated_x / self.grid_resolution)
            target_grid_y = int(center_y + rotated_z / self.grid_resolution)
        else:
            # Odometryが初期化されていない場合は単純変換
            target_grid_x = int(center_x + target_x / self.grid_resolution)
            target_grid_y = int(center_y + target_z / self.grid_resolution)
        
        # グリッド境界チェック
        if (target_grid_x < 0 or target_grid_x >= self.grid_width or
            target_grid_y < 0 or target_grid_y >= self.grid_height):
            self.path_clear = True
            return
        
        # 以降の処理は既存のコードを活用
        # ...
        
        # ロボットの幅（グリッドセル単位）
        # ロボットの実際の幅 + 安全マージン
        robot_width_cells = max(2, int(0.6 / self.grid_resolution))  # ロボット幅 + 安全マージン
        
        # 線分抽出アルゴリズムで経路をチェック
        obstacle_found = False
        left_obstacles = 0
        right_obstacles = 0
        
        # 基本経路を取得（ロボットからターゲットへの直線）
        base_path = self.bresenham_line(center_x, center_y, target_grid_x, target_grid_y)
        
        # 経路の方向ベクトルを計算
        path_len = len(base_path)
        if path_len < 2:
            self.path_clear = True
            return
        
        # 経路をチェック（基本経路 + 両側の拡張経路）
        all_path_points = set()
        
        for i, (x, y) in enumerate(base_path):
            # 経路の進行方向に垂直な方向を計算
            # （最初のポイントはロボット位置なのでスキップ）
            if i == 0 or i == path_len - 1:
                # ロボット位置とゴール位置はそのまま追加
                all_path_points.add((x, y))
                continue
                
            # 前後のポイントから方向を計算
            prev_x, prev_y = base_path[i-1]
            next_x, next_y = base_path[i+1] if i < path_len - 1 else base_path[i]
            
            # 進行方向ベクトル
            dx = next_x - prev_x
            dy = next_y - prev_y
            
            # 垂直方向ベクトル（90度回転）
            perp_dx = -dy
            perp_dy = dx
            
            # ベクトルの正規化
            length = math.sqrt(perp_dx**2 + perp_dy**2)
            if length > 0:
                perp_dx /= length
                perp_dy /= length
            
            # 両側に拡張した点を追加
            for offset in range(-robot_width_cells, robot_width_cells + 1):
                px = int(x + offset * perp_dx)
                py = int(y + offset * perp_dy)
                
                if 0 <= px < self.grid_width and 0 <= py < self.grid_height:
                    all_path_points.add((px, py))
        
        # すべての経路ポイントをチェック
        for x, y in all_path_points:
            if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                cell_value = self.latest_grid[y, x]
                if cell_value >= 80:  # 障害物または障害物周辺
                    obstacle_found = True
                    
                    # 障害物が経路の左右どちらに多いかを判定
                    if (x - center_x) * (target_grid_y - center_y) - (y - center_y) * (target_grid_x - center_x) > 0:
                        left_obstacles += 1
                    else:
                        right_obstacles += 1
        
        self.path_clear = not obstacle_found
        
        # 障害物がある場合は回避方向を決定
        if obstacle_found:
            if left_obstacles <= right_obstacles:
                self.avoid_direction = -1.0  # 左回避
            else:
                self.avoid_direction = 1.0   # 右回避
    def bresenham_line(self, x0, y0, x1, y1):
        """ブレゼンハムアルゴリズムで線分上のグリッドセルを取得"""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        
        return points
    
    def publish_cmd(self, linear, angular):
        """速度コマンドをパブリッシュ"""
        cmd = Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        self.cmd_pub.publish(cmd)

def main(args=None):
        rclpy.init(args=args)
        node = FollowPersonWithObstacleAvoidance()
        
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            node.get_logger().error(f'Unexpected error: {str(e)}')
        finally:
            # 終了時の後処理
            node.publish_cmd(0.0, 0.0)  # 停止コマンド送信
            cv2.destroyAllWindows()
            node.destroy_node()
            rclpy.shutdown()
    
if __name__ == '__main__':
        main()