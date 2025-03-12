#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2D, Detection2DArray
import tf2_ros
from tf2_ros import TransformException
import math
import cv2
import torch
from sensor_msgs.msg import Image, PointCloud2, LaserScan
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
from ultralytics import YOLO

class PersonDetector:
    def __init__(self, node: Node):
        self.node = node
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, node)
        self.cv_bridge = CvBridge()
        
        # YOLOモデルの初期化（ultralytics）
        self.model = YOLO('yolov8n.pt')
        if torch.cuda.is_available():
            self.model.to('cuda')
        
        # Parameters
        self.tracking_radius = 1.0  # meters from last known position
        self.person_confidence_threshold = 0.5  # YOLO検出の信頼度閾値
        self.last_person_position = None
        self.detection_confidence = 0.0
        
        # カメラパラメータ
        self.fx = 525.0  # focal length x
        self.fy = 525.0  # focal length y
        self.cx = 319.5  # optical center x
        self.cy = 239.5  # optical center y
        
        # 最新のデータ
        self._latest_depth_image = None
        self._latest_rgb_image = None
        self._latest_scan = None
        
        # データの同期用
        self._latest_pc_result = None
        self._latest_scan_result = None
        
        # Subscribers for RealSense camera
        self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self._rgb_callback,
            10
        )
        self.create_subscription(
            Image,
            '/camera/aligned_depth_to_color/image_raw',
            self._depth_callback,
            10
        )

    def _rgb_callback(self, msg: Image):
        """RGB画像のコールバック処理"""
        try:
            self._latest_rgb_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.node.get_logger().error(f'Error converting rgb image: {e}')

    def _depth_callback(self, msg: Image):
        """Depth画像のコールバック処理"""
        try:
            self._latest_depth_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            # 無効な深度値（0やNaN）をマスク
            self._latest_depth_image[np.isnan(self._latest_depth_image)] = 0
            self._latest_depth_image[self._latest_depth_image < 0.1] = 0  # 10cm未満は無効
            self._latest_depth_image[self._latest_depth_image > 10.0] = 0  # 10m以上は無効
        except Exception as e:
            self.node.get_logger().error(f'Error converting depth image: {e}')

    def process_pointcloud(self, msg: PointCloud2) -> tuple:
        """PointCloudからの人物検出処理"""
        if self._latest_rgb_image is None or self._latest_depth_image is None:
            return None, 0.0

        # YOLOで人物検出を実行
        results = self.model(self._latest_rgb_image)
        
        if not results[0].boxes:  # 検出結果がない場合
            return None, 0.0
            
        # 人物の検出結果をフィルタリング（クラス0は'person'）
        person_detections = []
        for box in results[0].boxes:
            if box.cls[0] == 0 and box.conf[0] >= self.person_confidence_threshold:
                person_detections.append({
                    'box': box.xyxy[0].cpu().numpy(),  # [x1, y1, x2, y2]
                    'confidence': float(box.conf[0])
                })

        if not person_detections:
            return None, 0.0

        # 最も信頼度の高い検出を選択
        best_detection = max(person_detections, key=lambda x: x['confidence'])
        box = best_detection['box']
        
        # バウンディングボックスの中心を計算
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        
        # PointCloud2からの3D位置推定
        points_list = []
        box_width = box[2] - box[0]
        box_height = box[3] - box[1]
        search_radius = min(box_width, box_height) * 0.2  # バウンディングボックスの20%の範囲で探索
        
        for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            # 2D投影
            if point[2] == 0:  # z=0の点は無視
                continue
                
            pixel_x = int((point[0] * self.fx) / point[2] + self.cx)
            pixel_y = int((point[1] * self.fy) / point[2] + self.cy)
            
            # バウンディングボックスの中心周辺の点のみを使用
            if abs(pixel_x - center_x) < search_radius and abs(pixel_y - center_y) < search_radius:
                points_list.append(point)

        if not points_list:
            # PointCloudからの推定が失敗した場合、Depthカメラの情報を使用
            position = self._convert_detection_to_3d_from_pixels(center_x, center_y)
            return position, best_detection['confidence']

        # 検出された3D点群の中央値を計算（外れ値に強い）
        points_array = np.array(points_list)
        median_position = np.median(points_array, axis=0)
        transformed_position = self._transform_to_base_link(median_position, msg.header.frame_id)
        
        return transformed_position, best_detection['confidence']

    def process_scan(self, msg: LaserScan) -> tuple:
        """LaserScanからの人物検出処理"""
        if self.last_person_position is None:
            return None, 0.0
            
        # 前回の検出位置の周辺をスキャン
        angles = np.arange(
            msg.angle_min,
            msg.angle_max + msg.angle_increment,
            msg.angle_increment
        )
        ranges = np.array(msg.ranges)
        
        # 有効な測定値をフィルタリング
        valid_mask = (ranges >= msg.range_min) & (ranges <= msg.range_max)
        valid_angles = angles[valid_mask]
        valid_ranges = ranges[valid_mask]
        
        if len(valid_ranges) == 0:
            return None, 0.0
            
        # 検出された物体をxy座標に変換
        x = valid_ranges * np.cos(valid_angles)
        y = valid_ranges * np.sin(valid_angles)
        points = np.column_stack((x, y))
        
        # 前回の位置周辺の点を探索
        dist_to_last = np.linalg.norm(points - self.last_person_position[:2], axis=1)
        nearby_points = points[dist_to_last < self.tracking_radius]
        
        if len(nearby_points) == 0:
            return None, 0.0
            
        # 近傍点の重心を計算
        centroid = np.mean(nearby_points, axis=0)
        position = np.array([centroid[0], centroid[1], 0.0])
        confidence = 0.3  # LaserScanは低い信頼度を設定
        
        return position, confidence

    def fuse_detections(self, pc_detection: tuple, scan_detection: tuple) -> tuple:
        """PointCloudとLaserScanの検出結果を統合"""
        if pc_detection[0] is None and scan_detection[0] is None:
            return None, 0.0

        if pc_detection[0] is None:
            return scan_detection

        if scan_detection[0] is None:
            return pc_detection

        # 両方の検出がある場合は、信頼度による重み付き平均を計算
        total_confidence = pc_detection[1] + scan_detection[1]
        if total_confidence == 0:
            return None, 0.0

        pc_weight = pc_detection[1] / total_confidence
        scan_weight = scan_detection[1] / total_confidence

        fused_position = (pc_detection[0] * pc_weight + scan_detection[0] * scan_weight)
        fused_confidence = max(pc_detection[1], scan_detection[1])  # 高い方の信頼度を採用

        self.last_person_position = fused_position  # 最新の位置を更新
        return fused_position, fused_confidence

    def _convert_detection_to_3d_from_pixels(self, pixel_x: float, pixel_y: float) -> np.ndarray:
        """2D検出を3D位置に変換"""
        if self._latest_depth_image is None:
            return None
            
        # 画像座標から3D座標への変換
        depth = self._latest_depth_image[int(pixel_y), int(pixel_x)]
        if depth == 0 or np.isnan(depth):
            return None
            
        # ピンホールカメラモデルを使用した3D変換
        x = (pixel_x - self.cx) * depth / self.fx
        y = (pixel_y - self.cy) * depth / self.fy
        z = depth
        
        position = np.array([x, y, z])
        
        # カメラ座標系からbase_link座標系への変換
        transformed_position = self._transform_to_base_link(position, 'camera_link')
        
        return transformed_position

    def _transform_to_base_link(self, position: np.ndarray, frame_id: str) -> np.ndarray:
        """Transform position from camera frame to base_link frame."""
        try:
            transform = self.tf_buffer.lookup_transform(
                'base_link',
                frame_id,
                rclpy.time.Time()
            )
            
            # 変換行列の作成と適用
            rotation = self._quaternion_to_rotation_matrix(
                transform.transform.rotation
            )
            translation = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])
            
            return position @ rotation.T + translation
            
        except TransformException as ex:
            self.node.get_logger().warn(f'Could not transform position: {ex}')
            return None

    def _quaternion_to_rotation_matrix(self, quaternion):
        """Convert quaternion to rotation matrix."""
        qx = quaternion.x
        qy = quaternion.y
        qz = quaternion.z
        qw = quaternion.w
        
        r11 = 1 - 2*qy*qy - 2*qz*qz
        r12 = 2*qx*qy - 2*qz*qw
        r13 = 2*qx*qz + 2*qy*qw
        r21 = 2*qx*qy + 2*qz*qw
        r22 = 1 - 2*qx*qx - 2*qz*qz
        r23 = 2*qy*qz - 2*qx*qw
        r31 = 2*qx*qz - 2*qy*qw
        r32 = 2*qy*qz + 2*qx*qw
        r33 = 1 - 2*qx*qx - 2*qy*qy
        
        return np.array([
            [r11, r12, r13],
            [r21, r22, r23],
            [r31, r32, r33]
        ])