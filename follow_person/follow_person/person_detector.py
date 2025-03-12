#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2D, Detection2DArray
import tf2_ros
from tf2_ros import TransformException
import math

class PersonDetector:
    def __init__(self, node: Node):
        self.node = node
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, node)
        
        # Parameters
        self.tracking_radius = 1.0  # meters from last known position
        self.person_confidence_threshold = 0.5  # YOLO検出の信頼度閾値
        self.last_person_position = None
        self.detection_confidence = 0.0
        
    def process_yolo_detection(self, detection_msg: Detection2DArray) -> tuple:
        """Process YOLO detections to find person."""
        # personクラスの検出のみをフィルタリング
        person_detections = [
            det for det in detection_msg.detections
            if (det.results[0].hypothesis.class_id == "person" and 
                det.results[0].hypothesis.score > self.person_confidence_threshold)
        ]
        
        if not person_detections:
            return None, 0.0

        # 最も信頼度の高い人物検出を選択
        best_detection = max(
            person_detections,
            key=lambda x: x.results[0].hypothesis.score
        )
        
        # 画像座標から3D座標への変換
        try:
            position = self._convert_detection_to_3d(best_detection)
            if position is None:
                return None, 0.0
                
            self.last_person_position = position
            confidence = best_detection.results[0].hypothesis.score
            
            return position, confidence
            
        except TransformException as ex:
            self.node.get_logger().warn(f'Could not transform detection: {ex}')
            return None, 0.0

    def _convert_detection_to_3d(self, detection: Detection2D) -> np.ndarray:
        """Convert 2D detection to 3D position using depth information."""
        # ここでDepth情報を使用して3D座標に変換する処理を実装
        # この実装は使用するカメラやセンサーに依存します
        # 例: Intel RealSenseの場合はrs2_deproject_pixel_to_pointを使用
        ...

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