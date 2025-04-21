#! /usr/bin/env python3
# Copyright 2021 Samsung Research America
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
import rclpy
from rclpy.duration import Duration
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from rclpy.node import Node
"""
Basic navigation demo to go to pose.
"""
class navigation2_demo(Node):
    def __init__(self):
        super().__init__('navigation2_demo') 
    
        self.nav2 = BasicNavigator()

        self.create_subscription(
            PoseWithCovarianceStamped,
            '/mcl_pose',
            self.pose_callback,
            10
        )
        self.pub_ = self.create_publisher(
            PoseStamped,
            '/debugGoalpose',
            10
        )
        self.current_pose = PoseWithCovarianceStamped()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.nav2 = BasicNavigator()

    def pose_callback(self, msg):
        # Handle the incoming pose message
        self.get_logger().info('Received pose: {}'.format(msg.pose.pose))
        self.current_pose = msg

    def get_transform(self, target_frame, source_frame):
        # Transform the pose from source frame to target frame
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                Duration(seconds=1.0))
            return transform
        except TransformException as ex:
            self.get_logger().error(f'Could not transform {target_frame} to {source_frame}: {ex}')
            return None
    def tranformposestamped(self, posestamped):
        # 最初に変換結果用のオブジェクトを初期化
        transformed_pose = PoseStamped()
        transformed_pose.header.frame_id = "map"
        transformed_pose.header.stamp = self.get_clock().now().to_msg()
        
        # デフォルト値を設定
        transformed_pose.pose.position.x = 0.0
        transformed_pose.pose.position.y = 0.0
        transformed_pose.pose.position.z = 0.0
        transformed_pose.pose.orientation.w = 1.0
        transformed_pose.pose.orientation.x = 0.0
        transformed_pose.pose.orientation.y = 0.0
        transformed_pose.pose.orientation.z = 0.0
        
        try:
            t = self.get_transform(
                "map",
                "camera_link")
            
            # 変換が成功したら、変換結果を設定
            transformed_pose.pose.position.x = t.transform.translation.x + posestamped.pose.position.x
            transformed_pose.pose.position.y = t.transform.translation.y + posestamped.pose.position.y
            transformed_pose.pose.position.z = 0.0
            transformed_pose.pose.orientation.x = t.transform.rotation.x
            transformed_pose.pose.orientation.y = t.transform.rotation.y
            transformed_pose.pose.orientation.z = t.transform.rotation.z
            transformed_pose.pose.orientation.w = t.transform.rotation.w
            
            # 変換結果をログ出力
            self.get_logger().info('Transformed pose: {}'.format(transformed_pose.pose))
        except TransformException as ex:
            self.get_logger().info(f'Could not transform map to camera link: {ex}')
            # odomフレームを試す
            try:
                t = self.tf_buffer.lookup_transform(
                    "odom",
                    "camera_link",
                    rclpy.time.Time())
                
                # odomフレームで変換を行う
                transformed_pose.header.frame_id = "odom"
                transformed_pose.pose.position.x = t.transform.translation.x + posestamped.pose.position.x
                transformed_pose.pose.position.y = t.transform.translation.y + posestamped.pose.position.y
                transformed_pose.pose.orientation = t.transform.rotation
                
                self.get_logger().info('Used odom frame instead of map')
            except TransformException as ex2:
                self.get_logger().error(f'Could not transform to any navigation frame: {ex2}')
                # ここは基本的なフォールバックポーズを使用する
        
        # 変換結果を返す（例外発生時も初期化済みのオブジェクトを返す）
        return transformed_pose
    
    def publish_goal_pose(self, goal_pose):
        # Publish the goal pose to the debugGoalpose topic
        self.pub_.publish(goal_pose)
        for i in range(10):
            self.get_logger().info('Published goal pose: {}'.format(goal_pose))

    def check_tf_path(self, target_frame, source_frame):
        """フレーム間のパスを詳細に調査"""
        if not self.tf_buffer.can_transform(target_frame, source_frame, self.get_clock().now()):
            self.get_logger().error(f"Cannot transform {target_frame} to {source_frame}")
            
            # TFフレームをチェックするためのリスト
            check_frames = ['map', 'odom', 'base_link', 'base_footprint', 'camera_link']
            
            # 各フレームの存在を確認する安全な方法
            for frame in check_frames:
                try:
                    # このフレームから自分自身への変換ができるか試す
                    if self.tf_buffer.can_transform(frame, frame, self.get_clock().now()):
                        self.get_logger().info(f"Frame {frame} exists")
                    else:
                        self.get_logger().warn(f"Frame {frame} might not exist")
                except Exception as e:
                    self.get_logger().error(f"Error checking frame {frame}: {e}")
            
            # フレーム間のパスをチェック
            for i in range(len(check_frames)-1):
                for j in range(i+1, len(check_frames)):
                    try:
                        if self.tf_buffer.can_transform(check_frames[i], check_frames[j], self.get_clock().now()):
                            self.get_logger().info(f"Can transform {check_frames[i]} to {check_frames[j]}")
                        else:
                            self.get_logger().warn(f"Cannot transform {check_frames[i]} to {check_frames[j]}")
                    except Exception as e:
                        self.get_logger().error(f"Error checking transform: {e}")
            
            return False
        return True

    def go_to_pose(self, goal_pose):
        self.nav2.goToPose(goal_pose)
        i = 0
        while not self.nav2.isTaskComplete():
            ################################################
            #
            # Implement some code here for your application!
            #
            ################################################

            # Do something with the feedback
            i = i + 1
            feedback = self.nav2.getFeedback()
            if feedback and i % 5 == 0:
                print('Estimated time of arrival: ' + '{0:.0f}'.format(
                    Duration.from_msg(feedback.estimated_time_remaining).nanoseconds / 1e9)
                    + ' seconds.')

                # Some navigation timeout to demo cancellation
                if Duration.from_msg(feedback.navigation_time) > Duration(seconds=600.0):
                    self.nav2.cancelTask()

                # Some navigation request change to demo preemption
                if Duration.from_msg(feedback.navigation_time) > Duration(seconds=18.0):
                    goal_pose.pose.position.x = -3.0
                    self.nav2.goToPose(goal_pose)

        # Do something depending on the return code
        result = self.nav2.getResult()
        if result == TaskResult.SUCCEEDED:
            print('Goal succeeded!')
        elif result == TaskResult.CANCELED:
            print('Goal was canceled!')
        elif result == TaskResult.FAILED:
            print('Goal failed!')
        else:
            print('Goal has an invalid return status!')

        


def main():
    rclpy.init()

    rclpy_node = navigation2_demo()
    for i in range(10):
        rclpy.spin_once(rclpy_node)
    rclpy_node.check_tf_path("map", "camera_link")
    try:
        all_frames = rclpy_node.tf_buffer.all_frames_as_string()
        print("\n--- Available TF frames ---")
        print(all_frames)
        print("---------------------------\n")
    except Exception as e:
        print(f"Error getting frames: {e}")
    goal = PoseStamped()
    goal.header.frame_id = 'camera_link'
    goal.header.stamp = rclpy_node.get_clock().now().to_msg()
    goal.pose.position.x = 2.0
    goal.pose.position.y = 2.0
    goal.pose.position.z = 0.0
    goal.pose.orientation.x = 0.0
    goal.pose.orientation.y = 0.0
    goal.pose.orientation.z = 0.0
    goal.pose.orientation.w = 1.0
    # Publish the goal pose
    rclpy_node.publish_goal_pose(goal)
    # Transform the goal pose to the camera link frame
    transformed_goal = rclpy_node.tranformposestamped(goal)
    # Go to the transformed goal pose
    rclpy_node.go_to_pose(transformed_goal)


if __name__ == '__main__':
    main()