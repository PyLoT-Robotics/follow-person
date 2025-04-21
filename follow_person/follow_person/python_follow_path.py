from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
import rclpy
import rclpy.duration
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, CameraInfo
from geometry_msgs.msg import Twist, Pose, PoseStamped, PoseWithCovarianceStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
from nav_msgs.msg import Path
from builtin_interfaces.msg import Duration
from tf2_ros import TransformListener, Buffer

class FollowPerson(Node):
    def __init__(self):
        super().__init__('follow_person_node')

        # Create publishers/subscribers.
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        # Create subscribers for camera and lidar data.
        self.create_subscription(Image, '/d455/camera/color/image_raw', self.image_callback, 10)
        self.create_subscription(Image, '/d455/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.create_subscription(CameraInfo, '/d455/camera/color/camera_info', self.camera_info_callback, 10)
        self.create_subscription(PoseWithCovarianceStamped, "/mcl_pose", self.mcl_pose_callback, 10)

        ##debug publisher
        self.debug_goal_pub = self.create_publisher(PoseStamped, '/debug_goalpose', 10)
        self.debug_path_pub = self.create_publisher(Path, '/debug_path', 10)
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)



        self.bridge = CvBridge()
        self.latest_color = None
        self.latest_depth = None
        self.obstacle_detected = False
        self.camera_info_received = False
        self.obstacle_distance = float('inf')
        
        self.nav2 = BasicNavigator()
        self.navigation_in_progress = False
        self.navigation_start_time = None
        self.navigation_timeout = 30.0  # seconds
        self.robot_pose = PoseWithCovarianceStamped()

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

        # trajectory recording parameters
        self.trajectory = []
        self.max_trajectory_points = 1000  # maximum number of points to record
        self.min_distance_between_points = 0.1  # minimum distance between points to record (m)
        self.iter = 0

        self.create_timer(0.1, self.timer_callback)
        self.create_timer(0.5, self.check_navigation_status)  # Check navigation status every 0.1 seconds

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
    def mcl_pose_callback(self, msg: Pose):
        # Update the robot's pose
        self.robot_pose =  PoseStamped()
        self.robot_pose.header = msg.header
        self.robot_pose.pose = msg.pose.pose
        self.robot_pose.pose.position.z = 0.0  # Set z to 0 for 2D navigation
        

    def timer_callback(self):
        self.iter += 1
        if self.latest_color is None or self.latest_depth is None:
            self.get_logger().warn('No image or depth data.')
            return

        # Run YOLO detection on the latest color image.
        # stream=Trueの場合はジェネレーターが返されるので、forループで処理する
        results_gen = self.model(self.latest_color, stream=True)
        
        # ジェネレーターから最初の結果を取得
        self.get_logger().info('Processing detection results...')
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
            X = (center_u - self.cx) * Z / self.fx   
            # convert to meters
            self.get_logger().info(f'X: {X}, Y: {Z}')
            goal = Pose()
            goal.header.stamp = self.get_clock().now().to_msg()
            goal.header.frame_id = 'camera_link'
            goal.pose.position.x = X
            goal.pose.position.y = Z
            goal.pose.position.z = 0.0
            goal.pose.orientation.w = 1.0  # no rotation
            goal.pose.orientation.x = 0.0
            goal.pose.orientation.y = 0.0
            goal.pose.orientation.z = 0.0
            

            self.tf_buffer.can_transform('map', 'camera_link', self.get_clock().now(), rclpy.duration.Duration(seconds=1.0))
            # Transform the goal pose to the map frame
            goal_transformed = self.tf_buffer.transform(goal, 'map')
            self.get_logger().info(f'Transformed goal: {goal_transformed}')
            self.debug_goal_pub.publish(goal)


            path = self.nav2.getPath(self.robot_pose, goal)
            self.get_logger().info(f"{goal}")
            if path is None or len(path.poses) == 0:
                self.get_logger().warn('No path found.')
                self.publish_cmd(0.0, 0.0)
                return
            # Smooth the path (optional)
            smoothed_path = self.nav2.smoothPath(path)
            # Follow the path to the detected person
            self.get_logger().info(f'Following path to person at {goal.pose.position.x}, {goal.pose.position.y}')
            self.nav2.followPath([goal])  # Follow the path to the detected person
            i = 0
            while not self.nav2.isTaskComplete():
                ################################################
                #
                # Implement some code here for your application!
                #
                ################################################

                # Do something with the feedback
                i += 1
                feedback = self.nav2.getFeedback()
                if feedback and i % 5 == 0:
                    print('Estimated distance remaining to goal position: ' +
                        '{0:.3f}'.format(feedback.distance_to_goal) +
                        '\nCurrent speed of the robot: ' +
                        '{0:.3f}'.format(feedback.speed))

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
            
             # ナビゲーション開始時刻を記録
            self.navigation_in_progress = True
            self.navigation_start_time = self.get_clock().now().seconds_nanoseconds()[0]
            self.get_logger().info('Navigation started.')
            self.get_logger().info(f'Navigation in progress... Elapsed time: {self.get_clock().now().seconds_nanoseconds()[0] - self.navigation_start_time} seconds')
        except StopIteration:
            self.get_logger().warn('No detection results.')
        cv2.imshow('YOLO', annotated_image)
        cv2.waitKey(1)

    def publish_cmd(self, linear_x, angular_z):
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        #self.cmd_pub.publish(twist)

     #新しい関数：ナビゲーション状態を監視

    def check_navigation_status(self):
        # ナビゲーションが進行中でない場合はスキップ
        self.get_logger().info('Checking navigation status...')
        if not self.navigation_in_progress:
            return
        self.get_logger().info('Navigation in progress...')
        self.get_logger().info(f'Elapsed time: {self.get_clock().now().seconds_nanoseconds()[0] - self.navigation_start_time} seconds')
        # タスク完了確認
        if self.nav2.isTaskComplete():
            self.get_logger().info('Navigation completed.')
            self.navigation_in_progress = False
            return
        
        # タイムアウト確認
        current_time = self.get_clock().now().seconds_nanoseconds()[0]
        elapsed_time = current_time - self.navigation_start_time
        
        if elapsed_time > self.navigation_timeout:
            self.get_logger().warn(f'Navigation timed out after {elapsed_time} seconds.')
            self.nav2.cancelTask()
            self.navigation_in_progress = False
        else:
            # 進捗状況を確認（オプション）
            try:
                feedback = self.nav2.getFeedback()
                distance_remaining = feedback.distance_remaining
                self.get_logger().info(f'Distance remaining: {distance_remaining}m')
            except Exception as e:
                self.get_logger().error(f'Failed to get feedback: {str(e)}')

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