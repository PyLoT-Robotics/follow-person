#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import PointCloud2, LaserScan
from geometry_msgs.msg import Twist, Pose2D
from follow_person_msgs.action import FollowPerson
from follow_person_msgs.srv import FollowControl
from .person_detector import PersonDetector
from .motion_controller import MotionController
import numpy as np

class PersonFollowerNode(Node):
    def __init__(self):
        super().__init__('person_follower')

        # Create callback group for concurrent processing
        self.callback_group = ReentrantCallbackGroup()

        # 基本パラメータの宣言
        self.declare_parameter('max_linear_speed', 0.5)
        self.declare_parameter('max_angular_speed', 1.0)
        self.declare_parameter('target_distance', 1.2)
        self.declare_parameter('update_rate', 10.0)
        self.declare_parameter('min_obstacle_distance', 0.5)
        self.declare_parameter('distance_tolerance', 0.1)
        self.declare_parameter('angle_tolerance', 0.1)

        # PIDパラメータの宣言 - 線形速度制御
        self.declare_parameter('linear_pid.Kp', 0.5)
        self.declare_parameter('linear_pid.Ki', 0.1)
        self.declare_parameter('linear_pid.Kd', 0.2)
        self.declare_parameter('linear_pid.integral_limit', 1.0)

        # PIDパラメータの宣言 - 角速度制御
        self.declare_parameter('angular_pid.Kp', 2.0)
        self.declare_parameter('angular_pid.Ki', 0.1)
        self.declare_parameter('angular_pid.Kd', 0.4)
        self.declare_parameter('angular_pid.integral_limit', 1.0)

        # 人物検出パラメータの宣言
        self.declare_parameter('detection.confidence_threshold', 0.5)
        self.declare_parameter('detection.tracking_radius', 1.0)

        # カメラパラメータの宣言
        self.declare_parameter('camera.fx', 525.0)
        self.declare_parameter('camera.fy', 525.0)
        self.declare_parameter('camera.cx', 319.5)
        self.declare_parameter('camera.cy', 239.5)

        # Initialize components
        self.person_detector = PersonDetector(self)
        self.motion_controller = MotionController(self)

        # Set basic parameters
        self.motion_controller.max_linear_speed = self.get_parameter('max_linear_speed').value
        self.motion_controller.max_angular_speed = self.get_parameter('max_angular_speed').value
        self.motion_controller.target_following_distance = self.get_parameter('target_distance').value
        self.motion_controller.min_obstacle_distance = self.get_parameter('min_obstacle_distance').value
        self.motion_controller.distance_tolerance = self.get_parameter('distance_tolerance').value
        self.motion_controller.angle_tolerance = self.get_parameter('angle_tolerance').value

        # Set PID parameters - Linear
        self.motion_controller.linear_pid['Kp'] = self.get_parameter('linear_pid.Kp').value
        self.motion_controller.linear_pid['Ki'] = self.get_parameter('linear_pid.Ki').value
        self.motion_controller.linear_pid['Kd'] = self.get_parameter('linear_pid.Kd').value
        self.motion_controller.linear_pid['integral_limit'] = self.get_parameter('linear_pid.integral_limit').value

        # Set PID parameters - Angular
        self.motion_controller.angular_pid['Kp'] = self.get_parameter('angular_pid.Kp').value
        self.motion_controller.angular_pid['Ki'] = self.get_parameter('angular_pid.Ki').value
        self.motion_controller.angular_pid['Kd'] = self.get_parameter('angular_pid.Kd').value
        self.motion_controller.angular_pid['integral_limit'] = self.get_parameter('angular_pid.integral_limit').value

        # Set detection parameters
        self.person_detector.person_confidence_threshold = self.get_parameter('detection.confidence_threshold').value
        self.person_detector.tracking_radius = self.get_parameter('detection.tracking_radius').value

        # Set camera parameters
        self.person_detector.fx = self.get_parameter('camera.fx').value
        self.person_detector.fy = self.get_parameter('camera.fy').value
        self.person_detector.cx = self.get_parameter('camera.cx').value
        self.person_detector.cy = self.get_parameter('camera.cy').value
        
        # 最新のセンサーデータ
        self._latest_pointcloud = None
        self._latest_scan = None
        
        # Subscribers
        self.create_subscription(
            PointCloud2,
            '/camera/depth/color/points',
            self._pointcloud_callback,
            10,
            callback_group=self.callback_group
        )
        self.create_subscription(
            LaserScan,
            '/scan',
            self._scan_callback,
            10,
            callback_group=self.callback_group
        )

        # Publishers
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            'cmd_vel',
            10
        )

        # Action server
        self._action_server = ActionServer(
            self,
            FollowPerson,
            'follow_person',
            self._execute_follow_callback,
            callback_group=self.callback_group
        )

        # Service server
        self.control_service = self.create_service(
            FollowControl,
            'follow_control',
            self._control_callback,
            callback_group=self.callback_group
        )

        # State
        self.current_state = 'STOPPED'
        self.total_distance = 0.0
        self.last_position = None
        self.is_active = False
        self.goal_handle = None

        # Timer for control loop
        self.update_timer = self.create_timer(
            1.0/self.get_parameter('update_rate').value,
            self._update_callback,
            callback_group=self.callback_group
        )

        self.get_logger().info('Person follower node initialized')

    def _pointcloud_callback(self, msg: PointCloud2):
        """Handle incoming pointcloud data."""
        try:
            self._latest_pointcloud = msg  # 最新のPointCloudを保存
            if not self.is_active:
                return
            # Update obstacle detection
            self.motion_controller.update_obstacles_pc(msg)
        except Exception as e:
            self.get_logger().error(f'Error in pointcloud callback: {str(e)}')

    def _scan_callback(self, msg: LaserScan):
        """Handle incoming laser scan data."""
        try:
            self._latest_scan = msg  # 最新のLaserScanを保存
            if not self.is_active:
                return
            # Update obstacle detection
            self.motion_controller.update_obstacles_scan(msg)
        except Exception as e:
            self.get_logger().error(f'Error in scan callback: {str(e)}')

    def _update_callback(self):
        """Main control loop."""
        if not self.is_active or self.goal_handle is None:
            return

        try:
            if self._latest_pointcloud is None:
                self.get_logger().warn('No PointCloud data received')
                return

            # Get person position from sensors
            pc_detection = self.person_detector.process_pointcloud(self._latest_pointcloud)
            scan_detection = self.person_detector.process_scan(self._latest_scan)
            
            # Fuse detections
            position, confidence = self.person_detector.fuse_detections(pc_detection, scan_detection)

            # 検出結果のログ出力
            if position is not None:
                self.get_logger().debug(
                    f'Person detected at: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}) '
                    f'with confidence: {confidence:.2f}'
                )
            else:
                self.get_logger().debug('No person detected')

            # Update total distance traveled if we have valid positions
            if position is not None and self.last_position is not None:
                delta = np.linalg.norm(position[:2] - self.last_position[:2])
                self.total_distance += delta
                if delta > 0.1:  # 10cm以上移動した場合のみログ出力
                    self.get_logger().debug(f'Distance moved: {delta:.2f}m, Total: {self.total_distance:.2f}m')

            self.last_position = position if position is not None else self.last_position

            # Compute velocity commands
            cmd_vel = self.motion_controller.compute_velocity(position, confidence)
            
            # Publish velocity commands
            self.cmd_vel_pub.publish(cmd_vel)

            # 速度コマンドのログ出力（有意な動きがある場合のみ）
            if abs(cmd_vel.linear.x) > 0.01 or abs(cmd_vel.angular.z) > 0.01:
                self.get_logger().debug(
                    f'Command velocity - Linear: {cmd_vel.linear.x:.2f} m/s, '
                    f'Angular: {cmd_vel.angular.z:.2f} rad/s'
                )

        except Exception as e:
            self.get_logger().error(f'Error in control loop: {str(e)}')
            self._stop_robot()  # エラー時は安全のため停止

        # Update action feedback
        if self.goal_handle is not None and self.goal_handle.is_active:
            feedback_msg = FollowPerson.Feedback()
            feedback_msg.current_state = self.current_state
            if position is not None:
                feedback_msg.current_distance_to_person = np.linalg.norm(position[:2])
                feedback_msg.person_position = Pose2D(x=position[0], y=position[1], theta=0.0)
            feedback_msg.current_speed = cmd_vel.linear.x
            self.goal_handle.publish_feedback(feedback_msg)

    async def _execute_follow_callback(self, goal_handle):
        """Execute the follow person action."""
        self.get_logger().info('Executing follow person action')
        
        # Update parameters from goal
        self.motion_controller.target_following_distance = goal_handle.request.target_distance
        self.motion_controller.max_linear_speed = goal_handle.request.max_speed
        
        # Store goal handle and activate
        self.goal_handle = goal_handle
        self.is_active = True
        self.current_state = 'TRACKING'
        self.total_distance = 0.0
        
        # Keep running until goal is canceled or fails
        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.is_active = False
                self.current_state = 'STOPPED'
                self._stop_robot()
                return FollowPerson.Result(
                    success=False,
                    message='Follow action canceled',
                    total_distance_traveled=self.total_distance
                )
            
            if not self.is_active:
                return FollowPerson.Result(
                    success=False,
                    message='Follow action stopped',
                    total_distance_traveled=self.total_distance
                )
            
            await rclpy.time.sleep(0.1)  # Prevent busy waiting
        
        # Should only reach here if node is shutting down
        return FollowPerson.Result(
            success=False,
            message='Node shutting down',
            total_distance_traveled=self.total_distance
        )

    def _control_callback(self, request, response):
        """Handle service control requests."""
        if request.command == 'start':
            if not self.is_active:
                self.is_active = True
                self.current_state = 'TRACKING'
                response.success = True
                response.message = 'Started following'
            else:
                response.success = False
                response.message = 'Already active'
        
        elif request.command == 'stop':
            if self.is_active:
                self.is_active = False
                self.current_state = 'STOPPED'
                self._stop_robot()
                response.success = True
                response.message = 'Stopped following'
            else:
                response.success = False
                response.message = 'Already stopped'
        
        elif request.command == 'pause':
            if self.is_active and self.current_state == 'TRACKING':
                self.current_state = 'PAUSED'
                self._stop_robot()
                response.success = True
                response.message = 'Paused following'
            else:
                response.success = False
                response.message = 'Not in tracking state'
        
        elif request.command == 'resume':
            if self.is_active and self.current_state == 'PAUSED':
                self.current_state = 'TRACKING'
                response.success = True
                response.message = 'Resumed following'
            else:
                response.success = False
                response.message = 'Not in paused state'
        
        else:
            response.success = False
            response.message = f'Unknown command: {request.command}'
        
        response.state = self.current_state
        return response

    def _stop_robot(self):
        """Stop the robot by publishing zero velocity."""
        cmd_vel = Twist()
        self.cmd_vel_pub.publish(cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    
    node = PersonFollowerNode()
    
    # Use multi-threaded executor for concurrent callbacks
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
