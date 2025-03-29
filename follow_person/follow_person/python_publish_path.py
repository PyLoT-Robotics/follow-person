import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
import tf2_ros
from tf2_geometry_msgs import do_transform_pose_stamped  # これを追加

class test_path_publish(Node):
    
    def __init__(self):
        super().__init__('test_path_publish_node')
        self.path_pub = self.create_publisher(Path, 'path', 10)
        self.trajectory = []
        self.max_trajectory_points = 100
        self.iter = 0
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.create_timer(0.1, self.publish_path)


    def publish_path(self):
        self.iter += 1
        if self.iter % 5 == 0:
            self.trajectory.append([self.iter, self.iter])
        if len(self.trajectory) > self.max_trajectory_points:
            self.trajectory.pop(0)
        transformed_trajectory = self.transform_trajectory(self.trajectory)
        if self.iter % 100 == 0:
            self.publish_path_ros(transformed_trajectory)

    def publish_path_ros(self, path):
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()
        for point in path:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1])
            path_msg.poses.append(pose)
        self.get_logger().info(f'Publish path{path_msg}')

        self.path_pub.publish(path_msg)
    def transform_trajectory(self, trajectory):
        transformed_trajectory = []
        for point in trajectory:
            x, y = point
            p = PoseStamped()
            p.header.frame_id = 'camera_link'
            p.header.stamp = self.get_clock().now().to_msg()
            p.pose.position.x = float(x)
            p.pose.position.y = float(y)
            try:
                # TransformStampedを取得
                transform = self.tf_buffer.lookup_transform(
                    'base_link',
                    'camera_link',
                    rclpy.time.Time())
                # PoseStampedを変換
                transformed_point = do_transform_pose_stamped(p, transform)
                transformed_trajectory.append([transformed_point.pose.position.x, transformed_point.pose.position.y])
            except (tf2_ros.TransformException, Exception) as e:
                self.get_logger().error('Taransform error: ' + str(e))
        return transformed_trajectory
    
def main(args=None):
    rclpy.init(args=args)
    node = test_path_publish()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()