from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition

def generate_launch_description():
    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'max_linear_speed',
            default_value='0.5',
            description='Maximum linear speed of the robot'
        ),
        DeclareLaunchArgument(
            'target_distance',
            default_value='1.2',
            description='Target following distance'
        ),
        DeclareLaunchArgument(
            'update_rate',
            default_value='10.0',
            description='Control loop update rate'
        ),

        # Person follower node
        Node(
            package='follow_person',
            executable='person_follower',
            name='person_follower',
            parameters=[{
                'max_linear_speed': LaunchConfiguration('max_linear_speed'),
                'target_distance': LaunchConfiguration('target_distance'),
                'update_rate': LaunchConfiguration('update_rate'),
            }],
            output='screen'
        )
    ])
