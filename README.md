# Follow Person Package

A ROS2 package for person following functionality using Realsense depth camera and 2D Lidar data, with obstacle avoidance capabilities.

## Features

- Person detection using Realsense depth camera (primary) and 2D Lidar (auxiliary)
- Obstacle avoidance using fused sensor data
- Action and Service interfaces for control
- Configurable parameters for following behavior
- No pre-existing map requirement
- Can be called from other programs

## Requirements

- ROS2 Humble
- Realsense camera
- 2D Lidar
- Python 3.8+
- Required ROS2 packages:
  - geometry_msgs
  - sensor_msgs
  - tf2_ros
  - rclpy

## Installation

```bash
# Clone the repository into your ROS2 workspace's src directory
cd ~/ros2_ws/src
git clone <repository_url> follow-person

# Install dependencies
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y

# Build the packages
colcon build --packages-select follow_person_msgs follow_person
```

## Usage

### Launch the Person Following Node

```bash
# Using default parameters
ros2 launch follow_person person_follower.launch.py

# With custom parameters
ros2 launch follow_person person_follower.launch.py \
    max_linear_speed:=0.6 \
    target_distance:=1.5 \
    update_rate:=15.0
```

### Using the Service Interface

The service interface provides quick commands for controlling the following behavior:

```bash
# Start following
ros2 service call /follow_control follow_person_msgs/srv/FollowControl "{command: 'start'}"

# Pause following
ros2 service call /follow_control follow_person_msgs/srv/FollowControl "{command: 'pause'}"

# Resume following
ros2 service call /follow_control follow_person_msgs/srv/FollowControl "{command: 'resume'}"

# Stop following
ros2 service call /follow_control follow_person_msgs/srv/FollowControl "{command: 'stop'}"
```

### Using the Action Interface

The action interface provides continuous feedback and more control over the following behavior:

```python
# Example Python code
from follow_person_msgs.action import FollowPerson
import rclpy
from rclpy.action import ActionClient

# Create action client
node = rclpy.create_node('follow_client')
action_client = ActionClient(node, FollowPerson, 'follow_person')

# Send goal
goal_msg = FollowPerson.Goal()
goal_msg.target_distance = 1.2
goal_msg.max_speed = 0.5
action_client.send_goal_async(goal_msg)
```

For a complete example of using both interfaces, see the [example client script](follow_person/examples/follow_person_client.py).

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| max_linear_speed | float | 0.5 | Maximum linear speed (m/s) |
| target_distance | float | 1.2 | Target following distance (m) |
| update_rate | float | 10.0 | Control loop update rate (Hz) |

## Nodes

### person_follower

Main node providing person following functionality.

#### Subscribed Topics
- `/camera/depth/color/points` (sensor_msgs/PointCloud2)
- `/scan` (sensor_msgs/LaserScan)

#### Published Topics
- `/cmd_vel` (geometry_msgs/Twist)

#### Services
- `/follow_control` (follow_person_msgs/srv/FollowControl)

#### Actions
- `/follow_person` (follow_person_msgs/action/FollowPerson)

## License

Apache License 2.0
