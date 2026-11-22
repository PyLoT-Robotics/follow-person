# Follow Person

Experimental ROS 2 person-following scripts using RealSense RGB-D images,
YOLOv8 pose detection, and (in the obstacle-avoidance variant) 2D LiDAR and odometry.
Developed by PyLoT Robotics for RoboCup@Home. Target environment: Ubuntu 22.04,
ROS 2 Humble, Python 3.10, and an OpenCV desktop display.

## Current implementation status

This repository contains research scripts, not a complete service/action-based
package. The earlier README described interfaces that are not implemented by the
available follower scripts.

| Component | Status |
|---|---|
| `person_follow_obstable_detect.py` | Standalone experimental RGB-D / LiDAR / odometry follower; spelling is the actual filename |
| `person_follow.py` | Contains unresolved merge-conflict markers and imports external `pylot_msgs`; do not use as the quick start |
| `ros2 run follow_person person_follower` / bundled launch | Entry point refers to missing `follow_person.person_follower`; currently unavailable |
| `follow_person_msgs` | Service/action definitions exist, but no corresponding server is implemented by the documented script |
| `config/params.yaml` | Not loaded by the standalone script; its control values are defined in source |

The instructions below exercise the standalone obstacle-avoidance script with
velocity output redirected to a debug topic. This documentation update does not
repair the package entry point or validate physical following behavior.

## Installation

Install ROS 2 Humble and initialize rosdep before these steps.

```bash
source /opt/ros/humble/setup.bash
mkdir -p ~/follow_ws/src
cd ~/follow_ws/src
git clone https://github.com/PyLoT-Robotics/follow-person.git
cd ~/follow_ws
rosdep install --from-paths src --ignore-src -r -y
sudo apt install python3-venv python3-pip python3-opencv python3-numpy
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install 'numpy<2' 'ultralytics>=8.0.0' 'torch>=2.0.0' 'opencv-python>=4.5.0,<4.12'
```

Use the ROS-compatible system Python rather than Conda so that `rclpy` and
`cv_bridge` can load. The virtual environment inherits ROS/system packages.
The NumPy/OpenCV bounds avoid pulling a NumPy 2 build into Humble's binary
`cv_bridge` environment. These are compatibility bounds, not a tested lockfile.
The documented standalone script does not need `colcon build` or the custom
service/action messages. Build those separately only if developing their servers.

## Prepare perception inputs

The follower runs YOLO internally; no separate person-detector node is required.
It loads `yolov8n-pose.pt` relative to the working directory. The repository
contains that file in `follow_person/follow_person/`; run from there as below.
Ultralytics may fetch additional tracking assets on first use. Model weights and
Ultralytics have their own licensing terms.

Start your camera, LiDAR and odometry publishers separately. They are not launched
by this repository. Match or remap these inputs:

| Input topic | ROS type | Expected content |
|---|---|---|
| `/d455/camera/color/image_raw` | `sensor_msgs/msg/Image` | Color image convertible to BGR8 |
| `/d455/camera/depth/image_rect_raw` | `sensor_msgs/msg/Image` | Depth converted by cv_bridge to 32FC1; code divides values by 1000 |
| `/d455/camera/color/camera_info` | `sensor_msgs/msg/CameraInfo` | Color-camera intrinsics |
| `/scan_filtered` | `sensor_msgs/msg/LaserScan` | Filtered 2D LiDAR scan |
| `/odometry/filtered` | `nav_msgs/msg/Odometry` | Base odometry for accumulated obstacle mapping |

Depth must correspond pixel-for-pixel to color: the script samples depth at color
image coordinates and does not align images itself. Check units explicitly:
it assumes millimeter-valued depth before division by 1000. A 32FC1 stream already
in meters needs a unit-conversion change; changing encoding alone does not fix
units. Camera principal-point defaults and LiDAR offsets also live in source.

For example, inspect the actual topics before choosing camera remaps:

```bash
ros2 topic list -t
ros2 topic hz /d455/camera/color/image_raw
ros2 topic hz /d455/camera/depth/image_rect_raw
ros2 topic echo /d455/camera/color/camera_info --once
ros2 topic hz /scan_filtered
ros2 topic hz /odometry/filtered
```

## Run and inspect output

In a sourced terminal with the virtual environment active:

```bash
cd ~/follow_ws/src/follow-person/follow_person/follow_person
python person_follow_obstable_detect.py --ros-args \
  -r /cmd_vel:=/follow_person/debug_cmd_vel
```

Adapt input names with ROS remapping, for example:

```bash
python person_follow_obstable_detect.py --ros-args \
  -r /cmd_vel:=/follow_person/debug_cmd_vel \
  -r /scan_filtered:=/scan \
  -r /d455/camera/color/image_raw:=/camera/camera/color/image_raw \
  -r /d455/camera/depth/image_rect_raw:=/camera/camera/aligned_depth_to_color/image_raw \
  -r /d455/camera/color/camera_info:=/camera/camera/color/camera_info
```

The camera topic names above are examples: inspect your own driver's output and
verify depth encoding/units before using them. OpenCV windows require a desktop.

In another sourced terminal:

```bash
ros2 node info /follow_person_obstacle_avoid_node
ros2 topic echo /follow_person/debug_cmd_vel
ros2 topic echo /laser_map --once
```

Check that camera frames arrive, detections appear in the OpenCV window, the
local obstacle map updates, and debug velocity responds to a person moving.
No detections should produce zero velocity. Stop with Ctrl+C. This is an
inspection procedure, not a claim of completed hardware validation.

## Connecting a mobile base

The script publishes `geometry_msgs/msg/Twist` to `/cmd_vel` without an explicit
start service; removing the debug remap enables that output immediately. Review
the source control gains and bounds for your base first, verify depth geometry,
and use your robot's velocity limiter/watchdog and physical stop mechanism.
Do not rely on the unimplemented `max_linear_speed` launch argument or the
`/follow_control` service to constrain/stop this script.

## Configuration and outputs

The standalone class is `FollowPersonWithObstacleAvoidance`. Its source defaults
include `follow_distance=1.0` m, `linear_gain=0.8`, `angular_gain=1.0`,
`obstacle_threshold=1.5` m and a 0.1 s timer. Edit source to change these values;
they are not declared ROS parameters.

| Output | Type | Purpose |
|---|---|---|
| `/cmd_vel` (remap in quick start) | `geometry_msgs/msg/Twist` | Following commands |
| `/laser_map` | `nav_msgs/msg/OccupancyGrid` | Local obstacle-map visualization |

## Known limitations / next steps

- Repair the missing entry point and merge conflicts before advertising a package launch.
- Implement service/action servers before documenting start/pause/resume controls.
- Parameterize camera geometry, depth units, gains and velocity limits.
- Add reproducible sensor-bag tests and hardware performance measurements.
- Image/depth synchronization and robust target selection need further validation.

## License

[Apache-2.0](LICENSE) for repository code. Third-party dependencies and model
weights retain their own licenses.
