import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge

from pylot_msgs.srv import PersonFollow
#from detic_onnx_ros2_msgs.srv import GraspBag
from audio_common_msgs.action import TTS
from rclpy.action import ActionClient

from realsense2_camera_msgs.msg import RGBD
import time
from ultralytics import YOLO
from ultralytics.engine.results import Results
from typing import List
from ros2_vosk_msgs.srv import SpeechRecognitionWord
from geometry_msgs.msg import Twist

from std_msgs.msg import String
import pylot_utils.robot
import time
import math
from audio_common_msgs.action import TTS
from rclpy.action import ActionClient
from std_msgs.msg import String
from dynamixel_sdk_custom_interfaces.msg import SetPositionFourMotor
from dynamixel_sdk_custom_interfaces.msg import SetPositionFiveMotor
from dynamixel_sdk_custom_interfaces.msg import SetPosition
from dynamixel_sdk_custom_interfaces.srv import LiftControl
from playsound import playsound
from sensor_msgs.msg import Joy
import numpy as np
from detic_onnx_ros2_msg.msg import BoundingBoxRgbd
from nav2_simple_commander.robot_navigator import BasicNavigator,TaskResult
from geometry_msgs.msg import PoseStamped

from rclpy.duration import Duration# gpsr_manager.py内に追加
import numpy as np
import math
from sensor_msgs.msg import Image
\
def quaternion_from_euler(ai, aj, ak, axes='sxyz'):
    """
    オイラー角からクォータニオンへ変換する関数
    
    Parameters:
    ai, aj, ak : Euler角（ラジアン）
    axes : 回転順序を指定する文字列 (default: 'sxyz')
    
    Returns:
    クォータニオン [x, y, z, w]
    """
    try:
        # デフォルトはsxyz（静的回転）
        if axes == 'sxyz':
            ci = math.cos(ai/2)
            si = math.sin(ai/2)
            cj = math.cos(aj/2)
            sj = math.sin(aj/2)
            ck = math.cos(ak/2)
            sk = math.sin(ak/2)
            
            # クォータニオン計算
            x = si*cj*ck - ci*sj*sk
            y = ci*sj*ck + si*cj*sk
            z = ci*cj*sk - si*sj*ck
            w = ci*cj*ck + si*sj*sk
            
            return np.array([x, y, z, w])
    except Exception as e:
        print(f"Error in quaternion calculation: {e}")
        return np.array([0.0, 0.0, 0.0, 1.0])  # デフォルト値

class TaskManager(Node):
    def __init__(self):
        super().__init__('task_manager')

        self.debug_image_pub = self.create_publisher(
        Image, 
        '~/debug_image',  # プライベートトピック
        10
    )
    
        self.tts_client = ActionClient(self, TTS, "/say" )
        self.bridge = CvBridge()
        self.model=YOLO('/home/pylot/pylot_robocup/src/carry_my_luggage_japan/carry_my_luggage_japan/yolov8n-pose.pt')

        self.img_flag = False

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)


        #self.subscription_present_position = self.create_subscription(SetPositionFiveMotor,'get_present_position_five_motor', self.position_feedback, 10)
        #self.arm_current_subscriber = self.create_subscription(SetPositionFiveMotor, 'get_present_current_five_motor', self.arm_current_callback, 10)
        """
        self.cli_person = self.create_client(PersonLeftRight, 'person_left_right')
        while not self.cli_person.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('detect_person service not available, waiting again...')

        self.cli_bag = self.create_client(GraspBag, 'grasp_bag')
        while not self.cli_bag.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('grasp_bag service not available, waiting again...')
        
        self.req_person = PersonLeftRight.Request()
        self.req_bag = GraspBag.Request()
        
        self.cli_follow = self.create_client(PersonFollow, 'person_follow')
        while not self.cli_follow.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('detect_person service not available, waiting again...')
        self.req_follow = PersonFollow.Request()
        """
        
        self.cli = self.create_client(LiftControl, 'lift_control')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = LiftControl.Request()

        self.cli = self.create_client(SpeechRecognitionWord, 'speech_recognition/word')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req_vosk = SpeechRecognitionWord.Request()
        self.listen = False

        self.dynamixel_publisher = self.create_publisher(SetPosition, 'set_position', 10)
        self.publisher_position = self.create_publisher(SetPositionFourMotor,'set_position_four_motor', 10)
        self.sub_d455 = self.create_subscription(RGBD, '/d455/camera/rgbd', self.d455_callback, 10)
        self.d455_flag = False
        self.sub_d435 = self.create_subscription(RGBD, '/d435/camera/rgbd', self.d435_callback, 10)
        self.d435_flag  = False
        
        self.gripper_open = [16,2500]
        self.gripper_close = [16,3400]
        self.wrist_nuetral = [15,3072]
        self.set_nuetral = [2024,800,210,3178]
        #nuetral
        self.set_nuetral_2 = [2024,1600,2010,2470]
        self.set_pick=[2024,2347,1800,3096]
        self.wrist_pick=[15,3022]


        self.follow_head = [1, 1200]

        self.create_timer(0.1, self.timer_callback)
        self.fx = 645.0
        self.fy = 643.0
        self.cx = 320.0
        self.cy = 240.0
        self.target_person_id = None
        self.tracking_lost_counter = 0
        self.max_tracking_lost = 30  # Number of frames to consider tracking lost
        self.tracking_interval = 100
        self.follow_flag = False
        self.follow_distance = 0.6  # desired distance in meters to the person
        self.linear_gain = 0.6
        self.angular_gain = 1.9
        self.obstacle_threshold = 0.8  # stop if obstacle within 0.5m
        self.debug_image_pub = self.create_publisher(Image, '~/debug_image', 10)
        self.left_bag=[-0.05,-0.03,0.21]

        #
    
    def vosk_request(self):
        self.req_vosk.model_name = "vosk-model-small-en-us-0.15"
        self.req_vosk.custom_vocabulary = ["stop"]
        self.future = self.cli.call_async(self.req_vosk)
        rclpy.spin_until_future_complete(self, self.future, timeout_sec=300.0)
        if self.future.done():
            if self.future.result() is not None:
                if self.future.result().is_speech_recognized:
                    self.get_logger().info('Request succeeded')
                    self.follow_flag = False
                    self.listen = True
                    return True
                else:
                    self.get_logger().info('Request failed')
                    self.listen = True
                    return False
    
    def timer_callback(self):
         if  not self.follow_flag:
             return
         if  not self.d455_flag:
             return
         results_gen = self.model.track(self.d455_image, persist=True, classes=[0], stream=True)
        
         try:
            result = next(results_gen)
            annotated_image = result.plot()


            if 'annotated_image' in locals():
                try:
                    debug_image_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
                    debug_image_msg.header.stamp = self.get_clock().now().to_msg()
                    self.debug_image_pub.publish(debug_image_msg)
                except Exception as e:
                    self.get_logger().error('Failed to publish debug image: ' + str(e))
            
            # 人物を検出できたか確認
            person_detected = False
            
            # 検出した人物の情報を格納するリスト
            detected_persons = []
            
            # 追跡IDが存在するか確認
            has_tracking_ids = hasattr(result.boxes, 'id') and result.boxes.id is not None
            if hasattr(result, 'boxes') and len(result.boxes) > 0:
                for i, box in enumerate(result.boxes):
                    cls = int(box.cls[0])
                    if cls == 0:  # class 0 = person in COCO
                        person_detected = True
                        
                        # バウンディングボックスを取得
                        xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                        
                        # 追跡IDを取得（存在する場合）
                        person_id = int(box.id[0]) if has_tracking_ids else i
                        
                        # バウンディングボックスの中心座標を計算
                        center_u = int((xyxy[0] + xyxy[2]) / 2)
                        center_v = int((xyxy[1] + xyxy[3]) / 2)
                        
                        # 対応する深度値を取得 - 改良版
                        try:
                                self.get_logger().info(f'Person ID {person_id}: center_u={center_u}, center_v={center_v}')
                                # バウンディングボックスの範囲を取得
                                x1, y1, x2, y2 = map(int, xyxy)
                                
                                # 境界チェック - 画像の範囲内に収める
                                height, width = self.d455_depth.shape
                                x1 = max(0, min(x1, width-1))
                                x2 = max(0, min(x2, width-1))
                                y1 = max(0, min(y1, height-1))
                                y2 = max(0, min(y2, height-1))
                                
                                # バウンディングボックス内の深度値を取得
                                depth_roi = self.d455_depth[y1:y2, x1:x2]
                                
                                # 有効な深度値のみをフィルタリング
                                valid_depths = depth_roi[(depth_roi > 0) & (~np.isnan(depth_roi))]
                                
                                if len(valid_depths) > 0:
                                    # 有効な深度値を昇順にソート
                                    sorted_depths = np.sort(valid_depths)
                                    
                                    # 最も近い値から30%の位置の深度を選択
                                    percentile_idx = int(len(sorted_depths) * 0.3)
                                    depth_value = sorted_depths[percentile_idx]
                                    
                                    # メートルに変換
                                    depth = depth_value / 1000.0
                                    self.get_logger().info(f'Person ID {person_id}: depth={depth:.2f}m (from ROI)')
                                    
                                    # 極端な値をフィルタリング
                                    if depth <= 0.1 or depth > 10.0:
                                        self.get_logger().warn(f'Extreme depth value: {depth}m at percentile 30%')
                                        continue
                                else:
                                    # 有効な深度値がない場合は中心点を使用
                                    depth = self.d455_depth[center_v, center_u] / 1000.0
                                    if np.isnan(depth) or depth <= 0.0:
                                        self.get_logger().warn(f'Invalid depth value at center: {depth}')
                                        continue
                                        
                                # 深度デバッグ情報
                                self.get_logger().info(f'Person ID {person_id}: depth={depth:.2f}m (from ROI)')
                                
                        except IndexError as ie:
                                self.get_logger().warn(f'Depth IndexError: {ie}')
                                continue
                        except Exception as e:
                                self.get_logger().warn(f'Depth error: {e}')
                                continue
                        
                        # 人物情報をリストに追加
                        person_info = {
                            'id': person_id,
                            'box': xyxy,
                            'center': (center_u, center_v),
                            'depth': depth
                        }
                        detected_persons.append(person_info)
            
            # 追跡対象が見つからなかった場合のカウンター処理
            if not person_detected or len(detected_persons) == 0:
                self.tracking_lost_counter += 1
                if self.tracking_lost_counter >= self.max_tracking_lost:
                    # 長時間見つからなければターゲットIDをリセット
                    self.target_person_id = None
                    self.get_logger().info('Target lost. Resetting tracking.')
                
                # 人物が見つからない場合はロボットを停止
                self.publish_cmd(0.0, 0.0)
                return
            
            # 検出した人物をソート（距離が近い順）
            detected_persons.sort(key=lambda p: p['depth'])
            
            # ターゲット選択ロジック
            target_person = None
            
            if self.target_person_id is not None:
                # 既存のターゲットIDを持つ人物を探す
                for person in detected_persons:
                    if person['id'] == self.target_person_id:
                        target_person = person
                        self.tracking_lost_counter = 0  # カウンターリセット
                        break
            
            # ターゲットが見つからないか未設定の場合、最も近い人物を選択
            if target_person is None:
                if len(detected_persons) > 0:
                    target_person = detected_persons[0]  # 最も近い人物
                    self.target_person_id = target_person['id']
                    self.tracking_lost_counter = 0
                    self.get_logger().info(f'New target selected: ID {self.target_person_id}')
            
            # ターゲットに対する制御コマンドを計算
            if target_person:
                center_u, center_v = target_person['center']
                Z = target_person['depth']  # メートル単位
                
                # バウンディングボックスに矩形を描画（追跡対象は赤色）
                x1, y1, x2, y2 = map(int, target_person['box'])
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated_image, f"ID: {target_person['id']} ({Z:.2f}m)", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # カメラ座標系への変換self.future.result().final_result
                X = (center_u - self.cx) * Z / self.fx
                
                # 制御コマンドの計算
                angular_z = -self.angular_gain * X
                distance_error = Z - self.follow_distance
                linear_x = self.linear_gain * distance_error

                # self.publish_cmd(linear_x,angular_z)

         except StopIteration:
            self.get_logger().warn('No detection results.')
            self.tracking_lost_counter += 1
            if self.tracking_lost_counter >= self.max_tracking_lost:
                self.target_person_id = None
            self.publish_cmd(0.0, 0.0)
         except Exception as e:
            self.get_logger().error(f'Error in timer callback: {str(e)}')
            self.publish_cmd(0.0, 0.0)
    
    def publish_cmd(self, linear_x, angular_z):
        twist = Twist()
        max_speed = 0.6
        max_angular_speed = 1.2
        if abs(linear_x) >= max_speed:
            if linear_x > 0:
                linear_x = max_speed
            else:
                linear_x = -max_speed
        if abs(angular_z) >= max_angular_speed:
            if angular_z > 0:
                angular_z = max_angular_speed
            else:
                angular_z = -max_angular_speed

        twist.linear.x = linear_x
        twist.angular.z = angular_z
        self.cmd_pub.publish(twist)
             

    def d455_callback(self, msg):
        self.d455_image = self.bridge.imgmsg_to_cv2(msg.rgb, "bgr8")
        self.d455_depth = self.bridge.imgmsg_to_cv2(msg.depth, "passthrough")
        self.d455_flag = True
    
    def d435_callback(self, msg):
        self.d435_image = self.bridge.imgmsg_to_cv2(msg.rgb, "bgr8")
        self.d435_depth = self.bridge.imgmsg_to_cv2(msg.depth, "passthrough")
        self.d435_flag = True



    #点群処理、ナビゲーション、Graspingをどのノードでやるか？

    
    def move_lift(self,lift):
        self.req.hight = int(lift)
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future, timeout_sec=100.0)
        if self.future.done():
            if self.future.result() is not None:
                response = self.future.result()
                print(response.success)
                return response.success
        response = False
        return response
    
    def pub_position(self,position):
        output = SetPosition()
        output.id = position[0]
        output.position = position[1]
        self.dynamixel_publisher.publish(output)

    def publish_position_four(self,position):
        output = SetPositionFourMotor()
        output.id_1 = 11
        output.id_2 = 12
        output.id_3 = 13
        output.id_4 = 14
        output.position_1 = position[0]
        output.position_2 = position[1]
        output.position_3 = position[2]
        output.position_4 = position[3]
        self.publisher_position.publish(output)
    
    def speak(self, text):
        self.tts_flag = False
        goal = TTS.Goal()
        goal.text = text
        if not self.tts_client.wait_for_server(timeout_sec=1.0):
            print("error")
            return

        self.future = self.tts_client.send_goal_async(goal, feedback_callback=self.tts_feedback_callback)
        self.future.add_done_callback(self.tts_response_callback)
        print(self.tts_flag)
        while not self.tts_flag:
            rclpy.spin_once(self)
            print(self.tts_flag)
        else:
            print(self.tts_flag)
            return
        """
        rclpy.spin_until_future_complete(self.tts_client, self.future)

        # 結果が得られたら sequence を取得して返す
        if self.result_future is not None and self.result_future.done():
            result = self.result_future.result().result
            self.node.get_logger().info('Result: {0}'.format(result.sequence))
            return result.sequence
        else:
            self.node.get_logger().info('No result received.')
            return None
        """
    #フィードバックを受け取る関数
    def tts_feedback_callback(self, feedback):
        print("Result callback triggered")

    # レスポンスの受信時に呼ばれる
    def tts_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
           return

        # 結果の取得

        self.result_future = goal_handle.get_result_async()
        self.result_future.add_done_callback(self.result_callback)

    # 結果の受信時に呼ばれる
    def result_callback(self, future):
        result = future.result().result
        print(result)
        self.tts_flag  = True
        return
    def go_to_abs(self,msg):
        nav = BasicNavigator()

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = nav.get_clock().now().to_msg()
        x = msg[0]
        y = msg[1]
        yaw = msg[2]
        goal_pose.pose.position.x = x
        goal_pose.pose.position.y = y
        goal_pose.pose.position.z = 0.0 
        q = quaternion_from_euler(0, 0, yaw)
        goal_pose.pose.orientation.x = q[0]
        goal_pose.pose.orientation.y = q[1]
        goal_pose.pose.orientation.z = q[2]
        goal_pose.pose.orientation.w = q[3]

        nav.goToPose(goal_pose)
        i = 0
        now = time.time()
        time_delta = now + 15
        n = 0
        while not nav.isTaskComplete():
                i += 1
                feedback = nav.getFeedback()
                if feedback and i % 10 == 0:
                    print(
                        f'Estimated time of arrival to destination is: \
                        {Duration.from_msg(feedback.estimated_time_remaining).nanoseconds / 1e9}'
                    )
                if time_delta < time.time():
                    if n < 10:
                        n += 1
                        print("time out")
                        nav.cancelTask()
                        time.sleep(1)
                        nav.goToPose(goal_pose)
                        time_delta = time.time() + 30
                        print("retry")
                        continue
                    else:
                        nav.cancelTask()
                        print("time out")
                        break
        if nav.isTaskComplete():
            result = nav.getResult()
            if result == TaskResult.SUCCEEDED:
                print("Goal succeeded!")
                return True
            elif result == TaskResult.CANCELED:
                print("Goal canceled!")
                return False
            elif result == TaskResult.FAILED:
                print("Goal failed!")
                return False


    def person_left_right(self):
        self.d455_flag = False
        while not self.d455_flag:
            rclpy.spin_once(self)
        results : List[Results] = self.model(self.d455_image,classes=[0])
        annotated_image = results[0].plot()

        #人を検知したかどうか
        if len(results) == 0:
            return False
        distance_list = [0] * len(results)
        for i in range (len(results)):
            # 各結果のボックスとキーポイントを取得
            boxes = results[i].boxes.cpu().numpy()
            keypoints = results[i].keypoints
            xys = keypoints.xy[0].tolist()

            distance_list[i] = self.d455_depth[int(xys[i][0]),int(xys[i][1])]
        most_near = distance_list.index(min(distance_list))

            

        keypoints = results[most_near].keypoints
        keypoints_xys = keypoints.xy[0].tolist()

        nose_y = keypoints_xys[0][1]
        left_eye_y = keypoints_xys[1][1]
        right_eye_y = keypoints_xys[2][1]
        eye_center = (left_eye_y+ right_eye_y) / 2
        left_hand_y = keypoints_xys[9][1]
        right_hand_y = keypoints_xys[10][1]

        #右か左かを検出
        if abs(eye_center - left_hand_y) < abs(eye_center - right_hand_y):
            print(eye_center, left_hand_y, right_hand_y)
            response = "right"
        else:
            response = "left"

        return response
    
    def pick_kimeuti(self):
        self.speak("I will pick up a bag")
        self.pub_position(self.gripper_open)
        self.pub_position(self.wrist_pick)
        time.sleep(2)
        self.publish_position_four(self.set_pick)
        time.sleep(5)
        self.pub_position(self.gripper_close)
        time.sleep(2)
        self.publish_position_four(self.set_nuetral_2)
        self.pub_position(self.wrist_nuetral)
        time.sleep(2)
        self.speak("I picked up a bag")
    
    #3,2,1でbagを渡す
    def hand_over(self):
        self.speak("3")
        time.sleep(1)
        self.speak("2")
        time.sleep(1)
        self.speak("1")
        time.sleep(1)
        self.speak("Please take it")
        self.pub_position(self.gripper_open)

    
    def avoid_bag(self):
        self.move_lift(100)
        self.pub_position(self.wrist_nuetral)

    def run(self):
        person_detect = self.person_hand_detection()
        bag_detect = self.bag_placing(person_detect,"d455")

    def carry_run(self):
        #人の腕の方向をサービスで検出
        self.speak("I detecting a person's hand")
        left_or_right = self.person_left_right()
        print(left_or_right)

        self.speak(f"The person point out the bag on the {left_or_right} from my point of view")
        
        self.speak("I will follow person")

        #追従
        self.follow_flag = True
        self.vosk_request()
        while self.listen is False:
            rclpy.spin_once(self)

        self.speak("I will go back")
        self.go_to_abs([0.0, 0.0, 0.0])
        self.speak("I completed the task")


    



def main(args=None):
    rclpy.init(args=args)

    task_manager = TaskManager() 


    task_manager.follow_flag = True
    # Call the person_hand_detection method
    # success = task_manager.person_left_right()
    # print(success)
    #task_manager.go_to_abs([0.0, 0.0, 0.0])
    #task_manager.pick_kimeuti()
#     task_manager.carry_run()
    rclpy.spin(task_manager)
# # 
    # Destroy the node explicitly
    task_manager.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()