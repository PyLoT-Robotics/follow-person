#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from follow_person_msgs.action import FollowPerson
from follow_person_msgs.srv import FollowControl
from rclpy.task import Future

class FollowPersonClient(Node):
    def __init__(self):
        super().__init__('follow_person_client')
        
        # Create callback group for concurrent processing
        self.callback_group = ReentrantCallbackGroup()
        
        # Action client
        self._action_client = ActionClient(
            self,
            FollowPerson,
            'follow_person',
            callback_group=self.callback_group
        )
        
        # Service client
        self._control_client = self.create_client(
            FollowControl,
            'follow_control',
            callback_group=self.callback_group
        )

    async def send_follow_goal(self, target_distance: float = 1.2, max_speed: float = 0.5) -> Future:
        """Send a follow person action goal."""
        self.get_logger().info('Waiting for action server...')
        self._action_client.wait_for_server()
        
        goal_msg = FollowPerson.Goal()
        goal_msg.target_distance = target_distance
        goal_msg.max_speed = max_speed
        
        self.get_logger().info('Sending goal...')
        
        send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self._feedback_callback
        )
        
        response = await send_goal_future
        if not response.accepted:
            self.get_logger().error('Goal rejected')
            return None
            
        self.get_logger().info('Goal accepted')
        return response.get_result_async()

    async def control_following(self, command: str) -> bool:
        """Send a control command via service."""
        self.get_logger().info('Waiting for service...')
        self._control_client.wait_for_service()
        
        request = FollowControl.Request()
        request.command = command
        
        response = await self._control_client.call_async(request)
        
        if response.success:
            self.get_logger().info(f'Command "{command}" succeeded: {response.message}')
        else:
            self.get_logger().error(f'Command "{command}" failed: {response.message}')
            
        return response.success

    def _feedback_callback(self, feedback_msg):
        """Handle action feedback."""
        feedback = feedback_msg.feedback
        self.get_logger().info(
            f'State: {feedback.current_state}, '
            f'Distance: {feedback.current_distance_to_person:.2f}m, '
            f'Speed: {feedback.current_speed:.2f}m/s'
        )

async def main(args=None):
    rclpy.init(args=args)
    client = FollowPersonClient()
    
    try:
        # Example: Start following using service
        await client.control_following('start')
        
        # Example: Send action goal
        result_future = await client.send_follow_goal(
            target_distance=1.5,
            max_speed=0.6
        )
        
        if result_future:
            result = await result_future
            client.get_logger().info(
                f'Action completed: {result.result.message}, '
                f'Distance traveled: {result.result.total_distance_traveled:.2f}m'
            )
        
        # Example: Pause following
        await client.control_following('pause')
        
        # Wait a bit
        await rclpy.time.sleep(2.0)
        
        # Example: Resume following
        await client.control_following('resume')
        
        # Wait a bit
        await rclpy.time.sleep(5.0)
        
        # Example: Stop following
        await client.control_following('stop')
        
    except KeyboardInterrupt:
        pass
    finally:
        client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
