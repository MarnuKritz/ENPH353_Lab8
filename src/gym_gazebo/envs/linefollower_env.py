import cv2
import rclpy
import time
import gym_gazebo
import gymnasium as gym
import numpy as np
from cv_bridge import CvBridge
from gym_gazebo.core.gazebo_env import GazeboEnv
from rclpy.node import Node as RosNode
from sensor_msgs.msg import Image as RosImage
from geometry_msgs.msg import Twist as RosTwist
import os

NUM_OBS_STATES = 3
NUM_ACTIONS = 3

class LineFollowerEnv(GazeboEnv):
    def __init__(self):
        # Intializing ROS2
        rclpy.init(args=None)
        
        super(LineFollowerEnv, self).__init__(launch_pkg='linefollow_ros',
                                    launch_file='my_agent.launch.xml',
                                    world_name='monza_gym')

        # Setting up ROS2 node
        self.ros_node = RosNode('line_follower_RL')

        # Initializing variables
        self.is_resetting = False
        self.latest_obs = None
        self.new_obs_event = False

        # Setting the observation and action spaces
        self.observation_space = gym.spaces.Discrete(NUM_OBS_STATES)
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)

        # ROS2 topics
        observation_topic = "/camera/image_raw"
        velocity_topic = "/cmd_vel"

        # Publishing and subscribing to ROS2 topics
        self.pub_cmd_vel_msg = self.ros_node.create_publisher(
            RosTwist, velocity_topic, 10)
            
        self.sub_obs = self.ros_node.create_subscription(
            RosImage, observation_topic, self.callback_obs_feed_, 10)


    def callback_obs_feed_(self, msg: RosImage):
        '''
        @brief observation callback resets the new_obs_event flag
        @param msg: Image from Gazebo sim
        '''
        # Takes in observation, returns a flag an observation is recieved
        if not self.is_resetting:
            self.latest_obs = msg
            self.new_obs_event = True


    ## MAIN FUNCTIONS:
    def reset(self, seed=None, options=None):
        '''
        @brief resets the simulation
        '''
        self.is_resetting = True
        print("Resetting")

        # Zero out velocities
        vel_cmd = RosTwist()
        vel_cmd.linear.x = 0.0
        vel_cmd.angular.z = 0.0
        self.pub_cmd_vel_msg.publish(vel_cmd)

        # Reset position and joints
        self._reset_agents()

        # Unpause the sim
        self._pause_sim(False)

        # Allow some time to settle
        time.sleep(0.1) # Wait 100 ms
        
        self.is_resetting = False

        # Wait for next observation to come in before kickstarting the data loop
        self.new_obs_event = False
        while not self.new_obs_event:
            # Telling executor (node) to spin
            rclpy.spin_once(self.ros_node, timeout_sec=0.01)

        # # Pause the sim after the first observation comes in
        self._pause_sim(True)
        
        state, _ = self.process_obs(self.latest_obs)

        print("Done resetting")
        return state, {}
    
    
    def step(self, action: int):
        '''
        @brief takes a step in the simulation
        @param action
        @retval state
        @retval reward
        @retval terminated
        @retval truncated
        '''
        # Process the action
        vel_cmd = self.process_action(action)

        # Unpause the sim
        self._pause_sim(False)

        # Moving the agent
        self.pub_cmd_vel_msg.publish(vel_cmd)

        # Leave running until next observation comes in
        self.new_obs_event = False
        while not self.new_obs_event:
            rclpy.spin_once(self.ros_node, timeout_sec=0.01)

        # Pause the simulation after the next observation comes in
        self._pause_sim(True)

        state, terminated = self.process_obs(self.latest_obs)
        
        reward = self.process_reward(action, state, terminated)

        truncated = False # Used if we ran maximum number of steps.
        return state, reward, terminated, truncated, {}

    
    ## HELPER FUNCTIONS
    def process_action(self, action: int):
        '''
        @brief Translate the action the robot is taking into a Twist message
        @param action: an integer where 0 is left, 1 is forward and 2 is right 
        @retval vel_cmd: Twist message equivalent of the action
        '''
        vel_cmd = RosTwist() # Initialize ROS command

        # When paction is set to "turn left", drive slightly forward and turn left
        if(action == 0):
            vel_cmd.linear.x = 0.4
            vel_cmd.angular.z = 2.5

        # When action is set to "go straight", drive forward
        elif(action == 1):
            vel_cmd.linear.x = 2.0
            vel_cmd.angular.z = 0.0

        # When action is set to "turn right", drive slightly forward and turn right
        elif(action == 2):
            vel_cmd.linear.x = 0.4
            vel_cmd.angular.z = -2.5
        
        return vel_cmd


    def process_reward(self, action, state, terminated):
        '''
        @brief depending on the action, state and terminated values assign
               a reward
        @retval reward: the reward for the current step
        '''
        reward = 0

        # Greatly discourage the robot from losing the line
        if(terminated): reward -= 100

        # Reward the robot for responding appropriately to needing to turn left. Lesser punishment for driving straight, as
        # it might be a viable option, strong discouragement for turning right when needing to turn left
        if(state == 0): reward += 5

        # Reward robot for responding appropriately to going straight. Equal punishment for unwaranted left or right turn
        elif(state == 1): reward += 5

        # Reward robot for responding appropriately to needing to turn right. Lesser punishment for driving straight, moreso
        # for turning left when it needs to go right
        elif(state == 2):reward += 5

        return reward


    def process_obs(self, obs: RosImage):
        '''
        @brief Identifies the line in the image and populates the state array
               based on the line position (i.e. state=0 line is left, 
               state=1 line is in the center)
        @param obs raw image from Gazebo sim
        @retval state: the line position as a value: 
            -1 (no line), 0 (left), 1 (center), 2 (right)
        @retval terminated: the termination condition occured - the line is no 
                    longer in the field of the camera
        '''

        # Recieve frame from ROS channel
        cv_image = CvBridge().imgmsg_to_cv2(obs, desired_encoding='bgr8')

        # Default as line is centered
        state = 1
        terminated = False

        height, width, channel = cv_image.shape
        left_border = width/3
        right_border = (width*2)/3

        ## @brief Reduces area of interest to bottom 30% of screen.
        startRead = int(height*0.90)
        bottom = cv_image[startRead:,:]

        ## @brief Converts area of interest to HSV and extracts saturation channel.
        hsv = cv2.cvtColor(bottom, cv2.COLOR_BGR2HSV)
        S = hsv[:,:,1]

        ## @brief Creates a mask to differentiate high-saturation pixels
        mask = S > 120

        ## @brief Place xy coordinates of mask pixels and in an array.

        y,x = np.where(mask)
        print("Detected pixels:", len(x))

        ## @brief Calculates error of robot centering based on central saturation pixel vs central image pixel, uploads correction turn speed to robot
        if len(x) > 0:
            centre_x = int(np.mean(x))

            if(centre_x < left_border): state=0

            elif(centre_x > right_border): state=2

            else: state=1

        else:
            state = -1
            terminated = True

        return state, terminated


    # Shutting down ROS2
    def user_close(self):
        self.ros_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()