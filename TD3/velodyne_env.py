import math
import os
import random
import subprocess
import time
from os import path

import numpy as np
import rospy
#点云格式（x,y,z）
import sensor_msgs.point_cloud2 as pc2
# 设置或获取仿真环境中模型（机器人或物体）的状态,model_name,pose,twist
from gazebo_msgs.msg import ModelState
#几何消息类型，表示线速度和角速度。liner and angular
from geometry_msgs.msg import Twist
# Odometry 是 ROS 中的标准消息类型，用于表示机器人的里程计信息。pose：机器人的位置和方向（包括位置和四元数方向）。twist：机器人的线速度和角速度。
from nav_msgs.msg import Odometry
# PointCloud2 是 ROS 中的标准消息类型，用于表示三维点云数据。
from sensor_msgs.msg import PointCloud2
# 四元数，用于处理方向和旋转。
from squaternion import Quaternion
Empty 是 ROS 中一种标准服务类型，不需要请求或返回数据。用于触发某种操作，暂停或恢复Gazebo仿真：重置仿真世界：
from std_srvs.srv import Empty
#Marker 是 ROS 中的消息类型，用于在RViz中可视化单个标记。type,pose,scale,color
from visualization_msgs.msg import Marker
# MarkerArray 是一个包含多个Marker的消息类型
from visualization_msgs.msg import MarkerArray

GOAL_REACHED_DIST = 0.3
COLLISION_DIST = 0.35
TIME_DELTA = 0.1


# Check if the random goal position is located on an obstacle and do not accept it if it is
#仿真空间为9*9，坐标原点位于中间点
def check_pos(x, y):
    goal_ok = True

    if -3.8 > x > -6.2 and 6.2 > y > 3.8:
        goal_ok = False

    if -1.3 > x > -2.7 and 4.7 > y > -0.2:
        goal_ok = False

    if -0.3 > x > -4.2 and 2.7 > y > 1.3:
        goal_ok = False

    if -0.8 > x > -4.2 and -2.3 > y > -4.2:
        goal_ok = False

    if -1.3 > x > -3.7 and -0.8 > y > -2.7:
        goal_ok = False

    if 4.2 > x > 0.8 and -1.8 > y > -3.2:
        goal_ok = False

    if 4 > x > 2.5 and 0.7 > y > -3.2:
        goal_ok = False

    if 6.2 > x > 3.8 and -3.3 > y > -4.2:
        goal_ok = False

    if 4.2 > x > 1.3 and 3.7 > y > 1.5:
        goal_ok = False

    if -3.0 > x > -7.2 and 0.5 > y > -1.5:
        goal_ok = False

    if x > 4.5 or x < -4.5 or y > 4.5 or y < -4.5:
        goal_ok = False

    return goal_ok


class GazeboEnv:
    """Superclass for all Gazebo environments."""

    def __init__(self, launchfile, environment_dim):
        self.environment_dim = environment_dim # 环境维度，用于描述激光雷达数据的离散分布。
        self.odom_x = 0 #机器人当前位置 x =0
        self.odom_y = 0

        self.goal_x = 1 #目标位置
        self.goal_y = 0.0

        self.upper = 5.0 # 随机生成目标点时，x 和 y 的最大边界。
        self.lower = -5.0
        # 用于存储激光雷达数据，初始值为全 10 的数组，表示激光测距的最大距离。
        self.velodyne_data = np.ones(self.environment_dim) * 10 
        self.last_odom = None

        # 初始化机器人模型的状态信息
        self.set_self_state = ModelState()  # ROS 消息类型，描述机器人模型的状态。
        self.set_self_state.model_name = "r1"  # 模型名称为 "r1"（代表机器人 ID）。
        self.set_self_state.pose.position.x = 0.0  # 初始位置 x 坐标。
        self.set_self_state.pose.position.y = 0.0  # 初始位置 y 坐标。
        self.set_self_state.pose.position.z = 0.0  # 初始位置 z 坐标（高度）。
        self.set_self_state.pose.orientation.x = 0.0  # 四元数表示的方向，x 分量。
        self.set_self_state.pose.orientation.y = 0.0  # 四元数方向的 y 分量。
        self.set_self_state.pose.orientation.z = 0.0  # 四元数方向的 z 分量。
        self.set_self_state.pose.orientation.w = 1.0  # 四元数方向的 w 分量（正向）。
        
        # 根据激光雷达数据的分辨率，定义激光雷达扫描范围的离散区间
        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / self.environment_dim]]
        for m in range(self.environment_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / self.environment_dim]
            )
        self.gaps[-1][-1] += 0.03

        
        # 启动 ROS 核心服务（roscore），以指定的端口运行。
        port = "11311"
        subprocess.Popen(["roscore", "-p", port])

        print("Roscore launched!")

        # Launch the simulation with the given launchfile name
        rospy.init_node("gym", anonymous=True)
        if launchfile.startswith("/"):
            fullpath = launchfile
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", launchfile)
        if not path.exists(fullpath):
            raise IOError("File " + fullpath + " does not exist")

        # 使用 `roslaunch` 启动 Gazebo 仿真环境，并加载指定的 launch 文件。
        subprocess.Popen(["roslaunch", "-p", port, fullpath])
        print("Gazebo launched!")

        # Set up the ROS publishers and subscribers
        # 发布器，用于发布机器人运动控制命令（线速度和角速度）。
        self.vel_pub = rospy.Publisher("/r1/cmd_vel", Twist, queue_size=1)
        
        # 发布器，用于设置 Gazebo 中模型的状态（例如位置和方向）。
        self.set_state = rospy.Publisher(
            "gazebo/set_model_state", ModelState, queue_size=10
        )
        
        # 服务代理，用于解除 Gazebo 的暂停状态。
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        # 服务代理，用于暂停 Gazebo 的物理模拟。
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        # 服务代理，用于重置 Gazebo 仿真环境。
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        # 发布器，用于在 RViz 中可视化目标点。
        self.publisher = rospy.Publisher("goal_point", MarkerArray, queue_size=3)
        # 发布器，用于在 RViz 中可视化线速度。
        self.publisher2 = rospy.Publisher("linear_velocity", MarkerArray, queue_size=1)
        # 发布器，用于在 RViz 中可视化角速度。
        self.publisher3 = rospy.Publisher("angular_velocity", MarkerArray, queue_size=1)
         
         #订阅器，订阅 `/velodyne_points` 话题，用于接收激光雷达点云数据。
        # 回调函数 `self.velodyne_callback` 用于处理点云数据。
        self.velodyne = rospy.Subscriber(
            "/velodyne_points", PointCloud2, self.velodyne_callback, queue_size=1
        )
        self.odom = rospy.Subscriber(
            "/r1/odom", Odometry, self.odom_callback, queue_size=1
        )

    # Read velodyne pointcloud and turn it into distance data, then select the minimum value for each angle
    # range as state representation
    def velodyne_callback(self, v):
        # 回调函数，用于处理订阅到的 `/velodyne_points` 话题消息（类型为 PointCloud2）。
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        # 使用 `pc2.read_points` 将点云数据从 `PointCloud2` 格式解码为点的列表。
        # 参数：
        # - `v`：订阅的 PointCloud2 消息。
        # - `skip_nans`：是否跳过 NaN 值的点。
        # - `field_names`：指定提取的字段，这里提取 `x`, `y`, `z` 三维坐标。
        self.velodyne_data = np.ones(self.environment_dim) * 10
        # 初始化激光雷达扫描的状态，假定所有方向上的障碍物距离为最大值 10。
        for i in range(len(data)):
            if data[i][2] > -0.2:
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)
                # 计算点到传感器的三维欧几里得距离。
                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                        # 检查点所在的角度范围是否属于当前扫描区间。
                        self.velodyne_data[j] = min(self.velodyne_data[j], dist)
                        # 更新该角度范围内的最小距离值。
                        break

    def odom_callback(self, od_data):
        # 回调函数，用于处理订阅到的 `/r1/odom` 话题（类型为 Odometry）。
        self.last_odom = od_data
        # 将最新的里程计信息保存到类的 `last_odom` 属性。
    
    
    # Perform an action and read a new state
    def step(self, action):
        # 该函数执行一个动作并获取新的环境状态。
        target = False

        # Publish the robot action
        # 创建 `Twist` 消息对象，用于发送机器人运动指令。
        vel_cmd = Twist()
        
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        self.vel_pub.publish(vel_cmd)
        
        self.publish_markers(action)
        # 在 RViz 中显示当前的动作信息（如线速度和角速度）。
        
        # 等待 Gazebo 的解除暂停服务可用。
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
            # 调用服务解除 Gazebo 的暂停状态。
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        # propagate state for TIME_DELTA seconds
        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        # 等待 Gazebo 的暂停服务可用。
        try:
            pass
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        # read velodyne laser state
        # 检查激光雷达状态，判断是否发生碰撞，并获取最小距离。
        done, collision, min_laser = self.observe_collision(self.velodyne_data)
        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]
        # 将激光雷达状态包装为列表。
        
        # Calculate robot heading from odometry data
        # 从里程计数据中提取机器人 x 方向位置。
        self.odom_x = self.last_odom.pose.pose.position.x
        self.odom_y = self.last_odom.pose.pose.position.y
        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        # 使用 `Quaternion` 将四元数转换为方向角（欧拉角）。
        # 转换为欧拉角（单位为弧度）。->提取其航向
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)

        # Calculate distance to the goal from the robot
        # 计算机器人当前位置到目标点的欧几里得距离。
        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        # Calculate the relative angle between the robots heading and heading toward the goal
        # 计算机器人到目标点的向量（skew_x, skew_y）。
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        # 计算夹角 与x的
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        # 根据目标点在 x-y 平面的位置调整角度的符号。
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        # 计算机器人航向与目标方向之间的相对角度 theta。
        theta = beta - angle
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta
        # 将 theta 归一化到 [-pi, pi] 范围。
        
        
        # Detect if the goal has been reached and give a large positive reward
        if distance < GOAL_REACHED_DIST:
            target = True
            done = True

        robot_state = [distance, theta, action[0], action[1]]
        # 将机器人状态编码为列表（距离、相对角度、线速度、角速度）。
        
        state = np.append(laser_state, robot_state)
        # 将激光雷达状态与机器人状态组合，形成完整的状态向量。
        
        reward = self.get_reward(target, collision, action, min_laser)
        # 根据目标是否到达、是否碰撞、动作等计算奖励。
        
        # 返回新的状态、奖励、是否结束标志和目标到达标志。
        return state, reward, done, target

    def reset(self):
        

        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()

        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        object_state = self.set_self_state

        x = 0
        y = 0
        position_ok = False
        while not position_ok:
            x = np.random.uniform(-4.5, 4.5)
            y = np.random.uniform(-4.5, 4.5)
            position_ok = check_pos(x, y)
        object_state.pose.position.x = x
        object_state.pose.position.y = y
        # object_state.pose.position.z = 0.
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w
        self.set_state.publish(object_state)

        self.odom_x = object_state.pose.position.x
        self.odom_y = object_state.pose.position.y

        # set a random goal in empty space in environment
        self.change_goal()
        # randomly scatter boxes in the environment
        self.random_box()
        self.publish_markers([0.0, 0.0])
        # 在 RViz 中发布目标点和机器人状态标记。
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]

        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y

        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))

        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle

        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        robot_state = [distance, theta, 0.0, 0.0]
        state = np.append(laser_state, robot_state)
        return state

    def change_goal(self):
        # Place a new goal and check if its location is not on one of the obstacles
        if self.upper < 10:
            self.upper += 0.004
        if self.lower > -10:
            self.lower -= 0.004

        goal_ok = False

        while not goal_ok:
            self.goal_x = self.odom_x + random.uniform(self.upper, self.lower)
            self.goal_y = self.odom_y + random.uniform(self.upper, self.lower)
            goal_ok = check_pos(self.goal_x, self.goal_y)

    def random_box(self):
        # Randomly change the location of the boxes in the environment on each reset to randomize the training
        # environment
        for i in range(4):
            name = "cardboard_box_" + str(i)

            x = 0
            y = 0
            box_ok = False
            while not box_ok:
                x = np.random.uniform(-6, 6)
                y = np.random.uniform(-6, 6)
                box_ok = check_pos(x, y)
                distance_to_robot = np.linalg.norm([x - self.odom_x, y - self.odom_y])
                distance_to_goal = np.linalg.norm([x - self.goal_x, y - self.goal_y])
                if distance_to_robot < 1.5 or distance_to_goal < 1.5:
                    box_ok = False
            box_state = ModelState()
            box_state.model_name = name
            box_state.pose.position.x = x
            box_state.pose.position.y = y
            box_state.pose.position.z = 0.0
            box_state.pose.orientation.x = 0.0
            box_state.pose.orientation.y = 0.0
            box_state.pose.orientation.z = 0.0
            box_state.pose.orientation.w = 1.0
            self.set_state.publish(box_state)

    def publish_markers(self, action):
        # Publish visual data in Rviz
        # 创建目标点的 MarkerArray 对象。
        markerArray = MarkerArray()
        # 创建单个目标点的 Marker 对象。
        marker = Marker()
        # 创建单个目标点的 Marker 对象。
        marker.header.frame_id = "odom"
        # 将目标点显示为圆柱体。
        marker.type = marker.CYLINDER
        # 动作类型：新增标记。
        marker.action = marker.ADD
        # 设置标记的尺寸，圆柱体的直径和高度。
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        # 设置标记颜色：绿色，不透明。
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = 0
        
        # 将目标点标记添加到 MarkerArray。
        markerArray.markers.append(marker)
        # 发布目标点标记到 `goal_point` 话题。
        self.publisher.publish(markerArray)

        
        # 以下逻辑用于显示动作（线速度和角速度）的可视化标记。
        markerArray2 = MarkerArray()
        marker2 = Marker()
        marker2.header.frame_id = "odom"
        marker2.type = marker.CUBE
        marker2.action = marker.ADD
        marker2.scale.x = abs(action[0])
        marker2.scale.y = 0.1
        marker2.scale.z = 0.01
        marker2.color.a = 1.0
        marker2.color.r = 1.0
        marker2.color.g = 0.0
        marker2.color.b = 0.0
        marker2.pose.orientation.w = 1.0
        marker2.pose.position.x = 5
        marker2.pose.position.y = 0
        marker2.pose.position.z = 0

        markerArray2.markers.append(marker2)
        self.publisher2.publish(markerArray2)
        # 显示线速度标记，大小与动作的线速度大小成比例。
        markerArray3 = MarkerArray()
        marker3 = Marker()
        marker3.header.frame_id = "odom"
        marker3.type = marker.CUBE
        marker3.action = marker.ADD
        marker3.scale.x = abs(action[1])
        marker3.scale.y = 0.1
        marker3.scale.z = 0.01
        marker3.color.a = 1.0
        marker3.color.r = 1.0
        marker3.color.g = 0.0
        marker3.color.b = 0.0
        marker3.pose.orientation.w = 1.0
        marker3.pose.position.x = 5
        marker3.pose.position.y = 0.2
        marker3.pose.position.z = 0

        markerArray3.markers.append(marker3)
        self.publisher3.publish(markerArray3)
        # 显示角速度标记，大小与动作的角速度大小成比例。
        
        
    @staticmethod
    def observe_collision(laser_data):
        # Detect a collision from laser data
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            return True, True, min_laser
        return False, False, min_laser

    @staticmethod
    def get_reward(target, collision, action, min_laser):
        # 静态方法：根据目标到达情况、碰撞、动作和最小距离计算奖励值。
        if target:
            return 100.0
        elif collision:
            return -100.0
        else:
            # 定义一个奖励函数，对距离小于1的值进行奖励衰减。
            r3 = lambda x: 1 - x if x < 1 else 0.0
            # 综合线速度、角速度和激光雷达最小距离计算奖励：
            # - `action[0] / 2`：鼓励较高的线速度。
            # - `-abs(action[1]) / 2`：惩罚较大的角速度（急转弯）。
            # - `-r3(min_laser) / 2`：惩罚靠近障碍物的行为。
            return action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2
