import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from velodyne_env import GazeboEnv


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800) # 第一隐藏层
        self.layer_2 = nn.Linear(800, 600)  # 第二隐藏层
        self.layer_3 = nn.Linear(600, action_dim) # 输出层
        self.tanh = nn.Tanh() # Tanh 激活函数

    def forward(self, s):
        s = F.relu(self.layer_1(s)) # 第一隐藏层 + ReLU
        s = F.relu(self.layer_2(s)) # 第二隐藏层 + ReLU
        a = self.tanh(self.layer_3(s)) # 输出层 + Tanh
        return a


# TD3 network
class TD3(object):
    def __init__(self, state_dim, action_dim):
        # Initialize the Actor network
        self.actor = Actor(state_dim, action_dim).to(device)

    def get_action(self, state):
        # Function to get the action from the actor
        state = torch.Tensor(state.reshape(1, -1)).to(device) # 将状态转换为 Tensor
        return self.actor(state).cpu().data.numpy().flatten() # 输出动作

    def load(self, filename, directory):
        # Function to load network parameters
        # 使用 torch.load 加载已保存的模型权重。
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )


# Set the parameters for the implementation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu
seed = 0  # Random seed number
max_ep = 500  # maximum number of steps per episode# 每个回合的最大步数
file_name = "TD3_velodyne"  # name of the file to load the policy from


# Create the testing environment
environment_dim = 20 # 激光雷达状态维度
robot_dim = 4 # 机器人状态维度（例如距离、角度、速度等）
env = GazeboEnv("multi_robot_scenario.launch", environment_dim) # 创建 Gazebo 环境
time.sleep(5)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = environment_dim + robot_dim  # 状态维度：激光雷达数据 + 机器人状态
action_dim = 2

# Create the network
network = TD3(state_dim, action_dim)
try:
    network.load(file_name, "./pytorch_models")
except:
    raise ValueError("Could not load the stored model parameters")

done = False
episode_timesteps = 0
state = env.reset()

# Begin the testing loop
while True:
    action = network.get_action(np.array(state)) # 通过网络生成动作

    # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
    # 将动作范围转换为环境所需的范围
    a_in = [(action[0] + 1) / 2, action[1]] # 线速度：[0, 1]，角速度：[-1, 1]
    next_state, reward, done, target = env.step(a_in)
    done = 1 if episode_timesteps + 1 == max_ep else int(done)  # 判断是否回合结束

    # On termination of episode
    if done:
        state = env.reset() # 重置环境
        done = False
        episode_timesteps = 0
    else:
        state = next_state  # 更新状态
        episode_timesteps += 1
