"""
Data structure for implementing experience replay
Author: Patrick Emami
"""
import random
from collections import deque

import numpy as np


class ReplayBuffer(object):
    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        """
        初始化经验回放缓冲区。
        
        参数：
        - buffer_size：缓冲区最大容量（存储的经验条数）。
        - random_seed：随机种子，用于保证结果的可重复性。
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque() # 使用双端队列存储经验。
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        """
        添加一条新的经验到缓冲区。
        
        参数：
        - s：当前状态。
        - a：采取的动作。
        - r：获得的奖励。
        - t：终止标志（1 表示结束，0 表示未结束）。
        - s2：下一状态。
        """
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            # 如果缓冲区未满，直接添加新经验。
            self.buffer.append(experience)
            self.count += 1
        else:
            # 如果缓冲区已满，移除最旧的经验，添加新经验。
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        """
        返回当前缓冲区存储的经验条数。
        """
        return self.count

    def sample_batch(self, batch_size):
        """
        随机采样一批经验数据。
        
        参数：
        - batch_size：采样的批量大小。
        
        返回值：
        - s_batch：当前状态批量。
        - a_batch：动作批量。
        - r_batch：奖励批量。
        - t_batch：终止标志批量。
        - s2_batch：下一状态批量。
        """
        batch = []

        if self.count < batch_size:
            # 如果当前存储的经验不足一个批量大小，采样所有数据。
            batch = random.sample(self.buffer, self.count)
        else:
            # 从缓冲区中随机采样 `batch_size` 条数据。
            batch = random.sample(self.buffer, batch_size)
        # 将经验分解为单独的状态、动作、奖励等数组。
        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch]).reshape(-1, 1)
        t_batch = np.array([_[3] for _ in batch]).reshape(-1, 1)
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        # 清空缓冲区
        self.buffer.clear()
        self.count = 0
