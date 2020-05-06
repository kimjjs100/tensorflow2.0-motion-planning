import numpy as np
import scipy.io
import math
import copy
import tensorflow as tf

pi = math.pi

# Graph & dijkstra

import collections
from collections import deque
import math
import random
import pickle


class Environment:
    def __init__(self):

        joint1_range = [-5.0, 5.0]
        joint2_range = [-5.0, 5.0]

        step_size = 0.5
        goal_bound = 0.1


        self.step_size = step_size
        self.goal_bound = goal_bound

        self.joint1_range = joint1_range
        self.joint2_range = joint2_range

        self.joint_min = np.array([[self.joint1_range[0], self.joint2_range[0]]])
        self.joint_max = np.array([[self.joint1_range[1], self.joint2_range[1]]])
        self.joint_range = self.joint_max - self.joint_min

        self.step_count = 0
        self.success = False
        self.p_len = 0

        self.location = 0
        self.goal = 0
        self.goal_dir = 0
        self.state = 0

        self.obs = []
        self.obs.append(np.array([[2.5, 2.5]]))
        self.obs.append(np.array([[2.5, -2.5]]))
        self.obs.append(np.array([[-2.5, 2.5]]))
        self.obs.append(np.array([[-2.5, -2.5]]))
        self.obs.append(np.array([[0.0, 0.0]]))
        self.obs_r = 1.0

        with open('./evaldata_100.pckl', 'rb') as f:
            self.evaldata = pickle.load(f)

        self.margin = self.goal_bound

    def reset(self):
        self.step_count = 0
        self.success = False
        self.p_len = 0
        self.time_step = 0

        self.location = (self.joint_range - 2.0 * self.margin) * np.random.rand(1, 2) + self.joint_min + self.margin
        while self.collision_check_init(self.location):
            self.location = (self.joint_range - 2.0 * self.margin) * np.random.rand(1, 2) + self.joint_min + self.margin

        self.goal = (self.joint_range - 2.0 * self.margin) * np.random.rand(1, 2) + self.joint_min + self.margin
        while (np.linalg.norm(self.location - self.goal) <= 2.0 * self.goal_bound) | self.collision_check_init(
                self.goal):
            self.goal = (self.joint_range - 2.0 * self.margin) * np.random.rand(1, 2) + self.joint_min + self.margin

        self.state = np.concatenate((self.location, self.goal), axis=1)

        return copy.deepcopy(self.state)

    def eval_reset(self, epi):
        self.step_count = 0
        self.success = False
        self.p_len = 0

        self.location = self.evaldata[epi][:, 0:2]
        self.goal = self.evaldata[epi][:, 2:4]

        self.state = np.concatenate((self.location, self.goal), axis=1)

        return copy.deepcopy(self.state)

    def step(self, action):

        self.time_step += 1
        self.step_count += 1

        a_len = np.linalg.norm(action)
        if a_len >= 1.0:
            action = action / a_len

        next_location = self.location + self.step_size * action

        if (next_location > self.joint_max).any() | (next_location < self.joint_min).any() | self.collision_check(
                next_location):
            reward = -1.0
            done = False

            self.p_len += self.step_size
        else:
            goal_d = np.linalg.norm(next_location - self.goal)
            self.location = np.copy(next_location)

            self.p_len += self.step_size * np.linalg.norm(action)

            if goal_d <= self.goal_bound:
                reward = 0.0
                done = True
                self.success = True
            else:
                reward = -1.0
                done = False

        self.state = np.concatenate((self.location, self.goal), axis=1)

        return copy.deepcopy(self.state), reward, done, 0

    def collision_check_init(self, config_p):

        flag = False

        for i in range(len(self.obs)):
            if np.linalg.norm(config_p - self.obs[i]) <= self.obs_r + self.margin:
                flag = True
                break

        return flag

    def collision_check(self, config_p):

        flag = False

        for i in range(len(self.obs)):
            if np.linalg.norm(config_p - self.obs[i]) <= self.obs_r:
                flag = True
                break

        return flag
