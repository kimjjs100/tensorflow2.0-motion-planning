import numpy as np
import copy
import tensorflow as tf
from tensorflow.keras.layers import Dense
from collections import deque
import random
import matplotlib.pyplot as plt
import time
import pickle
import scipy.io

from hyper_parameter_3 import hp

class Env:
    def __init__(self, step_size, goal_bound, joint1_range, joint2_range):
        
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
        self.obs_r = 1.0
        

        self.margin = self.goal_bound

    def reset(self):
        self.step_count = 0
        self.success = False
        self.p_len = 0

        self.location = (self.joint_range - 2.0 * self.margin) * np.random.rand(1, 2) + self.joint_min + self.margin
        while self.collision_check_init(self.location):
            self.location = (self.joint_range - 2.0 * self.margin) * np.random.rand(1, 2) + self.joint_min + self.margin

        self.goal = (self.joint_range - 2.0 * self.margin) * np.random.rand(1, 2) + self.joint_min + self.margin
        while (np.linalg.norm(self.location - self.goal) <= 2.0*self.goal_bound) | self.collision_check_init(self.goal):
            self.goal = (self.joint_range - 2.0 * self.margin) * np.random.rand(1, 2) + self.joint_min + self.margin

        self.state = np.concatenate((self.location, self.goal), axis=1)
        
        return copy.deepcopy(self.state)
    
    def step(self, action):
        
        self.step_count += 1

        a_len = np.linalg.norm(action)
        if a_len > 1.0:
            action = action/a_len
        
        next_location = self.location + self.step_size*action + np.random.randn(1, 2)/500

        if (next_location > self.joint_max).any() | (next_location < self.joint_min).any() | self.collision_check(next_location):
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
        
        return copy.deepcopy(self.state), reward, done


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

class Actor(tf.keras.Model):
    def __init__(self, state_shape, action_dim, max_action, name="Actor"):
        super().__init__(name=name)

        self.l1 = Dense(400, name="L1")
        self.l2 = Dense(300, name="L3")
        self.l3 = Dense(action_dim, name="L7")

        self.max_action = max_action


    def call(self, inputs):
        features = tf.nn.relu(self.l1(inputs))
        features = tf.nn.relu(self.l2(features))
        features = self.l3(features)
        action = self.max_action * tf.nn.tanh(features)
        return action

class Critic(tf.keras.Model):
    def __init__(self, state_shape, action_dim, name="Critic"):
        super().__init__(name=name)

        self.l1 = Dense(400, name="L1")
        self.l2 = Dense(300, name="L2")
        self.l3 = Dense(1, name="L6")

    def call(self, inputs):
        states, actions = inputs
        xu = tf.concat([states, actions], axis=1)
        x1 = tf.nn.relu(self.l1(xu))
        x1 = tf.nn.relu(self.l2(x1))
        x1 = self.l3(x1)

        return x1


class TD3():
    def __init__(self):
        super().__init__()

        gpu = 0
        max_action = 1.
        state_shape = hp.state_dim #env.observation_space.shape
        action_dim = hp.action_dim #env.action_space.high.size

        actor_update_freq = 2
        policy_noise = 0.2
        noise_clip = 0.5
        lr_critic = 0.001
        lr_actor = 0.001
        sigma = 0.1
        tau = 0.005


        self.policy_name = "TD3"
        self.update_interval = 1
        self.batch_size = 512
        self.discount = 0.98
        self.n_warmup = 2000
        self.n_epoch = 1
        self.max_grad = 10
        self.memory_capacity = 1000000
        self.device = "/cpu:0"


        # Define and initialize Actor network
        self.actor = Actor(state_shape, action_dim, max_action)
        self.actor_target = Actor(state_shape, action_dim, max_action)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_actor)


        # Define and initialize Critic network
        self.critic = Critic(state_shape, action_dim)
        self.critic_target = Critic(state_shape, action_dim)
        self.critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_critic)


        self.critic_2 = Critic(state_shape, action_dim)
        self.critic_2_target = Critic(state_shape, action_dim)
        self.critic_2_optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_critic)


        # Set hyper-parameters
        self.sigma = sigma
        self.tau = tau

        self._policy_noise = policy_noise
        self._noise_clip = noise_clip

        self._actor_update_freq = actor_update_freq
        self._it = tf.Variable(0, dtype=tf.int32)

    def _save_model(self):
        self.actor.save_weights('./save/actor', save_format='tf')
        self.actor_target.save_weights('./save/actor_target', save_format='tf')
        self.critic.save_weights('./save/critic', save_format='tf')
        self.critic_target.save_weights('./save/critic_target', save_format='tf')
        self.critic_2.save_weights('./save/critic_2', save_format='tf')
        self.critic_2_target.save_weights('./save/critic_2_target', save_format='tf')

    def _load_model(self):
        self.actor.load_weights('./save/actor')
        self.actor_target.load_weights('./save/actor_target')
        self.critic.load_weights('./save/critic')
        self.critic_target.load_weights('./save/critic_target')
        self.critic_2.load_weights('./save/critic_2')
        self.critic_2_target.load_weights('./save/critic_2_target')

    def get_action(self, state, test=False, tensor=False):
        is_single_state = len(state.shape) == 1
        if not tensor:
            assert isinstance(state, np.ndarray)
        state = np.expand_dims(state, axis=0).astype(
            np.float32) if is_single_state else state
        action = self._get_action_body(
            tf.constant(state), self.sigma * (1. - test),
            tf.constant(self.actor.max_action, dtype=tf.float32))
        if tensor:
            return action
        else:
            return action.numpy()[0] if is_single_state else action.numpy()

    @tf.function
    def _get_action_body(self, state, sigma, max_action):
        with tf.device(self.device):
            action = self.actor(state)
            action += tf.random.normal(shape=action.shape,
                                       mean=0., stddev=sigma, dtype=tf.float32)
            return tf.clip_by_value(action, -max_action, max_action)


if __name__ == "__main__":

    joint1_range = [-5.0, 5.0]
    joint2_range = [-5.0, 5.0]

    step_size = 0.5
    goal_bound = 0.1


    env = Env(step_size=step_size, goal_bound=goal_bound, joint1_range=joint1_range, joint2_range=joint2_range)

    state_dim = 4
    action_dim = 2

    agent = TD3()

    agent._load_model()


    global_step = 0
    scores = []
    suc_rate = []
    p_len = []

    print(step_size, goal_bound)
    for e in range(20):

        state = env.reset()
        #state = np.reshape(state, [1, state_dim])

        done = False

        local_step = 0
        score = 0

        visual_memory = []

        while (done is False) & (env.step_count < 100):
            global_step += 1
            local_step += 1

            visual_memory.append(env.location)

            action = agent.get_action(state, test=True)


            next_state, reward, done = env.step(action)


            state = next_state
            score += reward

        visual_memory.append(env.location)

        suc_rate.append(env.success)
        p_len.append(env.p_len)
        scores.append(score)

        print('local_step:', local_step, '/ score:', score)

        plt.figure(figsize=(6,6))
        path = np.reshape(visual_memory, [local_step+1, 2])
        circle1=plt.Circle(env.obs[0][0],env.obs_r,color='k')
        circle2=plt.Circle(env.obs[1][0],env.obs_r,color='k')
        circle3=plt.Circle(env.obs[2][0],env.obs_r,color='k')
        circle4=plt.Circle(env.obs[3][0],env.obs_r,color='k')
        plt.gca().add_artist(circle1)
        plt.gca().add_artist(circle2)
        plt.gca().add_artist(circle3)
        plt.gca().add_artist(circle4)
        plt.plot(path[0, 0], path[0, 1], 'kD', markersize=8, mfc='none', markeredgewidth=2., label='start')
        plt.plot(env.goal[0, 0], env.goal[0, 1], 'ko', markersize=10, mfc='none', markeredgewidth=2., label='goal')
        plt.plot(path[:,0],path[:,1],'b')
        plt.plot(path[:,0],path[:,1],'b.')
        plt.xlim(-5,5)
        plt.ylim(-5,5)
        plt.show()


