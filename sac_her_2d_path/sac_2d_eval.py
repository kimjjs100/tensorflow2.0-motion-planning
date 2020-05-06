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


class DiagonalGaussian():
    def __init__(self, dim):
        self._dim = dim
        self._tiny = 1e-8

    @property
    def dim(self):
        return self._dim

    def likelihood_ratio(self, x, old_param, new_param):
        llh_new = self.log_likelihood(x, new_param)
        llh_old = self.log_likelihood(x, old_param)
        return tf.math.exp(llh_new - llh_old)

    def log_likelihood(self, x, param):
        """
        Compute log likelihood as:
        -N/2log(2*pi*sigma^2)-1/(2*sigma^2) * Sum(x - mu)^2
        """
        means = param["mean"]
        log_stds = param["log_std"]
        assert means.shape == log_stds.shape
        zs = (x - means) / tf.exp(log_stds)
        return - tf.reduce_sum(log_stds, axis=-1) \
               - 0.5 * tf.reduce_sum(tf.square(zs), axis=-1) \
               - 0.5 * self.dim * tf.math.log(2 * np.pi)

    def sample(self, param):
        means = param["mean"]
        log_stds = param["log_std"]
        # reparameterization
        return means + tf.random.normal(shape=means.shape) * tf.math.exp(log_stds)

    def entropy(self, param):
        log_stds = param["log_std"]
        return tf.reduce_sum(log_stds + tf.math.log(tf.math.sqrt(2 * np.pi * np.e)), axis=-1)

class GaussianActor(tf.keras.Model):
    LOG_SIG_CAP_MAX = 2  # np.e**2 = 7.389
    LOG_SIG_CAP_MIN = -20  # np.e**-10 = 4.540e-05
    EPS = 1e-6

    def __init__(self, state_shape, action_dim, max_action, hidden_activation="relu",
                 tanh_mean=False, tanh_std=False,
                 fix_std=False, const_std=0.1,
                 state_independent_std=False, name='GaussianPolicy'):
        super().__init__(name=name)
        self.dist = DiagonalGaussian(dim=action_dim)
        self._fix_std = fix_std
        self._tanh_std = tanh_std
        self._const_std = const_std
        self._max_action = max_action
        self._state_independent_std = state_independent_std

        self.l1 = Dense(400, name="L1", activation=hidden_activation)
        self.l2 = Dense(300, name="L2", activation=hidden_activation)
        self.out_mean = Dense(action_dim, name="L_mean", activation='tanh')
        activation = 'tanh' if tanh_std else None
        self.out_log_std = Dense(action_dim, name="L_sigma", activation=activation)


    def _compute_dist(self, states):
        features = self.l1(states)
        features = self.l2(features)
        mean = self.out_mean(features)
        log_std = self.out_log_std(features)
        log_std = tf.clip_by_value(log_std, self.LOG_SIG_CAP_MIN, self.LOG_SIG_CAP_MAX)

        return {"mean": mean, "log_std": log_std}

    def call(self, states, test=False):
        param = self._compute_dist(states)
        if test:
            raw_actions = param["mean"]
        else:
            raw_actions = self.dist.sample(param)
        logp_pis = self.dist.log_likelihood(raw_actions, param)

        actions = raw_actions

        return actions * self._max_action, logp_pis, param


class CriticV(tf.keras.Model):
    def __init__(self, state_shape, name='vf'):
        super().__init__(name=name)

        self.l1 = Dense(400, name="L1", activation='relu')
        self.l2 = Dense(300, name="L2", activation='relu')
        self.l3 = Dense(1, name="L3", activation='linear')


    def call(self, states):
        features = self.l1(states)
        features = self.l2(features)
        values = self.l3(features)

        return tf.squeeze(values, axis=1, name="values")

class CriticQ(tf.keras.Model):
    def __init__(self, state_shape, action_dim, name='qf'):
        super().__init__(name=name)

        self.l1 = Dense(400, name="L1", activation='relu')
        self.l2 = Dense(300, name="L2", activation='relu')
        self.l3 = Dense(1, name="L2", activation='linear')


    def call(self, inputs):
        [states, actions] = inputs
        features = tf.concat([states, actions], axis=1)
        features = self.l1(features)
        features = self.l2(features)
        values = self.l3(features)

        return tf.squeeze(values, axis=1)


class SAC():
    def __init__(self):
        super().__init__()

        gpu = 0
        max_action = 1.
        state_shape = hp.state_dim #env.observation_space.shape
        action_dim = hp.action_dim #env.action_space.high.size

        lr = 3e-4
        sigma = 0.1
        tau = 0.005
        self.state_ndim = len((1, state_shape))

        self.policy_name = "SAC"
        self.update_interval = 1
        self.batch_size = 512
        self.discount = 0.98
        self.n_warmup = 2000
        self.n_epoch = 1
        self.max_grad = 10
        self.memory_capacity = 1000000
        self.device = "/gpu:{}".format(gpu) if gpu >= 0 else "/cpu:0"


        # Define and initialize Actor network
        self.actor = GaussianActor(state_shape, action_dim, max_action, tanh_mean=False, tanh_std=False)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        # Define and initialize Critic network
        self.qf1 = CriticQ(state_shape, action_dim, name="qf1")
        self.qf2 = CriticQ(state_shape, action_dim, name="qf2")
        self.qf1_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.qf2_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        self.vf = CriticV(state_shape)
        self.vf_target = CriticV(state_shape)
        self.vf_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        # Set hyper-parameters
        self.sigma = sigma
        self.tau = tau

        self.log_alpha = tf.Variable(0., dtype=tf.float32)
        self.alpha = tf.Variable(0., dtype=tf.float32)
        self.alpha.assign(tf.exp(self.log_alpha))
        self.target_alpha = -action_dim
        self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def _save_model(self):
        self.actor.save_weights('./save/actor', save_format='tf')
        self.qf1.save_weights('./save/qf1', save_format='tf')
        self.qf2.save_weights('./save/qf2', save_format='tf')
        self.vf.save_weights('./save/vf', save_format='tf')
        self.vf_target.save_weights('./save/vf_target', save_format='tf')

    def _load_model(self):
        self.actor.load_weights('./save/actor')
        self.qf1.load_weights('./save/critic')
        self.qf2.load_weights('./save/critic_target')
        self.vf.load_weights('./save/critic_2')
        self.vf_target.load_weights('./save/critic_2_target')

    def get_action(self, state, test=False):
        assert isinstance(state, np.ndarray)
        is_single_state = len(state.shape) == self.state_ndim

        state = np.expand_dims(state, axis=0).astype(
            np.float32) if is_single_state else state
        action = self._get_action_body(tf.constant(state), test)

        return action.numpy()[0] if is_single_state else action

    @tf.function
    def _get_action_body(self, state, test):
        return self.actor(state, test)[0]


if __name__ == "__main__":

    joint1_range = [-5.0, 5.0]
    joint2_range = [-5.0, 5.0]

    step_size = 0.5
    goal_bound = 0.1


    env = Env(step_size=step_size, goal_bound=goal_bound, joint1_range=joint1_range, joint2_range=joint2_range)

    state_dim = 4
    action_dim = 2

    agent = SAC()

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


