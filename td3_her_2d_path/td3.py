from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import copy

from cpprb import ReplayBuffer

import tensorflow as tf
from tensorflow.keras.layers import Dense

from hyper_parameter_3 import hp
from environment_3 import Environment


if tf.config.experimental.list_physical_devices('GPU'):
    for cur_device in tf.config.experimental.list_physical_devices("GPU"):
        print(cur_device)
        tf.config.experimental.set_memory_growth(cur_device, enable=True)

def update_target_variables(target_variables,
                            source_variables,
                            tau=1.0,
                            use_locking=False,
                            name="update_target_variables"):

    if not isinstance(tau, float):
        raise TypeError("Tau has wrong type (should be float) {}".format(tau))
    if not 0.0 < tau <= 1.0:
        raise ValueError("Invalid parameter tau {}".format(tau))
    if len(target_variables) != len(source_variables):
        raise ValueError("Number of target variables {} is not the same as "
                         "number of source variables {}".format(
                             len(target_variables), len(source_variables)))

    same_shape = all(trg.get_shape() == src.get_shape()
                     for trg, src in zip(target_variables, source_variables))
    if not same_shape:
        raise ValueError("Target variables don't have the same shape as source "
                         "variables.")

    def update_op(target_variable, source_variable, tau):
        if tau == 1.0:
            return target_variable.assign(source_variable, use_locking)
        else:
            return target_variable.assign(
                tau * source_variable + (1.0 - tau) * target_variable, use_locking)

    # with tf.name_scope(name, values=target_variables + source_variables):
    update_ops = [update_op(target_var, source_var, tau)
                  for target_var, source_var
                  in zip(target_variables, source_variables)]
    return tf.group(name="update_all_variables", *update_ops)




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
        self.device = "/gpu:{}".format(gpu) if gpu >= 0 else "/cpu:0"


        # Define and initialize Actor network
        self.actor = Actor(state_shape, action_dim, max_action)
        self.actor_target = Actor(state_shape, action_dim, max_action)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_actor)
        update_target_variables(self.actor_target.weights,
                                self.actor.weights, tau=1.)

        # Define and initialize Critic network
        self.critic = Critic(state_shape, action_dim)
        self.critic_target = Critic(state_shape, action_dim)
        self.critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_critic)
        update_target_variables(
            self.critic_target.weights, self.critic.weights, tau=1.)

        self.critic_2 = Critic(state_shape, action_dim)
        self.critic_2_target = Critic(state_shape, action_dim)
        self.critic_2_optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_critic)
        update_target_variables(
            self.critic_2_target.weights, self.critic_2.weights, tau=1.)

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

    def train(self, states, actions, next_states, rewards, done, weights=None):
        if weights is None:
            weights = np.ones_like(rewards)
        actor_loss, critic_loss, td_errors = self._train_body(
            states, actions, next_states, rewards, done, weights)

        return actor_loss, critic_loss, td_errors

    @tf.function
    def _train_body(self, states, actions, next_states, rewards, done, weights):

        with tf.device(self.device):
            with tf.GradientTape(persistent=True) as tape:
                td_error1, td_error2 = self._compute_td_error_body(states, actions, next_states, rewards, done)

                critic_loss = tf.reduce_mean(tf.square(td_error1))
                critic_2_loss = tf.reduce_mean(tf.square(td_error2))

            critic_grad = tape.gradient(
                critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(
                zip(critic_grad, self.critic.trainable_variables))

            critic_2_grad = tape.gradient(
                critic_2_loss, self.critic_2.trainable_variables)
            self.critic_2_optimizer.apply_gradients(
                zip(critic_2_grad, self.critic_2.trainable_variables))

            self._it.assign_add(1)
            with tf.GradientTape() as tape:
                next_actions = self.actor(states)
                actor_loss = - tf.reduce_mean(self.critic([states, next_actions])) + tf.reduce_mean(tf.square(self.actor(states)))

            if tf.math.equal(self._it % self._actor_update_freq, 0):
                actor_grad = tape.gradient(
                    actor_loss, self.actor.trainable_variables)
                self.actor_optimizer.apply_gradients(
                    zip(actor_grad, self.actor.trainable_variables))

                # Update target networks
                update_target_variables(
                    self.critic_target.weights, self.critic.weights, self.tau)
                update_target_variables(
                    self.critic_2_target.weights, self.critic_2.weights, self.tau)

                update_target_variables(
                    self.actor_target.weights, self.actor.weights, self.tau)


            return actor_loss, critic_loss, tf.abs(td_error1) + tf.abs(td_error2)

    def compute_td_error(self, states, actions, next_states, rewards, dones):
        td_errors1, td_errors2 = self._compute_td_error_body(
            states, actions, next_states, rewards, dones)
        return np.squeeze(np.abs(td_errors1.numpy()) + np.abs(td_errors2.numpy()))

    @tf.function
    def _compute_td_error_body(self, states, actions, next_states, rewards, dones):
        with tf.device(self.device):
            not_dones = 1. - dones

            # Get noisy action
            next_action = self.actor_target(next_states)
            noise = tf.cast(tf.clip_by_value(
                tf.random.normal(shape=tf.shape(next_action),
                                 stddev=self._policy_noise),
                -self._noise_clip, self._noise_clip), tf.float32)
            next_action = tf.clip_by_value(
                next_action + noise, -self.actor_target.max_action, self.actor_target.max_action)

            target_Q1 = self.critic_target([next_states, next_action])
            target_Q2 = self.critic_2_target([next_states, next_action])
            target_Q = tf.minimum(target_Q1, target_Q2)
            target_Q = rewards + (not_dones * self.discount * target_Q)
            target_Q = tf.stop_gradient(target_Q)
            current_Q1 = self.critic([states, actions])
            current_Q2 = self.critic_2([states, actions])

        return target_Q - current_Q1, target_Q - current_Q2



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


def get_replay_buffer(policy, env):
    size = policy.memory_capacity

    kwargs = {
        "size": size,
        "default_dtype": np.float32,
        "env_dict": {
            "obs": {
                "shape": hp.state_dim},
            "next_obs": {
                "shape": hp.state_dim},
            "act": {
                "shape": hp.action_dim},
            "rew": {},
            "done": {}}}

    return ReplayBuffer(**kwargs)

class Trainer:
    def __init__(self, policy, env):
        self._policy = policy
        self._env = env

        # experiment settings
        self._max_steps = int(1e7)
        self._episode_max_steps = 50
        self._n_experiments = 1
        self._show_progress = False
        self._save_model_interval = 5000
        self._save_summary_interval = int(1e3)


        # tensorboard
        self.log_dir = 'logs/'
        self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        self.success_rate = tf.keras.metrics.Mean('success', dtype=tf.float32)

    def __call__(self):
        total_steps = 0
        episode_steps = 0
        episode_return = 0
        episode_start_time = time.perf_counter()
        n_episode = 0
        n_training = 0

        random_action_prob = 0.1

        replay_buffer = get_replay_buffer(
            self._policy, self._env)

        obs = self._env.reset()

        local_memory = []

        h_score = []
        h_success = []

        while total_steps < self._max_steps:
            if total_steps < self._policy.n_warmup:
                action = 2.0 * np.random.rand(hp.action_dim) - 1.0
                action_length = np.linalg.norm(action)
                if action_length > 1.0:
                    action = action / action_length
            else:
                if np.random.rand() < random_action_prob:
                    action = 2.0 * np.random.rand(hp.action_dim) - 1.0
                    action_length = np.linalg.norm(action)
                    if action_length > 1.0:
                        action = action / action_length
                else:
                    action = self._policy.get_action(obs)

            next_obs, reward, done, _ = self._env.step(action)


            episode_steps += 1
            episode_return += reward
            total_steps += 1
            if episode_steps == self._episode_max_steps:
                done = True

            replay_buffer.add(obs=obs, act=action, next_obs=next_obs, rew=reward, done=done)
            local_memory.append((obs, action, reward, next_obs, done))

            obs = next_obs

            if done:

                for h in range(episode_steps):
                    state, action, reward, next_state, done = copy.deepcopy(local_memory[h])

                    for her in range(4):
                        future = np.random.randint(h, episode_steps)

                        _, _, _, g, _ = copy.deepcopy(local_memory[future])
                        state[:, 2:4] = g[:, 0:2]
                        if np.linalg.norm(state[:, 0:2] - state[:, 2:4]) <= self._env.goal_bound:
                            continue
                        next_state[:, 2:4] = g[:, 0:2]
                        goal_d = np.linalg.norm(next_state[:, 0:2] - next_state[:, 2:4])
                        if goal_d <= self._env.goal_bound:
                            reward = 0.0
                            done = True
                        else:
                            reward = -1.0
                            done = False
                        replay_buffer.add(obs=state, act=action, next_obs=next_state, rew=reward, done=done)

                local_memory = []


                success = self._env.success
                h_score.append(episode_return)
                h_success.append(success)


                obs = self._env.reset()

                n_episode += 1
                fps = episode_steps / (time.perf_counter() - episode_start_time)
                print("Total Epi: {0: 5} Steps: {1: 7} Episode Steps: {2: 5} Return: {3: 5.4f} FPS: {4:5.2f}".format(
                    n_episode, total_steps, episode_steps, episode_return, fps))

                if ((n_episode + 1) % 10) == 0 :
                    with self.train_summary_writer.as_default():
                        tf.summary.scalar('reward', np.mean(h_score), step=n_episode)
                        tf.summary.scalar('success', np.mean(h_success), step=n_episode)
                    h_score = []
                    h_success = []

                episode_steps = 0
                episode_return = 0
                episode_start_time = time.perf_counter()

                if total_steps > self._policy.n_warmup:
                    n_training += 1
                    samples = replay_buffer.sample(self._policy.batch_size)
                    actor_loss, critic_loss, _ = self._policy.train(samples["obs"], samples["act"], samples["next_obs"],
                                                                    samples["rew"],
                                                                    np.array(samples["done"], dtype=np.float32))

                    with self.train_summary_writer.as_default():
                        tf.summary.scalar('actor loss', actor_loss, step=n_training)
                        tf.summary.scalar('critic loss', critic_loss, step=n_training)


            if total_steps < self._policy.n_warmup:
                continue

            if n_episode % self._save_model_interval == 0 :
                self._policy._save_model()



if __name__ == '__main__':

    env = Environment()
    policy = TD3()
    trainer = Trainer(policy, env)
    trainer()
