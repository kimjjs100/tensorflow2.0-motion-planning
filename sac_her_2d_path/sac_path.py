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

def huber_loss(x, delta=1.):
    delta = tf.ones_like(x) * delta
    less_than_max = 0.5 * tf.square(x)
    greater_than_max = delta * (tf.abs(x) - 0.5 * delta)
    return tf.where(
        tf.abs(x) <= delta,
        x=less_than_max,
        y=greater_than_max)

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
        update_target_variables(self.vf_target.weights,
                                self.vf.weights, tau=1.)
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
        self.actor_target.load_weights('./save/actor_target')
        self.critic.load_weights('./save/critic')
        self.critic_target.load_weights('./save/critic_target')
        self.critic_2.load_weights('./save/critic_2')
        self.critic_2_target.load_weights('./save/critic_2_target')

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

    def train(self, states, actions, next_states, rewards, done, weights=None):
        if weights is None:
            weights = np.ones_like(rewards)
        td_errors, actor_loss, vf_loss, qf_loss, logp_min, logp_max, logp_mean = \
            self._train_body(states, actions, next_states,
                             rewards, done, weights)

        return actor_loss, qf_loss, td_errors

    @tf.function
    def _train_body(self, states, actions, next_states, rewards, dones, weights):
        with tf.device(self.device):
            if tf.rank(rewards) == 2:
                rewards = tf.squeeze(rewards, axis=1)
            not_dones = 1. - tf.cast(dones, dtype=tf.float32)

            with tf.GradientTape(persistent=True) as tape:
                # Compute loss of critic Q
                current_q1 = self.qf1([states, actions])
                current_q2 = self.qf2([states, actions])
                vf_next_target = self.vf_target(next_states)

                target_q = tf.stop_gradient(
                    rewards + not_dones * self.discount * vf_next_target)

                td_loss_q1 = tf.reduce_mean(huber_loss(
                    target_q - current_q1, delta=self.max_grad) * weights)
                td_loss_q2 = tf.reduce_mean(huber_loss(
                    target_q - current_q2, delta=self.max_grad) * weights)  # Eq.(7)

                # Compute loss of critic V
                current_v = self.vf(states)

                sample_actions, logp, _ = self.actor(states)  # Resample actions to update V
                current_q1 = self.qf1([states, sample_actions])
                current_q2 = self.qf2([states, sample_actions])
                current_min_q = tf.minimum(current_q1, current_q2)

                target_v = tf.stop_gradient(
                    current_min_q - self.alpha * logp)
                td_errors = target_v - current_v
                td_loss_v = tf.reduce_mean(
                    huber_loss(td_errors, delta=self.max_grad) * weights)  # Eq.(5)

                # Compute loss of policy
                policy_loss = tf.reduce_mean(
                    (self.alpha * logp - current_min_q) * weights)  # Eq.(12)

                alpha_loss = -tf.reduce_mean(
                    (self.log_alpha * tf.stop_gradient(logp + self.target_alpha)))

            q1_grad = tape.gradient(td_loss_q1, self.qf1.trainable_variables)
            self.qf1_optimizer.apply_gradients(
                zip(q1_grad, self.qf1.trainable_variables))
            q2_grad = tape.gradient(td_loss_q2, self.qf2.trainable_variables)
            self.qf2_optimizer.apply_gradients(
                zip(q2_grad, self.qf2.trainable_variables))

            vf_grad = tape.gradient(td_loss_v, self.vf.trainable_variables)
            self.vf_optimizer.apply_gradients(
                zip(vf_grad, self.vf.trainable_variables))
            update_target_variables(
                self.vf_target.weights, self.vf.weights, self.tau)

            actor_grad = tape.gradient(
                policy_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(actor_grad, self.actor.trainable_variables))

            alpha_grad = tape.gradient(alpha_loss, [self.log_alpha])
            self.alpha_optimizer.apply_gradients(
                zip(alpha_grad, [self.log_alpha]))
            self.alpha.assign(tf.exp(self.log_alpha))

            del tape

        return td_errors, policy_loss, td_loss_v, td_loss_q1, tf.reduce_min(logp), tf.reduce_max(logp), tf.reduce_mean(
            logp)


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




def get_default_rb_dict(size, env):
    return {
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

def get_replay_buffer(policy, env, size=None):
    if policy is None or env is None:
        return None

    kwargs = get_default_rb_dict(policy.memory_capacity, env)

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
    policy = SAC()
    trainer = Trainer(policy, env)
    trainer()
