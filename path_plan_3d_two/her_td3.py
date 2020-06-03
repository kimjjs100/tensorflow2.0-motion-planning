import numpy as np
import tensorflow.compat.v1 as tf   # 2.x 에서 1.x 코드 사용하기
from collections import deque
import random
import matplotlib.pyplot as plt
from hyper_parameter import hp
from environment import Environment
import pickle

class ddpg(object):

    def __init__(self, active_writer):

        # 상태 및 행동의 크기 정의
        self.state_dim = hp.state_dim
        self.action_dim = hp.action_dim

        # actor와 critic의 learning rate 정의
        self.lr_actor = hp.lr_actor
        self.lr_critic = hp.lr_critic

        self.target_noise_std = hp.target_noise_std     # 0.2
        self.target_noise_clip = hp.target_noise_clip   # 0.5
        self.policy_delay = hp.policy_delay     # 2
        self.actor_l2 = hp.actor_l2     # 1.0

        # target 신경망의 업데이트 비율 정의
        self.tau = hp.tau   # 0.005

        # discount factor
        self.discount_factor = hp.discount_factor

        # 한 번의 업데이트에 샘플링하여 사용할 데이터 갯수 정의
        self.batch_size = hp.batch_size

        # 메모리 정의
        self.memory = deque(maxlen=hp.memory_size)

        self.graph = tf.Graph()

        with self.graph.as_default():
            with tf.device('/device:GPU:0'):

                # 학습에 사용할 데이터 변수 정의 (상태, 행동, 보상, 다음상태, 종료여부)
                self.state_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim], name='state_ph')
                self.reward_ph = tf.placeholder(dtype=tf.float32, shape=[None], name='reward_ph')
                self.action_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.action_dim], name='action_ph')
                self.next_state_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim], name='next_state_ph')
                self.done_ph = tf.placeholder(dtype=tf.float32, shape=[None], name='done_ph')

                # tensorboard 기록용 변수 정의
                if active_writer:
                    self.smry_reward_ph = tf.placeholder(dtype=tf.float32, shape=[], name='smry_reward_ph')
                    self.smry_time_step_ph = tf.placeholder(dtype=tf.float32, shape=[], name='smry_time_step_ph')
                    self.smry_length_ph = tf.placeholder(dtype=tf.float32, shape=[], name='smry_length_ph')
                    self.smry_success_ph = tf.placeholder(dtype=tf.float32, shape=[], name='smry_success_ph')

                    self.smry_test_length_ph = tf.placeholder(dtype=tf.float32, shape=[], name='smry_test_length_ph')
                    self.smry_test_success_ph = tf.placeholder(dtype=tf.float32, shape=[], name='smry_test_success_ph')

                    self.smry_actor_loss = tf.placeholder(dtype=tf.float32, shape=[], name='smry_actor_loss_ph')
                    self.smry_critic1_loss = tf.placeholder(dtype=tf.float32, shape=[], name='smry_critic1_loss_ph')


                # actor 모델과 타겟 생성
                self.actor_model = self.build_actor(self.state_ph, 'actor_model')
                self.actor_target = self.build_actor(self.next_state_ph, 'actor_target')

                # critic 모델과 타겟 생성
                self.critic_model_1 = self.build_critic(self.state_ph, self.action_ph, 'critic_model_1')
                self.critic_model_1_copy = self.build_critic(self.state_ph, self.actor_model, 'critic_model_1')
                self.critic_model_2 = self.build_critic(self.state_ph, self.action_ph, 'critic_model_2')

                self.critic_target_action = self.build_critic_target_action(self.actor_target, 'critic_target_action')

                self.critic_target_1 = self.build_critic(self.next_state_ph, self.critic_target_action, 'critic_target_1')
                self.critic_target_2 = self.build_critic(self.next_state_ph, self.critic_target_action, 'critic_target_2')


                # actor 모델의 업데이트 수식 정의
                self.actor_opt = self.build_actor_opt('actor_opt')

                # critic 모델의 업데이트 수식 정의
                self.critic_opt = self.build_critic_opt('critic_opt')


                self.actor_loss = self.build_actor_loss('actor_loss')
                self.critic_loss = self.build_critic_loss('critic_loss')

                # actor 타겟의 업데이트 수식 정의
                self.actor_target_update = self.build_actor_target_update('actor_target_update')

                # critic 타겟의 업데이트 수식 정의
                self.critic_target_update = self.build_critic_target_update('critic_target_update')

                # actor와 critic 타겟의 초기값 업데이트 수식 정의
                self.target_initialize = self.build_target_initialize('target_initialize')

                # tensorboard 기록용 변수 정의
                if active_writer:
                    self.summary, self.summary_test = self.build_summary()

                # tensorboard 기록 공간 정의
                if active_writer:
                    self.writer = tf.summary.FileWriter(logdir='./tf_board', graph=self.graph)

                # 학습한 신경망을 저장하기위해 사용
                self.saver = tf.train.Saver()

                config = tf.ConfigProto(allow_soft_placement=True)
                config.gpu_options.allow_growth = True

                self.sess = tf.Session(config=config, graph=self.graph)

                self.sess.run(tf.global_variables_initializer())

                self.sess.run(self.target_initialize)

    # actor 신경망 구조를 정의하는 함수
    # 신경망의 입력은 state가 되고 출력은 action이 됨
    def build_actor(self, state, name):

        with tf.variable_scope(name_or_scope=name, reuse=tf.AUTO_REUSE):

            layer1 = tf.layers.dense(inputs=state, units=400, activation=tf.nn.relu, name='layer1')
            layer2 = tf.layers.dense(inputs=layer1, units=300, activation=tf.nn.relu, name='layer2')
            layer3 = tf.layers.dense(inputs=layer2, units=self.action_dim, activation=tf.nn.tanh,
                                     kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3), name='layer3')

        return layer3

    # critic 신경망 구조를 정의하는 함수
    # 신경망의 입력은 state와 action이 되고 출력은 Q 값이 됨
    def build_critic(self, state, action, name):

        with tf.variable_scope(name_or_scope=name, reuse=tf.AUTO_REUSE):

            layer1 = tf.layers.dense(inputs=tf.concat([state, action], axis=1), units=400, activation=tf.nn.relu, name='layer1')
            layer2 = tf.layers.dense(inputs=layer1, units=300, activation=tf.nn.relu, name='layer2')
            layer3 = tf.layers.dense(inputs=layer2, units=1, activation=None,
                                     kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3), name='layer3')

        return layer3

    def build_critic_target_action(self, action, name):

        with tf.variable_scope(name_or_scope=name, reuse=tf.AUTO_REUSE):

            epsilon = tf.random_normal(tf.shape(action), stddev=self.target_noise_std)  # 0.2
            epsilon = tf.clip_by_value(epsilon, -self.target_noise_clip, self.target_noise_clip)    # 0.5
            action_noise = action + epsilon     # target noise : paper 5.3
            action_noise_clip = tf.clip_by_value(action_noise, -1.0, 1.0)

        return action_noise_clip

    # actor 신경망의 업데이트 수식을 정의하는 함수
    # critic 신경망의 Q 값 자체가 목적 함수가 됨. 이 값을 증가시키는 방향으로 actor 신경망을 업데이트
    def build_actor_opt(self, name):

        with tf.variable_scope(name_or_scope=name):

            loss = - tf.reduce_mean(self.critic_model_1_copy) + self.actor_l2 * tf.reduce_mean(tf.square(self.actor_model))

            train = tf.train.AdamOptimizer(learning_rate=self.lr_actor).minimize(loss=loss, var_list=tf.trainable_variables(scope='actor_model'))

        return train

    def build_actor_loss(self, name):

        with tf.variable_scope(name_or_scope=name):

            loss = - tf.reduce_mean(self.critic_model_1_copy) + self.actor_l2 * tf.reduce_mean(tf.square(self.actor_model))

        return loss

    # critic 신경망의 업데이트 수식을 정의하는 함수
    # DQN의 수식과 유사. 현재 상태와 행동에따른 Q(s,a) 와 보상 및 다음 상태와 행동에 따른 r + gaama * Q(s',a')의 차이를 줄이는 방향으로 critic 신경망을 업데이트
    def build_critic_opt(self, name):

        with tf.variable_scope(name_or_scope=name):

            target_value = tf.minimum(self.critic_target_1, self.critic_target_2)   # clipped : paper 4.2

            target = tf.stop_gradient(self.reward_ph + self.discount_factor * (1.0 - self.done_ph) * tf.squeeze(target_value, axis=1))

            loss1 = tf.reduce_mean(tf.square(tf.squeeze(self.critic_model_1, axis=1) - target))

            loss2 = tf.reduce_mean(tf.square(tf.squeeze(self.critic_model_2, axis=1) - target))

            train1 = tf.train.AdamOptimizer(learning_rate=self.lr_critic).minimize(loss1, var_list=tf.trainable_variables( scope='critic_model_1'))

            train2 = tf.train.AdamOptimizer(learning_rate=self.lr_critic).minimize(loss2, var_list=tf.trainable_variables( scope='critic_model_2'))

        return train1, train2

    def build_critic_loss(self, name):

        with tf.variable_scope(name_or_scope=name):

            target_value = tf.minimum(self.critic_target_1, self.critic_target_2)   # clipped : paper 4.2

            target = tf.stop_gradient(self.reward_ph + self.discount_factor * (1.0 - self.done_ph) * tf.squeeze(target_value, axis=1))

            loss1 = tf.reduce_mean(tf.square(tf.squeeze(self.critic_model_1, axis=1) - target))

            loss2 = tf.reduce_mean(tf.square(tf.squeeze(self.critic_model_2, axis=1) - target))

        return loss1, loss2

    # actor 타겟 신경망의 업데이트 수식을 정의하는 함수  # slowly update : paper 5.2
    # actor 모델의 값을 일정 비율 가져오는 방식
    def build_actor_target_update(self, name):

        with tf.variable_scope(name_or_scope=name):

            var_actor_model = tf.trainable_variables(scope='actor_model')
            var_actor_target = tf.trainable_variables(scope='actor_target')

            update = []

            for i in range(len(var_actor_model)):
                update.append(tf.assign(ref=var_actor_target[i], value=self.tau * var_actor_model[i] + (1.0 - self.tau) * var_actor_target[i]))

            update = tf.group(*update)

        return update

    # critic 타겟 신경망의 업데이트 수식을 정의하는 함수
    # critic 모델의 값을 일정 비율 가져오는 방식
    def build_critic_target_update(self, name):

        with tf.variable_scope(name_or_scope=name):

            var_critic_model_1 = tf.trainable_variables(scope='critic_model_1')
            var_critic_model_2 = tf.trainable_variables(scope='critic_model_2')
            var_critic_target_1 = tf.trainable_variables(scope='critic_target_1')
            var_critic_target_2 = tf.trainable_variables(scope='critic_target_2')

            update = []

            for i in range(len(var_critic_model_1)):
                update.append(tf.assign(ref=var_critic_target_1[i], value=self.tau * var_critic_model_1[i] + (1.0 - self.tau) * var_critic_target_1[i]))
                update.append(tf.assign(ref=var_critic_target_2[i], value=self.tau * var_critic_model_2[i] + (1.0 - self.tau) * var_critic_target_2[i]))

            update = tf.group(*update)

        return update

    # actor와 critic 타겟 신경망의 초기값을 정의하는 함수
    # actor와 critic 모델의 값을 복사 하여 그대로 사용
    def build_target_initialize(self, name):

        with tf.variable_scope(name_or_scope=name):

            var_actor_model = tf.trainable_variables(scope='actor_model')
            var_actor_target = tf.trainable_variables(scope='actor_target')

            var_critic_model_1 = tf.trainable_variables(scope='critic_model_1')
            var_critic_model_2 = tf.trainable_variables(scope='critic_model_2')
            var_critic_target_1 = tf.trainable_variables(scope='critic_target_1')
            var_critic_target_2 = tf.trainable_variables(scope='critic_target_2')

            update = []

            for i in range(len(var_actor_model)):
                update.append(tf.assign(ref=var_actor_target[i], value=var_actor_model[i]))

            for j in range(len(var_critic_model_1)):
                update.append(tf.assign(ref=var_critic_target_1[j], value=var_critic_model_1[j]))
                update.append(tf.assign(ref=var_critic_target_2[j], value=var_critic_model_2[j]))

            update = tf.group(*update)

        return update

    # tensorboard 기록용 변수들을 정의하는 함수
    def build_summary(self):

        summary_reward = tf.summary.scalar(name='reward', tensor=self.smry_reward_ph)
        summary_time_step = tf.summary.scalar(name='time_step', tensor=self.smry_time_step_ph)
        summary_length = tf.summary.scalar(name='length', tensor=self.smry_length_ph)
        summary_success = tf.summary.scalar(name='success', tensor=self.smry_success_ph)

        summary_test_length = tf.summary.scalar(name='test_length', tensor=self.smry_test_length_ph)
        summary_test_success = tf.summary.scalar(name='test_success', tensor=self.smry_test_success_ph)

        summary_actor_loss = tf.summary.scalar(name='actor_loss', tensor=self.smry_actor_loss)
        summary_critic1_loss = tf.summary.scalar(name='critic1_loss', tensor=self.smry_critic1_loss)


        summary = tf.summary.merge([summary_reward,
                                    summary_time_step,
                                    summary_length,
                                    summary_success,
                                    summary_actor_loss,
                                    summary_critic1_loss])

        summary_test = tf.summary.merge([summary_test_length,
                                         summary_test_success])

        return summary, summary_test

    # tensorboard 기록을 실행하는 함수
    def run_summary(self, reward, time_step, length, success, e, actor_loss, critic1_loss):

        result = self.sess.run(self.summary, feed_dict={self.smry_reward_ph: reward,
                                                        self.smry_time_step_ph: time_step,
                                                        self.smry_length_ph: length,
                                                        self.smry_success_ph: success,
                                                        self.smry_actor_loss: actor_loss,
                                                        self.smry_critic1_loss: critic1_loss})

        self.writer.add_summary(summary=result, global_step=e)

    def run_summary_test(self, test_length, test_success, e):

        result = self.sess.run(self.summary_test, feed_dict={self.smry_test_length_ph: test_length,
                                                             self.smry_test_success_ph: test_success})

        self.writer.add_summary(summary=result, global_step=e)

    # actor 및 critic 모델과 타겟 신경망의 업데이트를 실행하는 함수
    # 메모리에서 데이터를 샘플링하여 모델의 업데이트에 필요한 데이터를 넣어주는 역할
    def train_model(self, iter):
        loss = 0
        loss1 = 0
        for i in range(iter):

            batch_data = random.sample(self.memory, self.batch_size)

            state = np.empty([self.batch_size, self.state_dim])
            action = np.empty([self.batch_size, self.action_dim])
            reward = np.empty([self.batch_size])
            next_state = np.empty([self.batch_size, self.state_dim])
            done = np.empty([self.batch_size])

            for j in range(self.batch_size):
                state[j] = batch_data[j][0]
                action[j] = batch_data[j][1]
                reward[j] = batch_data[j][2]
                next_state[j] = batch_data[j][3]
                done[j] = batch_data[j][4]

            self.sess.run(self.critic_opt, feed_dict={self.state_ph: state,
                                                      self.action_ph: action,
                                                      self.reward_ph: reward,
                                                      self.next_state_ph: next_state,
                                                      self.done_ph: done})

            if ((i + 1) % self.policy_delay) == 0:
                self.sess.run(self.actor_opt, feed_dict={self.state_ph: state})
                self.sess.run([self.critic_target_update, self.actor_target_update])

            loss = self.sess.run(self.actor_loss, feed_dict={self.state_ph: state})
            loss1 = self.sess.run(self.critic_loss, feed_dict={self.state_ph: state,
                                                      self.action_ph: action,
                                                      self.reward_ph: reward,
                                                      self.next_state_ph: next_state,
                                                      self.done_ph: done})
        return loss, loss1

    # 주어진 상태에 대해 actor 모델에의한 행동을 출력하는 함수
    def get_action(self, state):

        action = self.sess.run(self.actor_model, feed_dict={self.state_ph: state[None, :]})[0]

        return action

    # 모델을 저장하는 함수
    def save_model(self):
        self.saver.save(self.sess, './save_folder/model.ckpt')

    # 모델을 로드하는 함수
    def load_model(self):
        self.saver.restore(self.sess, './save_folder/model.ckpt')


# ddpg기반 학습을 실행
def learn():

    with open('./test_state.pckl', 'rb') as f:
        test_state = pickle.load(f)

    divide_idx = int(hp.state_dim / 2)

    agent = ddpg(active_writer=True)

    env = Environment()

    random_action_prob = hp.random_action_prob
    action_noise_std = hp.action_noise_std

    global_step = 0

    best_success = 0.0

    h_score, h_time_step, h_length, h_success = [], [], [], []
    h_loss, h_loss1 = [], []
    # 다양한 에피소드를 반복하여 데이터 수집
    for e in range(10000000):

        state = env.reset()

        done = False

        score = 0.0

        local_memory = []

        # 하나의 에피소드 시작
        while (done is False) & (env.time_step < hp.max_time_step):  # max_step : 50

            global_step += 1

            # 학습 초기 단계에서는 완전한 무작위 행동을 통해 데이터 수집
            # 일정 시간이 지난 후 0.1의 확률로 무작위 행동, 0.9의 확률로 잡음이 섞인 actor 모델의 행동을 취함
            if global_step <= 10000:
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
                    action = agent.get_action(state) + np.random.normal(loc=0.0, scale=action_noise_std, size=[hp.action_dim])
                    action = np.clip(action, -1.0, 1.0)

            # 행동을 취하고, 다음 상태 / 보상 / 종료 여부를 리턴 받음
            next_state, reward, done = env.step(action)

            agent.memory.append((state, action, reward, next_state, float(done)))
            local_memory.append((state, action, reward, next_state, float(done)))

            state = next_state

            score += reward
        print("e : ", e, " score : ", score)

        # HER algorithm
        for h in range(env.time_step):
            state, action, reward, next_state, done = local_memory[h]

            state = state.copy()
            next_state = next_state.copy()

            for k in range(hp.her_k):   # her_k = 4,  divide_idx = 6/2 = 3

                # export random state
                future_idx = np.random.randint(low=h, high=env.time_step)
                _, _, _, future_next_state, _ = local_memory[future_idx]

                # if agent did not moved
                if (np.linalg.norm(state[:divide_idx] - next_state[:divide_idx]) == 0) & (np.linalg.norm(action) > 0):
                    continue

                # change the goal as exported state
                state[divide_idx:] = future_next_state[:divide_idx]
                next_state[divide_idx:] = future_next_state[:divide_idx]

                #
                if np.linalg.norm(next_state[:divide_idx] - next_state[divide_idx:]) <= env.goal_bound:
                    reward = 0.0
                    done = 1.0
                else:
                    reward = -1.0
                    done = 0.0

                agent.memory.append((state, action, reward, next_state, done))

        # 메모리에 데이터가 어느 정도 생성되면 매 에피소드 종료 후 학습
        if global_step >= 5000:
            loss, loss1 = agent.train_model(env.time_step)
            h_loss.append(loss)
            h_loss1.append(loss1)


        h_score.append(score)
        h_time_step.append(env.time_step)
        h_length.append(env.length)
        h_success.append(env.success)



        # 일정 에피소드 마다 성능과 관련된 변수들 기록
        if ((e + 1) % 10) == 0 and global_step >= 5000:
            agent.run_summary(np.mean(h_score), np.mean(h_time_step), np.mean(h_length), np.mean(h_success), e + 1, np.mean(h_loss), np.mean(h_loss1))
            h_score, h_time_step, h_length, h_success = [], [], [], []
            h_loss, h_loss1 = [], []
        # 일정 에피소드 마다 모델 저장
        if ((e + 1) % 5000) == 0:

            test_length, test_success = [], []

            for test_idx in range(500):#range(len(test_state)):

                state = env.eval_reset(test_state[test_idx])

                done = False

                while (done is False) & (env.time_step < hp.max_time_step):

                    action = agent.get_action(state)

                    next_state, _, done = env.step(action)

                    state = next_state

                test_length.append(env.length)
                test_success.append(env.success)

            test_length = np.mean(test_length)
            test_success = np.mean(test_success)

            agent.run_summary_test(test_length, test_success, e + 1)

            if test_success > best_success:
                best_success = test_success
                agent.save_model()


def save_mat():

    import scipy.io

    agent = ddpg(active_writer=False)

    agent.load_model()

    with agent.graph.as_default():
        actor_var = tf.trainable_variables(scope='actor_model')

    actor_var_result = agent.sess.run(actor_var)

    obj = np.zeros((len(actor_var_result),), dtype=np.object)

    for i in range(len(actor_var_result)):
        obj[i] = actor_var_result[i]

    scipy.io.savemat('./actor.mat', mdict={'actor_var': obj})


if __name__ == '__main__':

    learn()
    #save_mat()












