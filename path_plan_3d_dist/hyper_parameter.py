import numpy as np


class hp(object):

    # env
    joint_max = np.array([140.0, -45.0, 150.0]) * np.pi / 180.0
    joint_min = np.array([-140.0, -180.0, 45.0]) * np.pi / 180.0
    step_size = 0.05 * (np.prod(joint_max - joint_min)**(1.0/3.0))
    goal_bound = 0.2 * step_size
    margin = 0.0
    env_noise = 0.002
    c_check_acc = 0.2
    state_dim = 6
    action_dim = 3

    # agent
    lr_actor = 0.001
    lr_critic = 0.001
    tau = 0.005
    actor_l2 = 1.0
    her_k = 4
    max_time_step = 50

    target_noise_std = 0.2
    target_noise_clip = 0.5
    policy_delay = 2
    discount_factor = 0.98
    batch_size = 512
    memory_size = 1000000
    random_action_prob = 0.1
    action_noise_std = 0.1




