import numpy as np
from hyper_parameter_2 import hp


class Environment(object):

    # 2자유도 로봇에 대한 configuration space를 단순화 시킨 환경

    def __init__(self):

        self.step_size = hp.step_size

        self.goal_bound = hp.goal_bound

        self.margin = hp.margin

        self.joint_max = hp.joint_max
        self.joint_min = hp.joint_min
        self.joint_range = self.joint_max - self.joint_min

        self.obstacle = self.build_obstacle()

        self.env_noise = hp.env_noise

        self.c_check_acc = hp.c_check_acc

        self.dim = int(hp.state_dim / 2) #dimension of configuration space = 6D

        # self.reset()과 self.step()에서 계산할 변수
        # location = 현재 위치
        # goal = 목표 위치
        # state = 상태
        # reward = 보상
        # done = 에피소드 종료 여부
        self.location = None
        self.goal = None

        self.state = None
        self.reward = None
        self.done = None

        # 하나의 에피소드를 진행하면서 경로가 잘 생성되는지, 얼마나 짧은 경로인지 등을 측정하기 위한 기록용 변수
        # time_step = self.step()의 호출 횟수를 카운트. 따라서 하나의 에피소드에 소요된 타임 스텝을 기록
        # success = 하나의 에피소드가 종료될 때, 목표위치에 도달한 경로가 생성됐는지를 기록
        # length = 하나의 에피소드가 종료될 때, 전체 경로의 길이를 기록
        self.time_step = None
        self.success = None
        self.length = None

    def reset(self):

        # 기록용 변수들 초기화
        self.time_step = 0
        self.success = False
        self.length = 0

        # 현재 위치를 무작위로 생성
        # 1.한계 범위를 벗어나지 않고, 2.충돌이 일어나지 않는 위치를 뽑을때까지 반복
        self.location = (self.joint_range - 2.0 * self.margin) * np.random.rand(self.dim) + self.joint_min + self.margin
        while self.collision_init_check(self.location):#sample free
            self.location = (self.joint_range - 2.0 * self.margin) * np.random.rand(self.dim) + self.joint_min + self.margin

        # 목표 위치를 무작위로 생성
        # 1.한계 범위를 벗어나지 않고, 2.충돌이 일어나지 않으며, 3.현재 위치와 너무 근접하지 않은 위치를 뽑을때까지 반복
        self.goal = (self.joint_range - 2.0 * self.margin) * np.random.rand(self.dim) + self.joint_min + self.margin
        while self.collision_init_check(self.goal) | self.goal_check(self.location):#sample free
            self.goal = (self.joint_range - 2.0 * self.margin) * np.random.rand(self.dim) + self.joint_min + self.margin

        # 상태 정의 == 현재 위치 및 목표 위치
        self.state = np.concatenate((self.location, self.goal), axis=0)

        return self.state.copy()

    def step(self, action):

        action_length = np.linalg.norm(action)
        if action_length > 1.0:
            action = action / action_length

        # 기록용 변수 업데이트
        self.time_step += 1
        self.length += self.step_size * np.linalg.norm(action)

        # 현재 위치에 대해 취해진 행동으로 다음 위치 계산
        next_location = self.location + self.step_size * action + self.env_noise * np.random.randn(self.dim)

        # 다음 위치에 따른 상태, 보상, 목표 달성 여부 등을 계산
        # (1) 먼저 현재 위치와 다음 위치를 연결하는 직선 경로가 충돌이 발생하는지, 한계 범위를 넘어서는지를 체크
        # (2) (1)에서 체크에 걸릴경우 이동은 일어나지 않으며 보상은 -1, 목표 달성 여부는 False
        # (3) (1)에서 체크를 통과할 경우 이동을 실행. location 변수를 업데이트.
        # (4) 이동된 위치가 목표 위치에 도달했는지를 체크
        # (5) (4)에서 체크에 걸릴경우 보상은 0, 목표 달성 여부는 True, 에피소드 종료
        # (6) (4)에서 체크를 통과할 경우 보상은 -1, 목표 달성 여부는 False
        if self.collision_path_check(self.location, next_location) | self.range_check(next_location):#range_check=check inside max min joint
            self.reward = -1.0
            self.done = False
        else:
            self.location = next_location
            if self.goal_check(self.location):#check the state is near goal state(define success)
                self.reward = 0.0
                self.done = True
                self.success = True
            else:
                self.reward = -1.0
                self.done = False

        # 상태 업데이트
        self.state = np.concatenate((self.location, self.goal), axis=0)

        return self.state.copy(), self.reward, self.done, 0

    def eval_reset(self, state):

        self.time_step = 0
        self.success = False
        self.length = 0

        self.state = state.copy()

        self.location = state[:self.dim].copy()
        self.goal = state[self.dim:].copy()

        return self.state.copy()

    def build_obstacle(self):

        obstacle = []

        obstacle.append(
            {'c': np.array([250., 180., 500.]).reshape(3, 1), 'e': 0.5 * np.array([50., 50., 1000.]).reshape(3, 1),
             'u': self.ZYZ_Rmaxtrix(0., 0., 0.)})

#        obstacle.append(
#            {'c': np.array([-150., 200., 50.]).reshape(3, 1), 'e': 0.5 * np.array([100., 250., 100.]).reshape(3, 1),
#             'u': self.ZYZ_Rmaxtrix(0., 0., 0.)})

#        obstacle.append(
#            {'c': np.array([200., -200., 112.5]).reshape(3, 1), 'e': 0.5 * np.array([200., 100., 225.]).reshape(3, 1),
#             'u': self.ZYZ_Rmaxtrix(0., 0., 0.)})

#        obstacle.append(
#            {'c': np.array([-50., 300., 425.]).reshape(3, 1), 'e': 0.5 * np.array([500., 500., 50.]).reshape(3, 1),
#             'u': self.ZYZ_Rmaxtrix(0., 0., 0.)})

        obstacle.append({'c': np.array([0., 0., -2.]).reshape(3, 1), 'e': np.array([800., 800., 1.]).reshape(3, 1),
                         'u': self.ZYZ_Rmaxtrix(0., 0., 0.)})

        return obstacle

    def ZYZ_Rmaxtrix(self, zpi, yth, zpsi):

        R = np.zeros([3, 3])

        R[0, 0] = np.cos(zpi) * np.cos(yth) * np.cos(zpsi) - np.sin(zpi) * np.sin(zpsi)
        R[0, 1] = -np.cos(zpi) * np.cos(yth) * np.sin(zpsi) - np.sin(zpi) * np.cos(zpsi)
        R[0, 2] = np.cos(zpi) * np.sin(yth)

        R[1, 0] = np.sin(zpi) * np.cos(yth) * np.cos(zpsi) + np.cos(zpi) * np.sin(zpsi)
        R[1, 1] = -np.sin(zpi) * np.cos(yth) * np.sin(zpsi) + np.cos(zpi) * np.cos(zpsi)
        R[1, 2] = np.sin(zpi) * np.sin(yth)

        R[2, 0] = -np.sin(yth) * np.cos(zpsi)
        R[2, 1] = np.sin(yth) * np.sin(zpsi)
        R[2, 2] = np.cos(yth)

        return R

    def collision_init_check(self, check_point):

        x = []

        x.append(check_point + np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        x.append(check_point + np.array([self.margin, 0.0, 0.0, 0.0, 0.0, 0.0]))
        x.append(check_point + np.array([-self.margin, 0.0, 0.0, 0.0, 0.0, 0.0]))
        x.append(check_point + np.array([0.0, self.margin, 0.0, 0.0, 0.0, 0.0]))
        x.append(check_point + np.array([0.0, -self.margin, 0.0, 0.0, 0.0, 0.0]))
        x.append(check_point + np.array([0.0, 0.0, self.margin, 0.0, 0.0, 0.0]))
        x.append(check_point + np.array([0.0, 0.0, -self.margin, 0.0, 0.0, 0.0]))
        x.append(check_point + np.array([0.0, 0.0, 0.0, self.margin, 0.0, 0.0]))
        x.append(check_point + np.array([0.0, 0.0, 0.0, -self.margin, 0.0, 0.0]))
        x.append(check_point + np.array([0.0, 0.0, 0.0, 0.0, self.margin, 0.0]))
        x.append(check_point + np.array([0.0, 0.0, 0.0, 0.0, -self.margin, 0.0]))
        x.append(check_point + np.array([0.0, 0.0, 0.0, 0.0, 0.0, self.margin]))
        x.append(check_point + np.array([0.0, 0.0, 0.0, 0.0, 0.0, -self.margin]))
    

        flag = False

        for i in range(13):
            if self.collision_check(x[i]):
                flag = True
                break

        return flag



    # a지점과 b지점을 연결하는 직선 경로에 대해 충돌이 발생하는지를 체크하는 함수
    # 방법은 a지점과 b지점을 연결하는 직선 경로를 n등분하여 각각의 지점에 대해 충돌 여부를 검사
    # 모두 충돌이 없을 경우 직선 경로는 충돌 없음. 한 지점이라도 충돌이 발생할 경우 직선 경로는 충돌 존재
    def collision_path_check(self, point_a, point_b):

        flag = False

        p = np.arange(start=0.0, stop=1.0, step=self.c_check_acc)

        check_points = np.matmul(p[:, None], point_a[None, :]) + np.matmul(1.0 - p[:, None], point_b[None, :])

        for i in range(check_points.shape[0]):
            if self.collision_check(check_points[i]):
                flag = True
                break

        return flag

    # 한 지점에 대해 충돌이 발생하는지를 체크하는 함수 Collision free node
    def collision_check(self, check_point):

        flag = False
        
        check_point1 = np.zeros([3])#joint robot1
        check_point2 = np.array([0.0, 0.0, 0.0])#joint robot2
        check_point1[0] = check_point[0]
        check_point1[1] = check_point[1]
        check_point1[2] = check_point[2]
        check_point2[0] = check_point[3]
        check_point2[1] = check_point[4]
        check_point2[2] = check_point[5]
        
        kine1 = self.kine_rmx52_1(check_point1)
        kine2 = self.kine_rmx52_2(check_point2)

        _, obb1, check_matrix1 = self.obb_rmx52_1(kine1)
        _, obb2, check_matrix2 = self.obb_rmx52_2(kine2)

        self_collision1, _, _ = self.self_collision_check(obb1, check_matrix1)
        self_collision2, _, _ = self.self_collision_check(obb2, check_matrix2)

        env_collision1, _, _ = self.env_collision_check(self.obstacle, obb1)
        env_collision2, _, _ = self.env_collision_check(self.obstacle, obb2)
        env_collision3, _, _ = self.env_collision_check(obb2, obb1)

        if self_collision1 | self_collision2 | env_collision1 | env_collision2 | env_collision3:
            flag = True

        return flag

    # 1. forward kinematics를 통해 로봇의 각 obb의 world 좌표계 기준 값을 구함
    def kine_rmx52_1(self, point):

        DOF = 4

        params = np.zeros([4, 4])

        LinkMatrices = []
        offset = []
        Rot = []
        q = []

        for i in range(DOF):
            LinkMatrices.append(params)
            offset.append(params)
            Rot.append(params)
            q.append(0)

        q[0] = point[0]
        q[1] = point[1]
        q[2] = point[2]
        q[3] = 0.0

        offset[0] = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 36.], [0., 0., 0., 1.]])
        offset[1] = np.array([[1., 0., 0., 0.], [0., 0., 1., 0.], [0., -1., 0., 40.5], [0., 0., 0., 1.]])
        offset[2] = np.array([[1., 0., 0., 128.], [0., 1., 0., 24.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
        offset[3] = np.array([[1., 0., 0., 124.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])

        for i in range(DOF):
            Rot[i] = np.eye(4)
            Rot[i][0, 0] = np.cos(q[i])
            Rot[i][0, 1] = -np.sin(q[i])
            Rot[i][1, 0] = np.sin(q[i])
            Rot[i][1, 1] = np.cos(q[i])
            LinkMatrices[i] = np.matmul(offset[i], Rot[i])

        return self.forward_kine(DOF, LinkMatrices)
    
    # 1. forward kinematics를 통해 로봇의 각 obb의 world 좌표계 기준 값을 구함
    def kine_rmx52_2(self, point):

        DOF = 4

        params = np.zeros([4, 4])

        LinkMatrices = []
        offset = []
        Rot = []
        q = []

        for i in range(DOF):
            LinkMatrices.append(params)
            offset.append(params)
            Rot.append(params)
            q.append(0)

        q[0] = point[0]
        q[1] = point[1]
        q[2] = point[2]
        q[3] = 0.0

        offset[0] = np.array([[1., 0., 0., 0.], [0., 1., 0., 350.], [0., 0., 1., 36.], [0., 0., 0., 1.]])
        offset[1] = np.array([[1., 0., 0., 0.], [0., 0., 1., 0.], [0., -1., 0., 40.5], [0., 0., 0., 1.]])
        offset[2] = np.array([[1., 0., 0., 128.], [0., 1., 0., 24.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
        offset[3] = np.array([[1., 0., 0., 124.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])

        for i in range(DOF):
            Rot[i] = np.eye(4)
            Rot[i][0, 0] = np.cos(q[i])
            Rot[i][0, 1] = -np.sin(q[i])
            Rot[i][1, 0] = np.sin(q[i])
            Rot[i][1, 1] = np.cos(q[i])
            LinkMatrices[i] = np.matmul(offset[i], Rot[i])

        return self.forward_kine(DOF, LinkMatrices)

    def forward_kine(self, DOF, LinkMatrices):

        params = np.zeros([4, 4])
        kine = []

        for i in range(DOF):
            kine.append(params)

        kine[0] = LinkMatrices[0]

        for i in range(DOF-1):
            kine[i+1] = np.matmul(kine[i], LinkMatrices[i+1])

        return kine

    # 2. 로봇에 대한 obb 생성
    def obb_rmx52_1(self, kine):

        obb_num = 6

        check_matrix = np.eye(obb_num)

        obb_index = [-1, 0, 1, 1, 2, 3]

        obb = []

        for i in range(obb_num):
            obb.append({'c': 0, 'u': 0, 'e': 0})
            obb[i]['u'] = np.eye(3)

        obb[0]['c'] = np.array([-12., 0., 17.]).reshape(3, 1)
        obb[0]['e'] = 0.5 * np.array([46.5, 28.5, 34.]).reshape(3, 1)

        obb[1]['c'] = np.array([0., 0., 27.]).reshape(3, 1)
        obb[1]['e'] = 0.5 * np.array([28.5, 37., 54.]).reshape(3, 1)

        obb[2]['c'] = np.array([65., 0., 0.]).reshape(3, 1)
        obb[2]['e'] = 0.5 * np.array([130., 28., 35.]).reshape(3, 1)

        obb[3]['c'] = np.array([125., 13., 0.]).reshape(3, 1)
        obb[3]['e'] = 0.5 * np.array([34., 57., 37.]).reshape(3, 1)

        obb[4]['c'] = np.array([63., 0., 0.]).reshape(3, 1)
        obb[4]['e'] = 0.5 * np.array([146., 28., 35.]).reshape(3, 1)

        obb[5]['c'] = np.array([73., -5., 0.]).reshape(3, 1)
        obb[5]['e'] = 0.5 * np.array([147., 68., 129.]).reshape(3, 1)

        for i in range(1, obb_num):
            j = obb_index[i]
            t = np.matmul(kine[j], np.concatenate((obb[i]['c'], np.array([[1.]])), axis=0))
            obb[i]['c'] = t[0:3, :]
            obb[i]['u'] = kine[j][0:3, 0:3]

        check_matrix[0, 1] = 1
        check_matrix[1, 0] = 1
        check_matrix[1, 2] = 1
        check_matrix[2, 1] = 1
        check_matrix[2, 3] = 1
        check_matrix[3, 2] = 1
        check_matrix[3, 4] = 1
        check_matrix[4, 3] = 1
        check_matrix[2, 4] = 1
        check_matrix[4, 2] = 1
        check_matrix[4, 5] = 1
        check_matrix[5, 4] = 1

        return obb_num, obb, check_matrix
    
    # 2. 로봇에 대한 obb 생성
    def obb_rmx52_2(self, kine):

        obb_num = 6

        check_matrix = np.eye(obb_num)

        obb_index = [-1, 0, 1, 1, 2, 3]

        obb = []

        for i in range(obb_num):
            obb.append({'c': 0, 'u': 0, 'e': 0})
            obb[i]['u'] = np.eye(3)

        obb[0]['c'] = np.array([-12., 350., 17.]).reshape(3, 1)
        obb[0]['e'] = 0.5 * np.array([46.5, 28.5, 34.]).reshape(3, 1)

        obb[1]['c'] = np.array([0., 0., 27.]).reshape(3, 1)
        obb[1]['e'] = 0.5 * np.array([28.5, 37., 54.]).reshape(3, 1)

        obb[2]['c'] = np.array([65., 0., 0.]).reshape(3, 1)
        obb[2]['e'] = 0.5 * np.array([130., 28., 35.]).reshape(3, 1)

        obb[3]['c'] = np.array([125., 13., 0.]).reshape(3, 1)
        obb[3]['e'] = 0.5 * np.array([34., 57., 37.]).reshape(3, 1)

        obb[4]['c'] = np.array([63., 0., 0.]).reshape(3, 1)
        obb[4]['e'] = 0.5 * np.array([146., 28., 35.]).reshape(3, 1)

        obb[5]['c'] = np.array([73., -5., 0.]).reshape(3, 1)
        obb[5]['e'] = 0.5 * np.array([147., 68., 129.]).reshape(3, 1)

        for i in range(1, obb_num):
            j = obb_index[i]
            t = np.matmul(kine[j], np.concatenate((obb[i]['c'], np.array([[1.]])), axis=0))
            obb[i]['c'] = t[0:3, :]
            obb[i]['u'] = kine[j][0:3, 0:3]

        check_matrix[0, 1] = 1
        check_matrix[1, 0] = 1
        check_matrix[1, 2] = 1
        check_matrix[2, 1] = 1
        check_matrix[2, 3] = 1
        check_matrix[3, 2] = 1
        check_matrix[3, 4] = 1
        check_matrix[4, 3] = 1
        check_matrix[2, 4] = 1
        check_matrix[4, 2] = 1
        check_matrix[4, 5] = 1
        check_matrix[5, 4] = 1

        return obb_num, obb, check_matrix

    # 3. 로봇의 obb간의 충돌 검사
    def self_collision_check(self, obb, check_matrix):

        collision = False

        d = check_matrix.shape[0]

        collision_list = np.zeros([d, 1])
        collision_matrix = np.zeros([d, d])

        for i in range(d):
            for j in range(d):
                if check_matrix[i, j] == 0:
                    if self.test_obbobb(obb[i], obb[j]):
                        collision_matrix[i, j] = 1
                        collision_matrix[j, i] = 1
                        collision_list[i, 0] = 1
                        collision_list[j, 0] = 1
                        collision = True
                        break

                    check_matrix[i, j] = 1
                    check_matrix[j, i] = 1

            if collision:
                break

        return collision, collision_list, collision_matrix

    # 4. 환경의 obb와 로봇의 obb간의 충돌 검사
    def env_collision_check(self, obb_env, obb_robot):

        # nEnv = np.shape(env)
        # nRobot = np.shape(robot)

        # env = copy.deepcopy(env_)

        nEnv = len(obb_env)
        nRobot = len(obb_robot)

        collision = False

        env_list = np.zeros([1, nEnv])
        robot_list = np.zeros([1, nRobot])

        for i in range(nEnv):
            for j in range(nRobot):
                if self.test_obbobb(obb_env[i], obb_robot[j]):
                    collision = True
                    robot_list[0, j] = 1
                    env_list[0, i] = 1
                    break
            if collision:
                break

        return collision, env_list, robot_list

    def test_obbobb(self, a, b):
        result = True

        if np.linalg.norm(a['c'] - b['c']) > np.linalg.norm(a['e']) + np.linalg.norm(b['e']):
            result = False

        R = np.zeros([3, 3])
        absR = np.zeros([3, 3])

        if result:

            for i in range(3):
                for j in range(3):
                    R[i, j] = np.inner(a['u'][:, i], b['u'][:, j])

            at = b['c'] - a['c']
            t = np.array([np.inner(at[:, 0], a['u'][:, 0]), np.inner(at[:, 0], a['u'][:, 1]), np.inner(at[:, 0], a['u'][:, 2])]).reshape(3, 1)

            for i in range(3):
                for j in range(3):
                    absR[i, j] = np.abs(R[i, j]) + 1e-6

            # L=A0, L=A1, L=A2
            for i in range(3):
                ra = a['e'][i, 0]
                rb = b['e'][0, 0] * absR[i, 0] + b['e'][1, 0] * absR[i, 1] + b['e'][2, 0] * absR[i, 2]

                if np.abs(t[i, 0]) > (ra + rb):
                    result = False
                    break

            # L=B0, L=B1, L=B2
            if result:
                for i in range(3):
                    ra = a['e'][0, 0] * absR[0, i] + a['e'][1, 0] * absR[1, i] + a['e'][2, 0] * absR[2, i]
                    rb = b['e'][i, 0]
                    chk = t[0, 0] * R[0, i] + t[1, 0] * R[1, i] + t[2, 0] * R[2, i]

                    if np.abs(chk) > (ra + rb):
                        result = False
                        break

            # L=A0 X B0
            if result:
                ra = a['e'][1, 0] * absR[2, 0] + a['e'][2, 0] * absR[1, 0]
                rb = b['e'][1, 0] * absR[0, 2] + b['e'][2, 0] * absR[0, 1]
                chk = t[2, 0] * R[1, 0] - t[1, 0] * R[2, 0]

                if np.abs(chk) > (ra + rb):
                    result = False

            # L=A0 X B1
            if result:
                ra = a['e'][1, 0] * absR[2, 1] + a['e'][2, 0] * absR[1, 1]
                rb = b['e'][0, 0] * absR[0, 2] + b['e'][2, 0] * absR[0, 0]
                chk = t[2, 0] * R[1, 1] - t[1, 0] * R[2, 1]

                if np.abs(chk) > (ra + rb):
                    result = False

            # L=A0 X B2
            if result:
                ra = a['e'][1, 0] * absR[2, 2] + a['e'][2, 0] * absR[1, 2]
                rb = b['e'][0, 0] * absR[0, 1] + b['e'][1, 0] * absR[0, 0]
                chk = t[2, 0] * R[1, 2] - t[1, 0] * R[2, 2]

                if np.abs(chk) > (ra + rb):
                    result = False

            # L=A1 X B0
            if result:
                ra = a['e'][0, 0] * absR[2, 0] + a['e'][2, 0] * absR[0, 0]
                rb = b['e'][1, 0] * absR[1, 2] + b['e'][2, 0] * absR[1, 1]
                chk = t[0, 0] * R[2, 0] - t[2, 0] * R[0, 0]

                if np.abs(chk) > (ra + rb):
                    result = False

            # L=A1 X B1
            if result:
                ra = a['e'][0, 0] * absR[2, 1] + a['e'][2, 0] * absR[0, 1]
                rb = b['e'][0, 0] * absR[1, 2] + b['e'][2, 0] * absR[1, 0]
                chk = t[0, 0] * R[2, 1] - t[2, 0] * R[0, 1]

                if np.abs(chk) > (ra + rb):
                    result = False

            # L=A1 X B2
            if result:
                ra = a['e'][0, 0] * absR[2, 2] + a['e'][2, 0] * absR[0, 2]
                rb = b['e'][0, 0] * absR[1, 1] + b['e'][1, 0] * absR[1, 0]
                chk = t[0, 0] * R[2, 2] - t[2, 0] * R[0, 2]

                if np.abs(chk) > (ra + rb):
                    result = False

            # L=A2 X B0
            if result:
                ra = a['e'][0, 0] * absR[1, 0] + a['e'][1, 0] * absR[0, 0]
                rb = b['e'][1, 0] * absR[2, 2] + b['e'][2, 0] * absR[2, 1]
                chk = t[1, 0] * R[0, 0] - t[0, 0] * R[1, 0]

                if np.abs(chk) > (ra + rb):
                    result = False

            # L=A2 X B1
            if result:
                ra = a['e'][0, 0] * absR[1, 1] + a['e'][1, 0] * absR[0, 1]
                rb = b['e'][0, 0] * absR[2, 2] + b['e'][2, 0] * absR[2, 0]
                chk = t[1, 0] * R[0, 1] - t[0, 0] * R[1, 1]

                if np.abs(chk) > (ra + rb):
                    result = False

            # L=A2 X B2
            if result:
                ra = a['e'][0, 0] * absR[1, 2] + a['e'][1, 0] * absR[0, 2]
                rb = b['e'][0, 0] * absR[2, 1] + b['e'][1, 0] * absR[2, 0]
                chk = t[1, 0] * R[0, 2] - t[0, 0] * R[1, 2]

                if np.abs(chk) > (ra + rb):
                    result = False

        return result

    # 한 지점에 대해 목표 위치에 도달했는지를 체크하는 함수
    def goal_check(self, check_point):

        if np.linalg.norm(check_point - self.goal) <= self.goal_bound:
            flag = True
        else:
            flag = False

        return flag

    # 한 지점에 대해 한계 범위를 넘어섰는지를 체크하는 함수
    def range_check(self, check_point):

        if (check_point >= self.joint_max).any() | (check_point <= self.joint_min).any():
            flag = True
        else:
            flag = False

        return flag
