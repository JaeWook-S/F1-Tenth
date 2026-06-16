# # MIT License
#
# # Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng
#
# # Permission is hereby granted, free of charge, to any person obtaining a copy
# # of this software and associated documentation files (the "Software"), to deal
# # in the Software without restriction, including without limitation the rights
# # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# # copies of the Software, and to permit persons to whom the Software is
# # furnished to do so, subject to the following conditions:
#
# # The above copyright notice and this permission notice shall be included in all
# # copies or substantial portions of the Software.
#
# # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# # SOFTWARE.
#
# '''
# Author: Hongrui Zheng
# '''
#
# # gym imports
# import gym
# from gym import error, spaces, utils
# from gym.utils import seeding
#
# # base classes
# from f110_gym.envs.base_classes import Simulator
#
# # others
# import numpy as np
# import os
# import time
#
# # gl
# import pyglet
# pyglet.options['debug_gl'] = False
# from pyglet import gl
#
# # constants
#
# # rendering
# VIDEO_W = 600
# VIDEO_H = 400
# WINDOW_W = 1000
# WINDOW_H = 800
#
# class F110Env(gym.Env, utils.EzPickle):
#     """
#     OpenAI gym environment for F1TENTH
#
#     Env should be initialized by calling gym.make('f110_gym:f110-v0', **kwargs)
#
#     Args:
#         kwargs:
#             seed (int, default=12345): seed for random state and reproducibility
#
#             map (str, default='vegas'): name of the map used for the environment. Currently, available environments include: 'berlin', 'vegas', 'skirk'. You could use a string of the absolute path to the yaml file of your custom map.
#
#             map_ext (str, default='png'): image extension of the map image file. For example 'png', 'pgm'
#
#             params (dict, default={'mu': 1.0489, 'C_Sf':, 'C_Sr':, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch':7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}): dictionary of vehicle parameters.
#             mu: surface friction coefficient
#             C_Sf: Cornering stiffness coefficient, front
#             C_Sr: Cornering stiffness coefficient, rear
#             lf: Distance from center of gravity to front axle
#             lr: Distance from center of gravity to rear axle
#             h: Height of center of gravity
#             m: Total mass of the vehicle
#             I: Moment of inertial of the entire vehicle about the z axis
#             s_min: Minimum steering angle constraint
#             s_max: Maximum steering angle constraint
#             sv_min: Minimum steering velocity constraint
#             sv_max: Maximum steering velocity constraint
#             v_switch: Switching velocity (velocity at which the acceleration is no longer able to create wheel spin)
#             a_max: Maximum longitudinal acceleration
#             v_min: Minimum longitudinal velocity
#             v_max: Maximum longitudinal velocity
#             width: width of the vehicle in meters
#             length: length of the vehicle in meters
#
#             num_agents (int, default=2): number of agents in the environment
#
#             timestep (float, default=0.01): physics timestep
#
#             ego_idx (int, default=0): ego's index in list of agents
#     """
#     metadata = {'render.modes': ['human', 'human_fast']}
#
#     def __init__(self, **kwargs):
#         # kwargs extraction
#         try:
#             self.seed = kwargs['seed']
#         except:
#             self.seed = 12345
#         try:
#             self.map_name = kwargs['map']
#             # different default maps
#             if self.map_name == 'berlin':
#                 self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/berlin.yaml'
#             elif self.map_name == 'skirk':
#                 self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/skirk.yaml'
#             elif self.map_name == 'levine':
#                 self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/levine.yaml'
#             else:
#                 self.map_path = self.map_name + '.yaml'
#         except:
#             self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/vegas.yaml'
#
#         try:
#             self.map_ext = kwargs['map_ext']
#         except:
#             self.map_ext = '.png'
#
#         try:
#             self.params = kwargs['params']
#         except:
#             self.params = {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch': 7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}
#
#         # simulation parameters
#         try:
#             self.num_agents = kwargs['num_agents']
#         except:
#             self.num_agents = 2
#
#         try:
#             self.timestep = kwargs['timestep']
#         except:
#             self.timestep = 0.01
#
#         # default ego index
#         try:
#             self.ego_idx = kwargs['ego_idx']
#         except:
#             self.ego_idx = 0
#
#         # radius to consider done
#         self.start_thresh = 0.5  # 10cm
#
#         # env states
#         self.poses_x = []
#         self.poses_y = []
#         self.poses_theta = []
#         self.collisions = np.zeros((self.num_agents, ))
#         self.collision_idx = np.zeros((self.num_agents, ))
#
#         # loop completion
#         self.near_start = True
#         self.num_toggles = 0
#
#         # race info
#         self.lap_times = np.zeros((self.num_agents, ))
#         self.lap_counts = np.zeros((self.num_agents, ))
#         self.current_time = 0.0
#
#         # finish line info
#         self.num_toggles = 0
#         self.near_start = True
#         self.near_starts = np.array([True]*self.num_agents)
#         self.toggle_list = np.zeros((self.num_agents,))
#         self.start_xs = np.zeros((self.num_agents, ))
#         self.start_ys = np.zeros((self.num_agents, ))
#         self.start_thetas = np.zeros((self.num_agents, ))
#         self.start_rot = np.eye(2)
#
#         # initiate stuff
#         self.sim = Simulator(self.params, self.num_agents, self.seed)
#         self.sim.set_map(self.map_path, self.map_ext)
#
#         # rendering
#         self.renderer = None
#         self.current_obs = None
#
#     def __del__(self):
#         """
#         Finalizer, does cleanup
#         """
#         pass
#
#     def _check_done(self):
#         """
#         Check if the current rollout is done
#
#         Args:
#             None
#
#         Returns:
#             done (bool): whether the rollout is done
#             toggle_list (list[int]): each agent's toggle list for crossing the finish zone
#         """
#
#         # this is assuming 2 agents
#         # TODO: switch to maybe s-based
#         left_t = 2
#         right_t = 2
#
#         poses_x = np.array(self.poses_x)-self.start_xs
#         poses_y = np.array(self.poses_y)-self.start_ys
#         delta_pt = np.dot(self.start_rot, np.stack((poses_x, poses_y), axis=0))
#         temp_y = delta_pt[1,:]
#         idx1 = temp_y > left_t
#         idx2 = temp_y < -right_t
#         temp_y[idx1] -= left_t
#         temp_y[idx2] = -right_t - temp_y[idx2]
#         temp_y[np.invert(np.logical_or(idx1, idx2))] = 0
#
#         dist2 = delta_pt[0,:]**2 + temp_y**2
#         closes = dist2 <= 0.1
#         for i in range(self.num_agents):
#             if closes[i] and not self.near_starts[i]:
#                 self.near_starts[i] = True
#                 self.toggle_list[i] += 1
#             elif not closes[i] and self.near_starts[i]:
#                 self.near_starts[i] = False
#                 self.toggle_list[i] += 1
#             self.lap_counts[i] = self.toggle_list[i] // 2
#             if self.toggle_list[i] < 4:
#                 self.lap_times[i] = self.current_time
#
#         done = np.all(self.collisions > 0) or np.all(self.toggle_list >= 4)
#
#         return done, self.toggle_list >= 4
#
#     def _update_state(self, obs_dict):
#         """
#         Update the env's states according to observations
#
#         Args:
#             obs_dict (dict): dictionary of observation
#
#         Returns:
#             None
#         """
#         self.poses_x = obs_dict['poses_x']
#         self.poses_y = obs_dict['poses_y']
#         self.poses_theta = obs_dict['poses_theta']
#         self.collisions = obs_dict['collisions']
#
#     def step(self, action):
#         """
#         Step function for the gym env
#
#         Args:
#             action (np.ndarray(num_agents, 2))
#
#         Returns:
#             obs (dict): observation of the current step
#             reward (float, default=self.timestep): step reward, currently is physics timestep
#             done (bool): if the simulation is done
#             info (dict): auxillary information dictionary
#         """
#
#         # call simulation step
#         obs = self.sim.step(action)
#         obs['lap_times'] = self.lap_times
#         obs['lap_counts'] = self.lap_counts
#
#         self.current_obs = obs
#
#         # times
#         reward = self.timestep
#         self.current_time = self.current_time + self.timestep
#
#         # update data member
#         self._update_state(obs)
#
#         # check done
#         done, toggle_list = self._check_done()
#         info = {'checkpoint_done': toggle_list}
#
#         return obs, reward, done, info
#
#     def reset(self, poses):
#         """
#         Reset the gym environment by given poses
#
#         Args:
#             poses (np.ndarray (num_agents, 3)): poses to reset agents to
#
#         Returns:
#             obs (dict): observation of the current step
#             reward (float, default=self.timestep): step reward, currently is physics timestep
#             done (bool): if the simulation is done
#             info (dict): auxillary information dictionary
#         """
#         # reset counters and data members
#         self.current_time = 0.0
#         self.collisions = np.zeros((self.num_agents, ))
#         self.num_toggles = 0
#         self.near_start = True
#         self.near_starts = np.array([True]*self.num_agents)
#         self.toggle_list = np.zeros((self.num_agents,))
#
#         # states after reset
#         self.start_xs = poses[:, 0]
#         self.start_ys = poses[:, 1]
#         self.start_thetas = poses[:, 2]
#         self.start_rot = np.array([[np.cos(-self.start_thetas[self.ego_idx]), -np.sin(-self.start_thetas[self.ego_idx])], [np.sin(-self.start_thetas[self.ego_idx]), np.cos(-self.start_thetas[self.ego_idx])]])
#
#         # call reset to simulator
#         self.sim.reset(poses)
#
#         # get no input observations
#         action = np.zeros((self.num_agents, 2))
#         obs, reward, done, info = self.step(action)
#         return obs, reward, done, info
#
#     def update_map(self, map_path, map_ext):
#         """
#         Updates the map used by simulation
#
#         Args:
#             map_path (str): absolute path to the map yaml file
#             map_ext (str): extension of the map image file
#
#         Returns:
#             None
#         """
#         self.sim.set_map(map_path, map_ext)
#
#     def update_params(self, params, index=-1):
#         """
#         Updates the parameters used by simulation for vehicles
#
#         Args:
#             params (dict): dictionary of parameters
#             index (int, default=-1): if >= 0 then only update a specific agent's params
#
#         Returns:
#             None
#         """
#         self.sim.update_params(params, agent_idx=index)
#
#     def render(self, mode='human'):
#         """
#         Renders the environment with pyglet. Use mouse scroll in the window to zoom in/out, use mouse click drag to pan. Shows the agents, the map, current fps (bottom left corner), and the race information near as text.
#
#         Args:
#             mode (str, default='human'): rendering mode, currently supports:
#                 'human': slowed down rendering such that the env is rendered in a way that sim time elapsed is close to real time elapsed
#                 'human_fast': render as fast as possible
#
#         Returns:
#             None
#         """
#         assert mode in ['human', 'human_fast']
#         if self.renderer is None:
#             # first call, initialize everything
#             from f110_gym.envs.rendering import EnvRenderer
#             self.renderer = EnvRenderer(WINDOW_W, WINDOW_H)
#             self.renderer.update_map(self.map_name, self.map_ext)
#         self.renderer.update_obs(self.current_obs)
#         self.renderer.dispatch_events()
#         self.renderer.on_draw()
#         self.renderer.flip()
#         if mode == 'human':
#             time.sleep(0.005)
#         elif mode == 'human_fast':
#             pass

# MIT License
# MIT License

# Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''
Author: Hongrui Zheng
'''

# gym imports
import gym
from gym import error, spaces, utils
from gym.utils import seeding

# base classes
from f110_gym.envs.base_classes import Simulator

# others
import numpy as np
import os
import time

# gl
import pyglet

pyglet.options['debug_gl'] = False
from pyglet import gl

# constants

# rendering
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

globwaypoints = np.genfromtxt(f"/Users/jaewook/Desktop/인하대_학업/4학년1학기/강화학습/f1tenth-riders-quickstart/pkg/src/pkg/maps/Oschersleben_centerline.csv", delimiter=',')

"""
__init__()     -> 환경 초기화
reset()        -> 에피소드 시작
step()         -> 한 step 진행
_check_done()  -> 종료 판단
reward         -> 보상 계산
render()       -> 화면 출력
"""
class DDQN_Best_Env(gym.Env, utils.EzPickle):
    """
    OpenAI gym environment for F1TENTH

    Env should be initialized by calling gym.make('f110_gym:f110-v0', **kwargs)

    Args:
        kwargs:
            seed (int, default=12345): seed for random state and reproducibility

            map (str, default='vegas'): name of the map used for the environment. Currently, available environments include: 'berlin', 'vegas', 'skirk'. You could use a string of the absolute path to the yaml file of your custom map.

            map_ext (str, default='png'): image extension of the map image file. For example 'png', 'pgm'

            params (dict, default={'mu': 1.0489, 'C_Sf':, 'C_Sr':, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch':7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}): dictionary of vehicle parameters.
            mu: surface friction coefficient
            C_Sf: Cornering stiffness coefficient, front -> 앞바퀴 타이어의 코너링 강성 
            C_Sr: Cornering stiffness coefficient, rear -> 뒷바퀴 타이어 코너링 강성
            lf: Distance from center of gravity to front axle -> 무게중심에서 앞바퀴까지 거리
            lr: Distance from center of gravity to rear axle -> 무게중심에서 뒷바퀴까지 거리
            h: Height of center of gravity -> 무게중심 높이
            m: Total mass of the vehicle -> -> 차량 전체 질량
            I: Moment of inertial of the entire vehicle about the z axis -> 차량이 회전하려는 관성
            s_min: Minimum steering angle constraint -> 최소 조향각
            s_max: Maximum steering angle constraint -> 최대 조향각
            sv_min: Minimum steering velocity constraint -> 조향 속도 최소값 (얼마나 빨리 핸들 움직일 수 있는지)
            sv_max: Maximum steering velocity constraint -> 조향 속도 최대값
            v_switch: Switching velocity (velocity at which the acceleration is no longer able to create wheel spin) -> 전환 속도
            a_max: Maximum longitudinal acceleration -> 최대 종방향 가속도 (얼마나 빨리 속도를 올릴 수 있는지)
            v_min: Minimum longitudinal velocity -> 최소 속도
            v_max: Maximum longitudinal velocity -> 최대 속도
            width: width of the vehicle in meters -> 차량 폭
            length: length of the vehicle in meters -> 차량 길이

            num_agents (int, default=2): number of agents in the environment

            timestep (float, default=0.01): physics timestep

            ego_idx (int, default=0): ego's index in list of agents
    """
    metadata = {'render.modes': ['human', 'human_fast']}

    def __init__(self, **kwargs):
        # kwargs extraction
        try:
            self.seed = kwargs['seed']
        except:
            self.seed = 12345
        try:
            self.map_name = kwargs['map']
            # different default maps
            if self.map_name == 'berlin':
                self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/berlin.yaml'
            elif self.map_name == 'skirk':
                self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/skirk.yaml'
            elif self.map_name == 'levine':
                self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/levine.yaml'
            else:
                self.map_path = self.map_name + '.yaml'
        except:
            self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/vegas.yaml'

        try:
            self.map_ext = kwargs['map_ext']
        except:
            self.map_ext = '.png'

        # 차량 dynamics
        try:
            self.params = kwargs['params']
        except:
            self.params = {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074,
                           'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2,
                           'v_switch': 7.319, 'a_max': 9.51, 'v_min': -5.0, 'v_max': 20.0, 'width': 0.31,
                           'length': 0.58}

        # simulation parameters
        try:
            self.num_agents = kwargs['num_agents']
        except:
            self.num_agents = 2

        # 몇초마다 step할 지(physics timestep)
        try:
            self.timestep = kwargs['timestep']
        except:
            self.timestep = 0.01

        # default ego index
        try:
            self.ego_idx = kwargs['ego_idx']
        except:
            self.ego_idx = 0

        # radius to consider done
        self.start_thresh = 0.5  # 10cm

        # env states
        self.poses_x = []
        self.poses_y = []
        self.poses_theta = []
        self.collisions = np.zeros((self.num_agents,))
        # TODO: collision_idx not used yet
        # self.collision_idx = -1 * np.ones((self.num_agents, ))

        # loop completion
        self.near_start = True
        self.num_toggles = 0

        # race info
        self.lap_times = np.zeros((self.num_agents,))
        self.lap_counts = np.zeros((self.num_agents,))
        self.current_time = 0.0

        # finish line info
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True] * self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))
        self.start_xs = np.zeros((self.num_agents,))
        self.start_ys = np.zeros((self.num_agents,))
        self.start_thetas = np.zeros((self.num_agents,))
        self.start_rot = np.eye(2)

        # initiate stuff
        self.sim = Simulator(self.params, self.num_agents, self.seed)
        self.sim.set_map(self.map_path, self.map_ext)

        # rendering
        self.renderer = None
        self.current_obs = None

        # map checkpoint
        self.checklist = np.zeros((15))  # 추가

        self.count = 0

        # ==========================================
        # [수정] Waypoint 방향성 및 진행도 관리를 위한 변수 초기화
        # ==========================================
        # globwaypoints 구조 정의: [x, y, (선택적 속도나 heading)] -> 앞에서 2열만 위치로 사용
        self.waypoints = globwaypoints[:, :2]
        self.num_waypoints = len(self.waypoints)
        
        # 주행 방향이 인덱스 역순(끝 -> 처음)이므로 쉽게 다루기 위해 미리 뒤집어 놓습니다.
        # 이렇게 하면 self.waypoints[0]부터 순방향으로 전진하는 것과 같아집니다.
        self.waypoints = np.flip(self.waypoints, axis=0)
        
        self.closest_idx = 0
        self.prev_dist = None


    def __del__(self):
        """
        Finalizer, does cleanup
        """
        pass

    def _check_done(self):
        """
        Check if the current rollout is done

        Args:
            None

        Returns:
            done (bool): whether the rollout is done
            toggle_list (list[int]): each agent's toggle list for crossing the finish zone
        """

        # this is assuming 2 agents
        # TODO: switch to maybe s-based
        # 출발선 영역 지정 -> 출발선 기준으로 y축 방향 -2~+2 범위를 출발선 근처 영역으로 정의
        left_t = 2
        right_t = 2

        # 현재 위치 -> 출발점 기준 상대 위치로 변환 
        poses_x = np.array(self.poses_x) - self.start_xs
        poses_y = np.array(self.poses_y) - self.start_ys
        # self.start_rot를 곱하여, 에이전트의 위치를 시작 지점의 진행 방향을 기준으로 한 로컬 좌표계로 변환
        delta_pt = np.dot(self.start_rot, np.stack((poses_x, poses_y), axis=0))

        # 경계 밖으로 얼마나 벗어났는지 거리 계산을 위해
        temp_y = delta_pt[1, :]
        idx1 = temp_y > left_t
        idx2 = temp_y < -right_t
        temp_y[idx1] -= left_t
        temp_y[idx2] = -right_t - temp_y[idx2]
        temp_y[np.invert(np.logical_or(idx1, idx2))] = 0

        dist2 = delta_pt[0, :] ** 2 + temp_y ** 2
        closes = dist2 <= 0.1
        for i in range(self.num_agents):
            if closes[i] and not self.near_starts[i]: # 출발선 진입 순간
                self.near_starts[i] = True
                self.toggle_list[i] += 1
            elif not closes[i] and self.near_starts[i]: # 출발선 이탈 순간
                self.near_starts[i] = False
                self.toggle_list[i] += 1
            self.lap_counts[i] = self.toggle_list[i] // 2
            if self.toggle_list[i] < 4:
                self.lap_times[i] = self.current_time

        # 종료 조건 -> 충돌 or agent가 2랩 완료 시 
        done = (self.collisions[self.ego_idx]) or np.all(self.toggle_list >= 4)

        return done, self.toggle_list >= 4

    def _update_state(self, obs_dict):
        """
        Update the env's states according to observations

        Args:
            obs_dict (dict): dictionary of observation (외부에서 받은 observation을 환경 내부 상태로 복사)

        Returns:
            None
        """
        self.poses_x = obs_dict['poses_x']
        self.poses_y = obs_dict['poses_y']
        self.poses_theta = obs_dict['poses_theta']
        self.collisions = obs_dict['collisions']

    def step(self, action):
        obs = self.sim.step(action)

        obs['lap_times'] = self.lap_times
        obs['lap_counts'] = self.lap_counts
        self.pre_lap_counts = list(self.lap_counts)
        self.current_obs = obs

        # ==========================================
        # 1. 차량 상태 추출 (Ego Agent 기준)
        # ==========================================
        X, Y = obs['poses_x'][self.ego_idx], obs['poses_y'][self.ego_idx]
        theta = obs['poses_theta'][self.ego_idx] # 차량의 현재 Heading Angle (라디안)
        current_vel = obs['linear_vels_x'][self.ego_idx]

        # ==========================================
        # 2. Waypoint 추적 & 인덱스 포워딩 (랜덤 포즈 완벽 대응)
        # ==========================================
        # 탐색 범위를 현재 인덱스 주변(-5 ~ +15)으로 제한하여 트랙 다른 곳의 중복 위치 오인 방지
        search_indices = [(self.closest_idx + i) % self.num_waypoints for i in range(-5, 15)]
        local_dists = np.linalg.norm(self.waypoints[search_indices] - np.array([X, Y]), axis=1)
        
        # 최적 인덱스 갱신
        best_local_search_idx = np.argmin(local_dists)
        self.closest_idx = search_indices[best_local_search_idx]
        self.count = self.closest_idx
        
        dist_to_wp = local_dists[best_local_search_idx]

        # 다음(Target) Waypoint 결정
        next_idx = (self.closest_idx + 1) % self.num_waypoints
        
        wp_now = self.waypoints[self.closest_idx]
        wp_next = self.waypoints[next_idx]

        # ==========================================
        # 3. 핵심 알고리즘: Waypoint 벡터와 차량 Heading의 유사도(Alignment) 계산
        # ==========================================
        # Waypoint가 가리키는 진행 방향 벡터
        wp_vector = wp_next - wp_now
        wp_norm = np.linalg.norm(wp_vector) + 1e-6
        wp_dir = wp_vector / wp_norm

        # 차량이 바라보고 있는 방향 벡터
        car_dir = np.array([np.cos(theta), np.sin(theta)])

        # 두 벡터의 내적 (Cosine Similarity: 완전 일치 시 1.0, 역주행 시 -1.0)
        alignment = np.dot(car_dir, wp_dir)

        # ==========================================
        # 4. 정밀 보상(Reward) 계산 설계
        # ==========================================
        reward = 0.0

        # (A) 전진 속도 보상: 올바른 방향을 볼 때만 속도 보상 부여 (역주행 가속 패널티)
        if alignment > 0:
            reward += 1.5 * current_vel * alignment
        else:
            reward -= 2.0 * abs(current_vel)  # 거꾸로 달리거나 돌면 강한 패널티

        # (B) 방향성(Alignment) 자체 보상
        # 직선 구간에 가까울수록(alignment 증가) 높은 보상, 조향이 어긋나면 스케일 감소
        reward += 2.0 * alignment 

        # (C) 트랙 이탈(횡오차) 패널티
        # 센터라인(Waypoint)에서 멀어질수록 감점
        reward -= 0.5 * dist_to_wp

        # (D) LiDAR 기반 장애물 및 벽 패널티
        scan = np.array(obs['scans'][self.ego_idx])
        n = len(scan)
        front_dist = np.min(scan[n//3 : -n//3])

        if front_dist < 0.5:
            reward -= 10.0  # 충돌 직전 상황 감점

        # (E) 이전 스텝 대비 Waypoint 접근 보상 (Progress Reward)
        if self.prev_dist is not None:
            # 올바른 주행 방향일 때만 거리가 가까워진 것에 보상 부여
            if alignment > 0.5:
                reward += (self.prev_dist - dist_to_wp) * 5.0
        self.prev_dist = dist_to_wp

        # ==========================================
        # 5. 상태 업데이트 및 종료 조건 처리
        # ==========================================
        self.current_time += self.timestep
        self._update_state(obs)

        done, toggle_list = self._check_done() # 아래 내부 헬퍼 함수 참고
        info = {'checkpoint_done': toggle_list}

        # 충돌 시 대형 패널티 및 에피소드 종료
        if self.collisions[self.ego_idx]:
            reward = -150.0

        if self.lap_counts[self.ego_idx] != self.pre_lap_counts[self.ego_idx]:
            self.checklist = np.zeros((15))

        return obs, reward, done, info

    def reset(self, poses):
        """
        Reset the gym environment by given poses

        Args:
            poses (np.ndarray (num_agents, 3)): poses to reset agents to

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxillary information dictionary
        """
        # reset counters and data members
        self.current_time = 0.0
        self.collisions = np.zeros((self.num_agents,))
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True] * self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))

        self.checklist = np.zeros((15))

        # states after reset
        self.start_xs = poses[:, 0]
        self.start_ys = poses[:, 1]
        self.start_thetas = poses[:, 2]
        self.start_rot = np.array(
            [[np.cos(-self.start_thetas[self.ego_idx]), -np.sin(-self.start_thetas[self.ego_idx])],
             [np.sin(-self.start_thetas[self.ego_idx]), np.cos(-self.start_thetas[self.ego_idx])]])

        # call reset to simulator
        self.sim.reset(poses)

        # ==========================================
        # [수정] 랜덤 Pose 진입 시 가장 가까운 Waypoint 탐색 및 초기화
        # ==========================================
        X, Y = poses[self.ego_idx, 0], poses[self.ego_idx, 1]
        dists = np.linalg.norm(self.waypoints - np.array([X, Y]), axis=1)
        self.closest_idx = int(np.argmin(dists))
        self.prev_dist = dists[self.closest_idx]
        self.count = self.closest_idx

        # get no input observations
        action = np.zeros((self.num_agents, 2))
        obs, reward, done, info = self.step(action)
        return obs, reward, done, info

    def update_map(self, map_path, map_ext):
        """
        Updates the map used by simulation

        Args:
            map_path (str): absolute path to the map yaml file
            map_ext (str): extension of the map image file

        Returns:
            None
        """
        self.sim.set_map(map_path, map_ext)

    def update_params(self, params, index=-1):
        """
        Updates the parameters used by simulation for vehicles

        Args:
            params (dict): dictionary of parameters
            index (int, default=-1): if >= 0 then only update a specific agent's params

        Returns:
            None
        """
        self.sim.update_params(params, agent_idx=index)

    def render(self, mode='human'):
        """
        Renders the environment with pyglet. Use mouse scroll in the window to zoom in/out, use mouse click drag to pan. Shows the agents, the map, current fps (bottom left corner), and the race information near as text.

        Args:
            mode (str, default='human'): rendering mode, currently supports:
                'human': slowed down rendering such that the env is rendered in a way that sim time elapsed is close to real time elapsed
                'human_fast': render as fast as possible

        Returns:
            None
        """
        assert mode in ['human', 'human_fast']
        if self.renderer is None:
            # first call, initialize everything
            from f110_gym.envs.rendering import EnvRenderer
            self.renderer = EnvRenderer(WINDOW_W, WINDOW_H)
            self.renderer.update_map(self.map_name, self.map_ext)
        self.renderer.update_obs(self.current_obs)
        self.renderer.dispatch_events()
        self.renderer.on_draw()
        self.renderer.flip()
        if mode == 'human':
            time.sleep(0.005)
        elif mode == 'human_fast':
            pass
# Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

