# MIT License

# Copyright (c) 2020 FT Autonomous Team One

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

import gym
import numpy as np

from gym import spaces
from pathlib import Path

def convert_range(value, input_range, output_range):
    # converts value(s) from range to another range
    # ranges ---> [min, max]
    (in_min, in_max), (out_min, out_max) = input_range, output_range
    in_range = in_max - in_min
    out_range = out_max - out_min
    return (((value - in_min) * out_range) / in_range) + out_min

class F110_Wrapped(gym.Wrapper):
    """
    This is a wrapper for the F1Tenth Gym environment intended
    for only one car, but should be expanded to handle multi-agent scenarios
    """

    def __init__(self, env):
        super().__init__(env)

        # normalised action space, steer and speed
        self.action_space = spaces.Box(low=np.array(
            [-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float)

        # normalised observations, just take the lidar scans
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1080,), dtype=np.float)

        # store allowed steering/speed/lidar ranges for normalisation
        self.s_min = self.env.params['s_min']
        self.s_max = self.env.params['s_max']
        self.v_min = self.env.params['v_min']
        self.v_max = self.env.params['v_max']
        self.lidar_min = 0
        self.lidar_max = 30  # see ScanSimulator2D max_range

        self.step_count = 0

        # set threshold for maximum angle of car, to prevent spinning
        self.max_theta = 100

    def step(self, action):
        # convert normalised actions (from RL algorithms) back to actual actions for simulator
        action_convert = self.un_normalise_actions(action)
        observation, _, done, info = self.env.step(np.array([action_convert]))

        self.step_count += 1

        # TODO -> do some reward engineering here and mess around with this
        reward = 0

        #eoins reward function
        vel_magnitude = np.linalg.norm(
            [observation['linear_vels_x'][0], observation['linear_vels_y'][0]])
        reward = vel_magnitude #/10 maybe include if speed is having too much of an effect

        # reward function that returns percent of lap left, maybe try incorporate speed into the reward too
        #waypoints = np.genfromtxt(f"./f1tenth_racetracks/{randmap}/{randmap}_centerline.csv", delimiter=',')
        if np.argmin(observation['scans'][0]) >= 300 and np.argmin(observation['scans'][0]) <= 780: # 가장 가까운 중앙 방향에 장애물 있는 경우
            reward -= 1
        elif np.argmin(observation['scans'][0]) < 300 or np.argmin(observation['scans'][0]) > 780: # 장애물이 옆에 있는 경우
            reward += 2
        if min(observation['scans'][0]) < 0.5: # 거리 기반 penalty -> 장애물 너무 가까울 때
            reward -= 5

        if observation['collisions'][0]:
            # reward = -100
            reward = -1

        # end episode if car is spinning
        if abs(observation['poses_theta'][0]) > self.max_theta:
            done = True

        """
        vel_magnitude = np.linalg.norm([observation['linear_vels_x'][0], observation['linear_vels_y'][0]])
        #print("V:",vel_magnitude)
        if vel_magnitude > 0.2:  # > 0 is very slow and safe, sometimes just stops in its tracks at corners
            reward += 0.1"""


        # penalise changes in car angular orientation (reward smoothness)
        """ang_magnitude = abs(observation['ang_vels_z'][0])
        #print("Ang:",ang_magnitude)
        if ang_magnitude > 0.75:
            reward += -ang_magnitude/10
        ang_magnitude = abs(observation['ang_vels_z'][0])
        if ang_magnitude > 5:
            reward = -(vel_magnitude/10)

        # if collisions is true, then the car has crashed
        if observation['collisions'][0]:
            self.count = 0
            #reward = -100
            reward = -1

        # end episode if car is spinning
        if abs(observation['poses_theta'][0]) > self.max_theta:
            self.count = 0
            reward = -100
            reward = -1
            done = True

        # just a simple counter that increments when the car completes a lap
        if self.env.lap_counts[0] > 0:
            self.count = 0
            reward += 1
            if self.env.lap_counts[0] > 1:
                reward += 1
                self.env.lap_counts[0] = 0"""
        info = 
        return self.normalise_observations(observation['scans'][0]), reward, bool(done), info

    def reset(self):
        poses = np.array([[0.8007017, -0.2753365, 4.1421595]]) 
        # reset car with chosen pose
        observation, _, _, _ = self.env.reset(poses)
        # reward, done, info can't be included in the Gym format
        return self.normalise_observations(observation['scans'][0])

    def un_normalise_actions(self, actions):
        # convert actions from range [-1, 1] to normal steering/speed range
        steer = convert_range(actions[0], [-1, 1], [self.s_min, self.s_max])
        speed = convert_range(actions[1], [-1, 1], [self.v_min, self.v_max])
        return np.array([steer, speed], dtype=np.float)

    def normalise_observations(self, observations):
        # convert observations from normal lidar distances range to range [-1, 1]
        return convert_range(observations, [self.lidar_min, self.lidar_max], [-1, 1])

    def update_map(self, map_path, map_ext):
        self.env.update_map(map_path, map_ext)



class ThrottleMaxSpeedReward(gym.RewardWrapper):
    """
    Slowly increase maximum reward for going fast, so that car learns
    to drive well before trying to improve speed
    """

    def __init__(self, env, start_step, end_step, start_max_reward, end_max_reward=None):
        super().__init__(env)
        # initialise step boundaries
        self.end_step = end_step
        self.start_step = start_step
        self.start_max_reward = start_max_reward
        # set finishing maximum reward to be maximum possible speed by default
        self.end_max_reward = self.v_max if end_max_reward is None else end_max_reward

        # calculate slope for reward changing over time (steps)
        self.reward_slope = (self.end_max_reward - self.start_max_reward) / (self.end_step - self.start_step)

    def reward(self, reward):
        # maximum reward is start_max_reward
        if self.step_count < self.start_step:
            return min(reward, self.start_max_reward)
        # maximum reward is end_max_reward
        elif self.step_count > self.end_step:
            return min(reward, self.end_max_reward)
        # otherwise, proportional reward between two step endpoints
        else:
            return min(reward, self.start_max_reward + (self.step_count - self.start_step) * self.reward_slope)