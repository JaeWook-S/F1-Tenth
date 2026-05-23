import os
import sys
import collections
import random
import gym
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

# interactive ON -> plt.show() 없이도 자동으로 그래프 업데이트 / 반복문 안에서 그래프 실시간 변경 가능
plt.ion()

# Hyperparameters
learning_rate = 0.00005
gamma = 0.98
buffer_limit = 50000
batch_size = 32
train_start = 20000

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)

# RACETRACK = 'map_easy3'
RACETRACK = 'Oschersleben'

#JY add : parameter
LIDAR_MIN = 0
LIDAR_MAX = 30
VELOCITY_MIN = -5
VELOCITY_MAX = 20

# 웨이포인트 로드 및 정방향 뒤집기
csv_path = "{}/maps/Oschersleben_centerline.csv".format(current_dir)
waypoints = np.genfromtxt(csv_path, delimiter=',') 
waypoints = np.flip(waypoints, axis=0)
num_waypoints = len(waypoints)

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(110, 256) # JY fix : 405 -> 408 -> 110
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 5) # 가능한 action 5개 

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def sample_action(self, obs, epsilon, memory_size):
        # buffer가 충분히 차기 전까지는 완전 랜덤 액션 
        if memory_size < train_start:
            return random.randint(0, 4)
        else:
            out = self.forward(obs) # Q값 계산
            coin = random.random()
            # e-greedy policy
            if coin < epsilon:
                return random.randint(0, 4)
            else:
                return out.argmax().item()

    def action(self, obs):
        # greedy action
        out = self.forward(obs)
        return out.argmax().item()


# 1080개 데이터 다운샘플링 -> 전체 시야의 양 끝 1/8씩 버림 + 2 간격만큼 다운샘플링
# def preprocess_lidar(ranges):
#     eighth = int(len(ranges) / 8)

#     return np.array(ranges[eighth:-eighth: 2])

# # JY add : input_range : 현재 range / output_range : 바꾸고자 하는 range 
# def convert_range(value, input_range, output_range):
#     (in_min, in_max), (out_min, out_max) = input_range, output_range
#     in_range = in_max - in_min
#     out_range = out_max - out_min

#     return (((value - in_min) * out_range) / in_range) + out_min

# # JY add : lidar 범위 -1 ~ 1로 변경 -> 모델 입력을 위해 
# def lidar_normalize(obs):
#     return convert_range(obs, [LIDAR_MIN, LIDAR_MAX], [-1, 1])

# # JY add : state function
# def define_state(obs):
#     lidar_point = lidar_normalize(preprocess_lidar(obs['scans'][0])) # 약 450개 라이다 포인터 존재 -> 현재 state는 라이다 포인터만 있음

#     # JY add : car speed, orientation state
#     car_orientation = obs['poses_theta'][0]
#     car_speed = np.tanh(obs['linear_vels_x'][0] / 10.0)

#     state = np.concatenate([lidar_point, np.array([np.sin(car_orientation), np.cos(car_orientation), car_speed])])

#     return state

def preprocess_lidar_min_pool(ranges, num_sectors=108):
    split_ranges = np.array_split(ranges, num_sectors)
    return np.array([np.min(sector) for sector in split_ranges])

def convert_range(value, input_range, output_range):
    (in_min, in_max), (out_min, out_max) = input_range, output_range
    in_range = in_max - in_min
    out_range = out_max - out_min
    return (((value - in_min) * out_range) / in_range) + out_min

def lidar_normalize(obs):
    return convert_range(obs, [LIDAR_MIN, LIDAR_MAX], [-1, 1])

def define_state(obs, prev_steer):
    lidar_point = lidar_normalize(preprocess_lidar_min_pool(obs['scans'][0]))
    car_speed = np.tanh(obs['linear_vels_x'][0] / 10.0)
    
    state = np.concatenate([
        lidar_point, 
        np.array([car_speed, prev_steer])
    ])
    return state


def run_random_search(num_trials=50):
    env = gym.make('f110_gym:f110-v3', 
                   map=f"{current_dir}/maps/{RACETRACK}", 
                   map_ext=".png", num_agents=1)

    # 모델 로드 (CPU 명시)
    q = Qnet()
    q.load_state_dict(torch.load("26-05-23_ddqn_v3_4step/fast-model_58.01sec_idx247_maxV10.5_epi6097.pt"))

    # [안전 출발선 배치] eval 모드 실행 시 억까 당하지 않도록 트랙 선상 위의 정방향 안전 라인에서 출발시킵니다.
    start_idx = 247
    start_x = waypoints[start_idx][0]
    start_y = waypoints[start_idx][1]
    next_idx = (start_idx + 1) % num_waypoints
    dx = waypoints[next_idx][0] - start_x
    dy = waypoints[next_idx][1] - start_y
    start_theta = np.arctan2(dy, dx)
    
    poses = np.array([[start_x, start_y, start_theta]])
    
    best_lap_time = float('inf')
    best_combination = None

    print(f"\n🚀 Starting CPU Random Search ({num_trials} trials)...")
    
    speed = 3.0
    for t in range(num_trials):
        # 10~20m/s 사이 속도 조합 샘플링
        v_strong = random.uniform(10.4*0.52, 10.6*0.52)   # 강한 회전 (Action 0, 4)
        v_weak = random.uniform(10.4*0.88, 10.6*0.88)     # 약한 회전 (Action 1, 3)
        #v_middle = random.uniform(5,15)
        v_straight = random.uniform(10.4, 10.6) # 직진 (Action 2)
        
        # speed_map = {0: v_strong, 1: v_middle, 2: v_weak, 3: v_straight, 4: v_weak, 5: v_middle, 6: v_strong}
        speed_map = {0: v_strong, 1: v_weak, 3: v_straight, 2: v_weak, 4: v_strong}

        obs, r, done, info = env.reset(poses=poses)

        prev_steer = 0.0
        s = define_state(obs, prev_steer)

        # s = define_state(obs)
        done = False

        laptime = 0.0
        
        while not done:

            actions = []

            a = q.action(torch.from_numpy(s).float())

            # steer = (a - 3) * (0.4189 / 3) # a=0일 때 -24도, a=3일 때 0도, a=6일 때 +24도
            steer = (a - 2) * (np.pi / 30)
            speed = speed_map[a]

            actions.append([steer, speed])
            actions = np.array(actions)
            obs, r, done, info = env.step(actions)
            
            s_prime = define_state(obs, steer) # JY add
            s = s_prime

            laptime += r

            # 충돌 패널티 확인 (충돌 시 해당 조합 무효)
            if r < -5: 
                break

            # 2바퀴 완주 체크
            if int(obs['lap_counts'][0]) == 2:
                lap_time = obs['lap_times'][0]
                if lap_time < best_lap_time:
                    best_lap_time = lap_time
                    # best_combination = (v_strong, v_weak, v_middle, v_straight)
                    best_combination = (v_strong, v_weak, v_straight)
                    print(f"✨ Trial {t+1:02d}: [BEST] {lap_time:.3f}s | Str:{v_straight:.2f}, Wk:{v_weak:.2f}, Sg:{v_strong:.2f}")
                break
        
        if (t + 1) % 5 == 0:
            print(f">>> {t + 1}/{num_trials} trials completed...")

    print("\n" + "="*50)
    print(f"🏆 FINAL BEST SPEED COMBINATION")
    print(f"Best Lap Time: {best_lap_time:.3f} sec")
    print("-" * 50)
    print(f"Action 3 (Straight) : {best_combination[2]:.4f} m/s")
    print(f"Action 2, 4 (Weak)  : {best_combination[1]:.4f} m/s")
    # print(f"Action 1, 5 (middle): {best_combination[2]:.4f} m/s")
    print(f"Action 0, 6 (Strong): {best_combination[0]:.4f} m/s")
    print("="*50)

    env.close()

if __name__ == '__main__':
    run_random_search(num_trials=10000)

