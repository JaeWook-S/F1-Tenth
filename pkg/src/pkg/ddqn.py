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

plt.ion()

# Hyperparameters
learning_rate = 0.00005
gamma = 0.98
buffer_limit = 50000
batch_size = 64  # 대규모 GPU 연산을 활용하기 위해 스케일업
train_start = 20000

# EMA 조향 스무딩 계수 
SMOOTHING_ALPHA = 0.3 

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)

RACETRACK = 'Oschersleben'

# 리다이 및 속도 파라미터
LIDAR_MIN = 0
LIDAR_MAX = 30
VELOCITY_MIN = -5
VELOCITY_MAX = 20

# 웨이포인트 로드 및 정방향 뒤집기
csv_path = "{}/maps/Oschersleben_centerline.csv".format(current_dir)
waypoints = np.genfromtxt(csv_path, delimiter=',') 
waypoints = np.flip(waypoints, axis=0)
num_waypoints = len(waypoints)

def get_today():
    now = time.localtime()
    s = "%02d-%02d-%02d" % (now.tm_year%100, now.tm_mon, now.tm_mday) 
    return s

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.priorities = collections.deque(maxlen=buffer_limit)
        self.max_priority = 1.0
        self.alpha = 0.6   

    def put(self, transition):
        self.buffer.append(transition)
        self.priorities.append(self.max_priority)

    def sample(self, n, beta):
        probs = np.array(self.priorities)
        probs = probs / probs.sum()

        indices = np.random.choice(len(self.buffer), n, p=probs)

        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights = weights / weights.max()

        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for idx in indices:
            s, a, r, s_prime, done_mask = self.buffer[idx]
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return (
            torch.tensor(np.array(s_lst), dtype=torch.float),
            torch.tensor(a_lst),
            torch.tensor(r_lst, dtype=torch.float),
            torch.tensor(np.array(s_prime_lst), dtype=torch.float),
            torch.tensor(done_mask_lst, dtype=torch.float),
            indices,
            torch.tensor(weights, dtype=torch.float)
        )

    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            priority = (abs(error) + 1e-5) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def size(self):
        return len(self.buffer)
    
class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        # State 차원: Min-Pooling (108개) + (speed, prev_steer 2개) = 110 차원
        self.fc1 = nn.Linear(110, 256) 
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 5) 

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def sample_action(self, obs, epsilon, memory_size):
        if memory_size < train_start:
            return random.randint(0, 4)
        else:
            out = self.forward(obs)
            coin = random.random()
            if coin < epsilon:
                return random.randint(0, 4)
            else:
                return out.argmax().item()

    def action(self, obs):
        out = self.forward(obs)
        # 평가(Eval) 시에는 노이즈 없이 100% Greedy 행동
        return out.argmax().item()

def plot_durations(laptimes):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(laptimes, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 10:
        means = durations_t.unfold(0, 10, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(9), means))
        plt.plot(means.numpy())

    plt.pause(0.001)
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def train(q, q_target, memory, optimizer, beta):
    for i in range(10):
        s, a, r, s_prime, done_mask, indices, weights = memory.sample(batch_size, beta)

        q_out = q(s)
        q_a = q_out.gather(1, a)

        # Double DQN (DDQN)
        with torch.no_grad():
            argmax_a = q(s_prime).argmax(dim=1, keepdim=True)
            max_q_prime = q_target(s_prime).gather(1, argmax_a)
            target = r + gamma * max_q_prime * done_mask

        loss = F.smooth_l1_loss(q_a, target, reduction='none')
        loss = (weights.unsqueeze(1) * loss).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            td_errors = torch.abs(q_a - target).cpu().numpy().flatten()

        memory.update_priorities(indices, td_errors)

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

def main():
    today = get_today()
    work_dir = "./" + today
    os.makedirs(work_dir + "_ddqn", exist_ok=True)

    env = gym.make('f110_gym:f110-v3',
                   map="{}/maps/{}".format(current_dir, RACETRACK),
                   map_ext=".png", num_agents=1)
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 10
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    fastlap = 10000.0
    laptimes = []

    for n_epi in range(10000):
        # Epsilon 스케줄링 (충분한 탐험을 위해 1.0에서 천천히 감소)
        #epsilon = max(0.01, 1.0 - (0.98 * (n_epi / 6000)))
        epsilon = max(0.01, 1.0 * (0.998 ** n_epi))
        beta = min(1.0, 0.4 + (1.0 - 0.4) * (n_epi / 10000))

        # 랜덤 스폰 위치 및 방향 계산
        random_idx = random.randint(0, num_waypoints - 1)
        rand_x = waypoints[random_idx][0]
        rand_y = waypoints[random_idx][1]
        
        next_idx = (random_idx + 1) % num_waypoints
        dx = waypoints[next_idx][0] - rand_x
        dy = waypoints[next_idx][1] - rand_y
        base_theta = np.arctan2(dy, dx)
        
        poses = np.array([[rand_x, rand_y, base_theta]])
        
        obs, r, done, info = env.reset(poses=poses)
        prev_steer = 0.0
        s = define_state(obs, prev_steer)

        done = False
        laptime = 0.0

        while not done:
            actions = []
            a = q.sample_action(torch.from_numpy(s).float(), epsilon, memory.size())
            
            steer = (a - 2) * (np.pi / 30)
            #steer = (SMOOTHING_ALPHA * target_steer) + ((1.0 - SMOOTHING_ALPHA) * prev_steer)
            
            # 고속 코너링 기어비 튜닝
            # if a == 3:
            #     speed = 10.0
            # elif a == 2 or a == 4:
            #     speed = 9.5  
            # elif a == 1 or a == 5:
            #     speed = 7.0  
            # else:
            #     speed = 4.5  
            # if a == 2:
            #     speed = 12
            # elif a == 1 or a == 3:
            #     speed = 10
            # else:
            #     speed = 8
            if a == 2:
                speed = 5
            elif a == 1 or a == 3:
                speed = 4.5
            else:
                speed = 4

            actions.append([steer, speed])
            actions = np.array(actions)
            obs, r, done, info = env.step(actions)
            
            s_prime = define_state(obs, steer)
            done_mask = 0.0 if done else 1.0

            # Env에서 계산된 보상을 스케일링만 하여 바로 적용 (추가 감점 로직 제거)
            memory.put((s, a, r / 10.0, s_prime, done_mask))
            
            s = s_prime
            # prev_steer = steer
            laptime += r # Env에서 넘겨주는 r(timestep)을 누적하여 실제 경과 시간 계산

            if done:
                laptimes.append(laptime)
                plot_durations(laptimes)
                lap = round(obs['lap_times'][0], 3)

                # 2랩 완료 했을 때 + 기존 best lab보다 빠를 때 저장
                if int(obs['lap_counts'][0]) == 2 and fastlap > lap:        
                    fastlap = lap
                    save_name = f"{work_dir}_ddqn/fast-model_{lap:.2f}sec_idx{random_idx}_epi{n_epi}.pt"
                    torch.save(q.state_dict(), save_name)
                break
        
        # 평균적인 성능 검증을 위한 주기적 모델 저장
        # if n_epi > 0 and n_epi % 1000 == 0:
        #     torch.save(q.state_dict(), work_dir + "_ddqn" + f'/periodic-model_{n_epi}.pt')

        if memory.size() > train_start:
            train(q, q_target, memory, optimizer, beta)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%, beta : {:.3f}"
                  .format(n_epi, laptime, memory.size(), epsilon * 100, beta))

    print('train finish')
    env.close()

def eval():
    env = gym.make('f110_gym:f110-v3',
                   map="{}/maps/{}".format(current_dir, RACETRACK),
                   map_ext=".png", num_agents=1)

    q = Qnet()
    # 주의: 훈련 시 저장된 모델 경로에 맞게 수정
    q.load_state_dict(torch.load("26-05-23_ddqn/fast-model_109.28sec_idx725_epi2731.pt"))

    # Eval 시에는 시작 지점을 트랙의 첫 번째 웨이포인트(정방향)로 고정하여 성능을 일정하게 비교
    # start_x, start_y = waypoints[11][0], waypoints[11][1]
    # dx = waypoints[11][0] - start_x
    # dy = waypoints[11][1] - start_y
    # start_theta = np.arctan2(dy, dx)
    
    # poses = np.array([[start_x, start_y, start_theta]])
    poses = np.array([[0., 0., np.radians(345)]])
    
    for t in range(5):
        obs, r, done, info = env.reset(poses=poses)
        prev_steer = 0.0
        s = define_state(obs, prev_steer)

        env.render()
        done = False
        laptime = 0.0
        
        while not done:
            actions = []
            a = q.action(torch.from_numpy(s).float())
            
            steer = (a - 2) * (np.pi / 30)

            if a == 2:
                speed = 5
            elif a == 1 or a == 3:
                speed = 4.5
            else:
                speed = 4

            actions.append([steer, speed])
            actions = np.array(actions)
            obs, r, done, info = env.step(actions)
            
            s_prime = define_state(obs, steer)
            
            s = s_prime
            prev_steer = steer

            laptime += r
            env.render(mode='human_fast')

            if done:
                lap = round(obs['lap_times'][0], 3)
                print(f"[EVAL {t}] lap_time: {lap:.3f} sec")
                break
    env.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'eval'])
    args = parser.parse_args()

    if args.mode == 'train':
        main()
    elif args.mode == 'eval':
        eval()

# epsilon = max(0.01, 1.0 * (0.998 ** n_epi)) 해보기