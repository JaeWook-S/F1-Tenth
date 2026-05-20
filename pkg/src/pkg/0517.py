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

# JY add : parameter
LIDAR_MIN = 0
LIDAR_MAX = 30
VELOCITY_MIN = -5
VELOCITY_MAX = 20

def get_today():
    now = time.localtime()
    s = "%02d-%02d-%02d" % (now.tm_year%100, now.tm_mon, now.tm_mday) # JY fix
    return s

# PER로 수정된 ReplayBuffer
class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.priorities = collections.deque(maxlen=buffer_limit)
        self.max_priority = 1.0
        self.alpha = 0.6   # priority 얼마나 강하게 반영할지

    def put(self, transition):
        self.buffer.append(transition)
        self.priorities.append(self.max_priority)

    def sample(self, n, beta):
        probs = np.array(self.priorities)
        probs = probs / probs.sum()

        indices = np.random.choice(len(self.buffer), n, p=probs)

        # 중요도 샘플링 가중치 계산 (매개변수로 받은 beta 사용)
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
            torch.tensor(s_lst, dtype=torch.float),
            torch.tensor(a_lst),
            torch.tensor(r_lst),
            torch.tensor(s_prime_lst, dtype=torch.float),
            torch.tensor(done_mask_lst),
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
        self.fc1 = nn.Linear(408, 256) # JY fix : 405 -> 408
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 7) # 가능한 action 5개 

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def sample_action(self, obs, epsilon, memory_size):
        if memory_size < train_start:
            return random.randint(0, 6)
        else:
            out = self.forward(obs) # Q값 계산
            coin = random.random()
            if coin < epsilon:
                return random.randint(0, 6)
            else:
                return out.argmax().item()

    def action(self, obs):
        out = self.forward(obs)
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

# 동적 beta 스케줄링이 반영된 train 함수
def train(q, q_target, memory, optimizer, beta):
    for i in range(10):
        s, a, r, s_prime, done_mask, indices, weights = memory.sample(batch_size, beta)

        q_out = q(s)
        q_a = q_out.gather(1, a)

        with torch.no_grad():
            max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
            target = r + gamma * max_q_prime * done_mask

        # weighted loss (중요도 가중치 반영)
        loss = F.smooth_l1_loss(q_a, target, reduction='none')
        loss = (weights.unsqueeze(1) * loss).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # TD-error 업데이트
        with torch.no_grad():
            td_errors = torch.abs(q_a - target).cpu().numpy().flatten()

        memory.update_priorities(indices, td_errors)

def preprocess_lidar(ranges):
    eighth = int(len(ranges) / 8)
    return np.array(ranges[eighth:-eighth: 2])

def convert_range(value, input_range, output_range):
    (in_min, in_max), (out_min, out_max) = input_range, output_range
    in_range = in_max - in_min
    out_range = out_max - out_min
    return (((value - in_min) * out_range) / in_range) + out_min

def lidar_normalize(obs):
    return convert_range(obs, [LIDAR_MIN, LIDAR_MAX], [-1, 1])

def define_state(obs):
    lidar_point = lidar_normalize(preprocess_lidar(obs['scans'][0]))
    car_orientation = obs['poses_theta'][0]
    car_speed = np.tanh(obs['linear_vels_x'][0] / 10.0)
    state = np.concatenate([lidar_point, np.array([np.sin(car_orientation), np.cos(car_orientation), car_speed])])
    return state

def main():
    today = get_today()
    work_dir = "./" + today
    os.makedirs(work_dir + '_'  + "add_reward", exist_ok=True)

    env = gym.make('f110_gym:f110-v1',
                   map="{}/maps/{}".format(current_dir, RACETRACK),
                   map_ext=".png", num_agents=1)
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    if RACETRACK == "map_easy3" :
        poses = np.array([[0.8007017, -0.2753365, 4.1421595]]) 
    else:
        poses = np.array([[0, 0, np.radians(345)]])

    print_interval = 10
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    speed = 3.0
    fastlap = 10000.0
    laptimes = []

    for n_epi in range(10000):
        # epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))  # Linear annealing from 8% to 1%
        epsilon = max(0.02, 0.5 - (0.48 * (n_epi / 3000)))
        # 🔥 beta 선형 스케줄링 (5000 에피소드 동안 0.4에서 1.0까지 선형 증가)
        beta = min(1.0, 0.4 + (1.0 - 0.4) * (n_epi / 5000))
        
        obs, r, done, info = env.reset(poses=poses)
        s = define_state(obs)

        done = False
        laptime = 0.0

        while not done:
            actions = []

            a = q.sample_action(torch.from_numpy(s).float(), epsilon, memory.size())
            steer = (a - 3) * (0.4189 / 3) # a=0일 때 -24도, a=3일 때 0도, a=6일 때 +24도

            if a == 3:
                speed = 10 # 직진일 때 최고 속도
            elif a == 2 or a == 4:
                speed = 8 # 완만한 코너
            elif a == 1 or a == 5:
                speed = 6  # 중간 코너
            else:
                speed = 4  # 급코너 (a=0, 6일 때 풀조향하며 감속)

            actions.append([steer, speed])
            actions = np.array(actions)
            obs, r, done, info = env.step(actions)
            s_prime = define_state(obs)

            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r / 100, s_prime, done_mask))
            s = s_prime

            laptime += r

            if done:
                laptimes.append(laptime)
                plot_durations(laptimes)
                lap = round(obs['lap_times'][0], 3)

                if int(obs['lap_counts'][0]) == 2 and fastlap > lap:
                    torch.save(q.state_dict(), work_dir  + "_add_reward" + '/fast-model' + str(
                        round(obs['lap_times'][0], 3)) + '_' + str(n_epi) + '.pt')
                    fastlap = lap
                    break

        if memory.size() > train_start:
            # 주입되는 beta를 기반으로 최적화 진행
            train(q, q_target, memory, optimizer, beta)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%, beta : {:.3f}"
                  .format(n_epi, laptime / print_interval, memory.size(), epsilon * 100, beta))

    print('train finish')
    env.close()

def eval():
    env = gym.make('f110_gym:f110-v1',
                   map="{}/maps/{}".format(current_dir, RACETRACK),
                   map_ext=".png", num_agents=1)

    q = Qnet()
    q.load_state_dict(torch.load("26-05-17_add_reward/fast-model37.99_5010.pt"))

    if RACETRACK == "map_easy3" :
        poses = np.array([[0.8007017, -0.2753365, 4.1421595]]) 
    else:
        poses = np.array([[0, 0, np.radians(345)]])
    
    speed = 3.0
    for t in range(5):
        obs, r, done, info = env.reset(poses=poses)
        s = define_state(obs)

        env.render()
        done = False
        laptime = 0.0

        while not done:
            actions = []

            a = q.action(torch.from_numpy(s).float())
            steer = (a - 3) * (0.4189 / 3) # a=0일 때 -24도, a=3일 때 0도, a=6일 때 +24도

            if a == 3:
                speed = 10 # 직진일 때 최고 속도
            elif a == 2 or a == 4:
                speed = 8 # 완만한 코너
            elif a == 1 or a == 5:
                speed = 6  # 중간 코너
            else:
                speed = 4  # 급코너 (a=0, 6일 때 풀조향하며 감속)

            actions.append([steer, speed])
            actions = np.array(actions)
            obs, r, done, info = env.step(actions)
            s_prime = define_state(obs)
            s = s_prime

            laptime += r
            env.render(mode='human')

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