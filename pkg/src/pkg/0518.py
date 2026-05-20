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

learning_rate = 0.00005
gamma = 0.98
buffer_limit = 50000
batch_size = 32
train_start = 20000

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)

RACETRACK = 'Oschersleben'

LIDAR_MIN = 0
LIDAR_MAX = 30

# 🔥 추가
SPEED_VALUES = [10, 8, 6, 4]

def decode_action(a):
    steer_idx = a // 4
    speed_idx = a % 4
    return steer_idx, speed_idx

def get_today():
    now = time.localtime()
    return "%02d-%02d-%02d" % (now.tm_year%100, now.tm_mon, now.tm_mday)

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
        self.fc1 = nn.Linear(408, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 28)  # 🔥 수정

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def sample_action(self, obs, epsilon, memory_size):
        if memory_size < train_start:
            return random.randint(0, 27)  # 🔥 수정
        else:
            out = self.forward(obs)
            if random.random() < epsilon:
                return random.randint(0, 27)  # 🔥 수정
            else:
                return out.argmax().item()

    def action(self, obs):
        return self.forward(obs).argmax().item()


def plot_durations(laptimes):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(laptimes, dtype=torch.float)
    plt.plot(durations_t.numpy())
    plt.pause(0.001)

    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def train(q, q_target, memory, optimizer, beta):
    for i in range(10):
        s, a, r, s_prime, done_mask, indices, weights = memory.sample(batch_size, beta)

        q_out = q(s)
        q_a = q_out.gather(1, a)

        with torch.no_grad():
            max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
            target = r + gamma * max_q_prime * done_mask

        loss = F.smooth_l1_loss(q_a, target, reduction='none')
        loss = (weights.unsqueeze(1) * loss).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            td_errors = torch.abs(q_a - target).cpu().numpy().flatten()

        memory.update_priorities(indices, td_errors)

def preprocess_lidar(ranges):
    eighth = int(len(ranges) / 8)
    return np.array(ranges[eighth:-eighth: 2])

def convert_range(value, input_range, output_range):
    (in_min, in_max), (out_min, out_max) = input_range, output_range
    return (((value - in_min) * (out_max - out_min)) / (in_max - in_min)) + out_min

def lidar_normalize(obs):
    return convert_range(obs, [LIDAR_MIN, LIDAR_MAX], [-1, 1])

def define_state(obs):
    lidar_point = lidar_normalize(preprocess_lidar(obs['scans'][0]))
    car_orientation = obs['poses_theta'][0]
    car_speed = np.tanh(obs['linear_vels_x'][0] / 10.0)
    return np.concatenate([lidar_point, np.array([np.sin(car_orientation), np.cos(car_orientation), car_speed])])

def main():
    today = get_today()
    work_dir = "./" + today
    os.makedirs(work_dir + '_'  + "add_reward", exist_ok=True)

    env = gym.make('f110_gym:f110-v2',
                   map="{}/maps/{}".format(current_dir, RACETRACK),
                   map_ext=".png", num_agents=1)

    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    poses = np.array([[0, 0, np.radians(345)]])

    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    fastlap = 10000.0
    laptimes = []

    for n_epi in range(10000):
        epsilon = max(0.02, 0.5 - (0.48 * (n_epi / 3000)))
        beta = min(1.0, 0.4 + (1.0 - 0.4) * (n_epi / 5000))
        
        obs, r, done, info = env.reset(poses=poses)
        s = define_state(obs)

        done = False
        laptime = 0.0

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon, memory.size())

            # 🔥 핵심 수정
            steer_idx, speed_idx = decode_action(a)
            steer = (steer_idx - 3) * (0.4189 / 3)
            speed = SPEED_VALUES[speed_idx]

            obs, r, done, info = env.step(np.array([[steer, speed]]))
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
                    torch.save(q.state_dict(), work_dir + "_add_reward" + '/fast-model' + str(lap) + '_' + str(n_epi) + '.pt')
                    fastlap = lap
                    break

        if memory.size() > train_start:
            train(q, q_target, memory, optimizer, beta)

        if n_epi % 10 == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            print(f"n_episode:{n_epi}, score:{laptime:.1f}, eps:{epsilon:.2f}")

    env.close()

def eval():
    env = gym.make('f110_gym:f110-v1',
                   map="{}/maps/{}".format(current_dir, RACETRACK),
                   map_ext=".png", num_agents=1)

    q = Qnet()
    q.load_state_dict(torch.load("26-05-18_add_action_speed/fast-model77.94_6829.pt"))

    poses = np.array([[0, 0, np.radians(345)]])

    for t in range(5):
        obs, _, done, _ = env.reset(poses=poses)
        s = define_state(obs)

        env.render()
        done = False
        laptime = 0.0

        while not done:
            a = q.action(torch.from_numpy(s).float())

            # 🔥 동일 수정
            steer_idx, speed_idx = decode_action(a)
            steer = (steer_idx - 3) * (0.4189 / 3)
            speed = SPEED_VALUES[speed_idx]

            obs, r, done, _ = env.step(np.array([[steer, speed]]))
            s = define_state(obs)
            laptime += r
            env.render("human_fast")
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
    else:
        eval()