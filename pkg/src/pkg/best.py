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

def get_today():
    now = time.localtime()
    # s = "%04d-%02d-%02d_%02d-%02d-%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    s = "%02d-%02d-%02d" % (now.tm_year%100, now.tm_mon, now.tm_mday) # JY fix
    return s

# 환경에서 나온 경험을 저장해두고, 랜덤하게 꺼내서 학습에 사용
class ReplayBuffer():
    def __init__(self):
        # buffer 찰 시 오래된 데이터를 새 데이터로 (FIFO 구조)
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    # n개 랜덤 샘플링
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        # state, action, reward, next state, done
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a]) # [] : shape 맞추기 위해
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
            torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
            torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(408, 256) # JY fix : 405 -> 408
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


def plot_durations(laptimes):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(laptimes, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # 10개의 에피소드 평균을 가져 와서 도표 그리기
    if len(durations_t) >= 10:
        means = durations_t.unfold(0, 10, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(9), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # 도표가 업데이트되도록 잠시 멈춤
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 1080개 데이터 다운샘플링 -> 전체 시야의 양 끝 1/8씩 버림 + 2 간격만큼 다운샘플링
def preprocess_lidar(ranges):
    eighth = int(len(ranges) / 8)

    return np.array(ranges[eighth:-eighth: 2])

# JY add : input_range : 현재 range / output_range : 바꾸고자 하는 range 
def convert_range(value, input_range, output_range):
    (in_min, in_max), (out_min, out_max) = input_range, output_range
    in_range = in_max - in_min
    out_range = out_max - out_min

    return (((value - in_min) * out_range) / in_range) + out_min

# JY add : lidar 범위 -1 ~ 1로 변경 -> 모델 입력을 위해 
def lidar_normalize(obs):
    return convert_range(obs, [LIDAR_MIN, LIDAR_MAX], [-1, 1])

# JY add : state function
def define_state(obs):
    lidar_point = lidar_normalize(preprocess_lidar(obs['scans'][0])) # 약 450개 라이다 포인터 존재 -> 현재 state는 라이다 포인터만 있음

    # JY add : car speed, orientation state
    car_orientation = obs['poses_theta'][0]
    car_speed = np.tanh(obs['linear_vels_x'][0] / 10.0)

    state = np.concatenate([lidar_point, np.array([np.sin(car_orientation), np.cos(car_orientation), car_speed])])

    return state

def main():
    today = get_today()
    work_dir = "./" + today
    os.makedirs(work_dir + '_' + RACETRACK + "_normalization")

    env = gym.make('f110_gym:f110-v0',
                   map="{}/maps/{}".format(current_dir, RACETRACK),
                   map_ext=".png", num_agents=1)
    q = Qnet()
    # q.load_state_dict(torch.load("{}\weigths\model_state_dict_easy1_fin.pt".format(current_dir)))
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    # poses = np.array([[0., 0., np.radians(0)]])
    # JY fix : start pos
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
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))  # Linear annealing from 8% to 1%
        obs, r, done, info = env.reset(poses=poses)
        # s = preprocess_lidar(obs['scans'][0]) # 약 450개 라이다 포인터 존재 -> 현재 state는 라이다 포인터만 있음
        # JY add
        s = define_state(obs)

        done = False

        laptime = 0.0

        while not done:
            env.render()

            actions = []

            a = q.sample_action(torch.from_numpy(s).float(), epsilon, memory.size())
            steer = (a - 2) * (np.pi / 30)
            if a == 2:
                speed = 12
            elif a == 1 or a == 3:
                speed = 10
            else:
                speed = 8
            # JY add 
            # steer_abs = abs(steer)
            # speed = 20 * np.exp(-3 * steer_abs)
            # speed = np.clip(speed, 4, 20)

            actions.append([steer, speed])
            actions = np.array(actions)
            obs, r, done, info = env.step(actions)
            # s_prime = preprocess_lidar(obs['scans'][0])
            s_prime = define_state(obs) # JY add

            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r / 100, s_prime, done_mask))
            s = s_prime

            laptime += r
            env.render(mode='human_fast')

            if done:
                laptimes.append(laptime)
                plot_durations(laptimes)
                lap = round(obs['lap_times'][0], 3)

                # 2랩 완료 했을 때 + 기존 best lab보다 빠를 때 저장
                if int(obs['lap_counts'][0]) == 2 and fastlap > lap:
                    torch.save(q.state_dict(), work_dir + '_' + RACETRACK + "_normalization" + '/fast-model' + str(
                        round(obs['lap_times'][0], 3)) + '_' + str(n_epi) + '.pt')
                    fastlap = lap
                    break

        if memory.size() > train_start:
            train(q, q_target, memory, optimizer)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%"
                  .format(n_epi, laptime / print_interval, memory.size(), epsilon * 100))

    print('train finish')
    env.close()

def eval():
    env = gym.make('f110_gym:f110-v0',
                   map="{}/maps/{}".format(current_dir, RACETRACK),
                   map_ext=".png", num_agents=1)

    q = Qnet()
    # q.load_state_dict(torch.load("{}\weigths\model_state_dict_easy1_fin.pt".format(current_dir)))
    q.load_state_dict(torch.load("26-05-13_Oschersleben_normalization/fast-model62.27_2149.pt"))
    # poses = np.array([[0., 0., np.radians(90)]])

    # JY fix: start pos
    if RACETRACK == "map_easy3" :
        poses = np.array([[0.8007017, -0.2753365, 4.1421595]]) 
    else:
        poses = np.array([[0, 0, np.radians(345)]])
    
    speed = 3.0
    for t in range(5):
        obs, r, done, info = env.reset(poses=poses)
        #s = preprocess_lidar(obs['scans'][0])
        s = define_state(obs) # JY add

        env.render()
        done = False

        laptime = 0.0

        while not done:
            actions = []

            a = q.action(torch.from_numpy(s).float())
            steer = (a - 2) * (np.pi / 30)
            if a == 2: # 11.23, 9.70, 6.65
                speed = 10.9912
            elif a == 1 or a == 3:
                speed = 12.3502
            else:
                speed = 7.4216
            # speed control

            # JY add
            # steer_abs = abs(steer)
            # speed = 20 * np.exp(-3 * steer_abs)
            # speed = np.clip(speed, 20, 4)

            actions.append([steer, speed])
            actions = np.array(actions)
            obs, r, done, info = env.step(actions)
            #s_prime = preprocess_lidar(obs['scans'][0])
            s_prime = define_state(obs) # JY add

            s = s_prime

            laptime += r
            env.render(mode='human')

            if done:
                lap = round(obs['lap_times'][0], 3)
                print(f"[EVAL {t}] lap_time: {lap:.3f} sec")
                break
    env.close()


if __name__ == '__main__':
    # JY add : parse 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'eval'])
    args = parser.parse_args()

    if args.mode == 'train':
        main()
    elif args.mode == 'eval':
        eval()
