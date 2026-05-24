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
batch_size = 64  
train_start = 20000

# EMA 조향 스무딩 계수 (속도 20 영역의 슬립 방지를 위한 필수 장치)
SMOOTHING_ALPHA = 0.3 

# 💡 [4-step TD 하이퍼파라미터 정의]
N_STEP = 4  # 4스텝 전방 지평선 확장 (고속 진입 관성 예측 최적화)

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

    def sample(self, n_samples, beta):
        probs = np.array(self.priorities)
        probs = probs / probs.sum()

        # ⚡ [수정] 4-step 미래 조회를 위해 샘플링 범위를 버퍼 끝 마진(N_STEP)만큼 제외
        valid_len = len(self.buffer) - N_STEP
        if valid_len <= 0:
            indices = np.random.choice(len(self.buffer), n_samples, p=probs)
        else:
            valid_probs = probs[:valid_len]
            valid_probs = valid_probs / valid_probs.sum()
            indices = np.random.choice(valid_len, n_samples, p=valid_probs)

        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights = weights / weights.max()

        s_lst, a_lst, r_n_lst, s_n_lst, done_mask_lst = [], [], [], [], []

        for idx in indices:
            s, a, r, _, _ = self.buffer[idx]
            
            # ⚡ [수정] 4-step Return 보상 복리 계산부
            n_reward = 0.0
            discount = 1.0
            final_done_mask = 1.0
            s_n = None
            
            for k in range(N_STEP):
                if idx + k < len(self.buffer):
                    curr_s, curr_a, curr_r, curr_s_prime, curr_done_mask = self.buffer[idx + k]
                    n_reward += discount * curr_r
                    discount *= gamma
                    
                    if curr_done_mask == 0.0:
                        final_done_mask = 0.0
                        s_n = curr_s_prime
                        break
                    s_n = curr_s_prime
            
            s_lst.append(s)
            a_lst.append([a])
            r_n_lst.append([n_reward])
            s_n_lst.append(s_n)
            done_mask_lst.append([final_done_mask])

        return (
            torch.tensor(np.array(s_lst), dtype=torch.float),
            torch.tensor(a_lst),
            torch.tensor(r_n_lst, dtype=torch.float),
            torch.tensor(np.array(s_n_lst), dtype=torch.float),
            torch.tensor(done_mask_lst, dtype=torch.float),
            indices,
            torch.tensor(weights, dtype=torch.float)
        )

    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            priority = (abs(error) + 1e-5) ** self.alpha
            if idx < len(self.priorities):
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)

    def size(self):
        return len(self.buffer)
    
class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(110, 256) 
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 5) # 💡 5개 액션 레이아웃 정직하게 고수

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
        # 💡 ReplayBuffer로부터 4스텝 미래 지평선이 압축된 리턴 및 다음 타겟 상태 수령
        s, a, r_n, s_n, done_mask, indices, weights = memory.sample(batch_size, beta)

        q_out = q(s)
        q_a = q_out.gather(1, a)

        with torch.no_grad():
            # ⚡ [수정] Double DQN 구조와 4-step Bellman Target 연산 최적화 동기화
            argmax_a = q(s_n).argmax(dim=1, keepdim=True)
            max_q_prime = q_target(s_n).gather(1, argmax_a)
            target = r_n + (gamma ** N_STEP) * max_q_prime * done_mask

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
    os.makedirs(work_dir + "_ddqn_v4_4step", exist_ok=True)

    env = gym.make('f110_gym:f110-v1',
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

    # 💡 [초고속 해금 영역용 정밀 스케줄러 세팅]
    current_max_speed = 9.0      # 검증이 완료된 안정 지대 시속 10.0m/s부터 4스텝 TD 훈련 스타트
    target_top_speed = 20.0      
    consecutive_success = 0      
    required_success = 2         # 억까 연속 매칭 스폰 방지를 위해 완화 (2회 연속 완주 시 해금)

    for n_epi in range(10000):
        epsilon = max(0.01, 1.0 * (0.998 ** n_epi))
        beta = min(1.0, 0.4 + (1.0 - 0.4) * (n_epi / 10000))

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
            
            # 조향 스무딩(EMA) 필터 활성화
            target_steer = (a - 2) * (np.pi / 30)
            steer = (SMOOTHING_ALPHA * target_steer) + ((1.0 - SMOOTHING_ALPHA) * prev_steer)
            
            # ⚡ [수정] 5개 액션 구조용 초고속 v3 하이브리드 기어비 스케줄링
            if current_max_speed >= 10.0:
                if a == 2:
                    speed = current_max_speed          # 완전 직진 주로: 100% 개방
                elif a == 1 or a == 3:
                    speed = current_max_speed * 0.88   # 완만 코너: 탄력 극대화 (88% 크루징)
                else:
                    speed = current_max_speed * 0.52   # 하드 코너: 탈출 마진 확보를 위한 하드 감속 (52%)
            else:
                # 10m/s 미만 빌드업용 기존 기어 매핑
                if a == 2:
                    speed = current_max_speed
                elif a == 1 or a == 3:
                    speed = current_max_speed * 0.85
                else:
                    speed = current_max_speed * 0.60

            actions.append([steer, speed])
            actions = np.array(actions)
            obs, r, done, info = env.step(actions)
            
            s_prime = define_state(obs, steer)
            done_mask = 0.0 if done else 1.0

            memory.put((s, a, r / 100.0, s_prime, done_mask))
            
            s = s_prime
            prev_steer = steer  
            laptime += r 

            if done:
                laptimes.append(laptime)
                plot_durations(laptimes)
                lap = round(obs['lap_times'][0], 3)

                if int(obs['lap_counts'][0]) == 2:
                    consecutive_success += 1  
                    
                    if consecutive_success >= required_success:
                        if current_max_speed < target_top_speed:
                            # ⚡ [수정] 10.0m/s 이상부터 0.1m/s씩 정밀 승격 메커니즘
                            speed_step = 0.1 if current_max_speed >= 10.0 else 0.2
                            current_max_speed = min(target_top_speed, current_max_speed + speed_step)
                            
                            # # 🧠 속도 승격 시 굳어버린 탐험률을 일시 복구하여 고속 적응력 극대화
                            if current_max_speed >= 10.0:
                                epsilon = max(epsilon, 0.15)
                                
                        consecutive_success = 0  
                    
                    if fastlap > lap:        
                        fastlap = lap
                        save_name = f"{work_dir}_ddqn_v4_4step/fast-model_{lap:.2f}sec_idx{random_idx}_maxV{current_max_speed:.1f}_epi{n_epi}.pt"
                        torch.save(q.state_dict(), save_name)
                else:
                    consecutive_success = 0

                break

        if memory.size() > train_start:
            train(q, q_target, memory, optimizer, beta)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%, max_speed : {:.1f}m/s (Streak: {}/{})"
                  .format(n_epi, laptime, memory.size(), epsilon * 100, current_max_speed, consecutive_success, required_success))

    print('train finish')
    env.close()

def eval():
    env = gym.make('f110_gym:f110-v1',
                   map="{}/maps/{}".format(current_dir, RACETRACK),
                   map_ext=".png", num_agents=1)

    q = Qnet()
    # 📝 [테스트 가이드] 새롭게 추출한 v3_4step 폴더 안의 검증 가중치를 로드하세요.
    q.load_state_dict(torch.load("26-05-24_ddqn_v4_4step/fast-model_59.66sec_idx428_maxV10.4_epi7421.pt"))

    start_idx = 428
    start_x = waypoints[start_idx][0]
    start_y = waypoints[start_idx][1]
    next_idx = (start_idx + 1) % num_waypoints
    dx = waypoints[next_idx][0] - start_x
    dy = waypoints[next_idx][1] - start_y
    start_theta = np.arctan2(dy, dx)
    
    poses = np.array([[start_x, start_y, start_theta]])
    
    # [동기화 매칭] 테스트할 모델 파일에 찍힌 maxV 값을 기입하세요.
    current_eval_max = 10.4
    
    for t in range(5):
        obs, r, done, info = env.reset(poses=poses)
        prev_steer = 0.0
        s = define_state(obs, prev_steer)

        env.render()
        done = False
        
        while not done:
            actions = []
            a = q.action(torch.from_numpy(s).float())
            
            target_steer = (a - 2) * (np.pi / 30)
            steer = (SMOOTHING_ALPHA * target_steer) + ((1.0 - SMOOTHING_ALPHA) * prev_steer)

            # v3 평가 모드 하이브리드 감속 기어 동기화
            if current_eval_max >= 10.0:
                if a == 2:
                    speed = current_eval_max
                elif a == 1 or a == 3:
                    speed = current_eval_max * 0.88
                else:
                    speed = current_eval_max * 0.52
            else:
                if a == 2:
                    speed = current_eval_max
                elif a == 1 or a == 3:
                    speed = current_eval_max * 0.85
                else:
                    speed = current_eval_max * 0.60

            actions.append([steer, speed])
            actions = np.array(actions)
            obs, r, done, info = env.step(actions)
            
            s_prime = define_state(obs, steer)
            s = s_prime
            prev_steer = steer
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