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
    os.makedirs(work_dir + "_ddqn_v2", exist_ok=True)

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

    # 💡 [커리큘럼 락 메커니즘 변수 정의]
    current_max_speed = 8.0      # 안전하게 5.0m/s로 학습 빌드업 시작
    target_top_speed = 20.0      # 최종 도달 목표 최고 속도
    consecutive_success = 0      # 실시간 연속 완주 성공 횟수 체크 카운터
    required_success = 3         # 속도 락 해제를 위한 필수 연속 완주 조건
    speed_increment = 0.2        # 뇌가 패닉에 빠지지 않도록 올리는 안전한 속도 스텝 크기

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
            
            # 고속 원심력 극복을 위한 비선형 감속 기어비 세팅
            if a == 2:
                speed = current_max_speed          # 직진 주로: 완전 개방
            elif a == 1 or a == 3:
                speed = current_max_speed * 0.85   
            else:
                speed = current_max_speed * 0.6   # 급코너: 55% 하드 브레이킹

            actions.append([steer, speed])
            actions = np.array(actions)
            obs, r, done, info = env.step(actions)
            
            s_prime = define_state(obs, steer)
            done_mask = 0.0 if done else 1.0

            # 베이스라인과의 공정한 정량 비교를 위한 정석 스케일 r/100.0 원복
            memory.put((s, a, r / 100.0, s_prime, done_mask))
            
            s = s_prime
            prev_steer = steer  # 조향 연속성 동기화
            laptime += r 

            if done:
                laptimes.append(laptime)
                plot_durations(laptimes)
                lap = round(obs['lap_times'][0], 3)

                # 💡 [연속 완주 기반 속도 해금 제어부]
                if int(obs['lap_counts'][0]) == 2:
                    consecutive_success += 1  # 연속 완주 달성 성공 카운트업
                    
                    # 5연속 주행을 완벽히 소화해 내서 순수 실력이 증명됐다면
                    if consecutive_success >= required_success:
                        if current_max_speed < target_top_speed:
                            current_max_speed = min(target_top_speed, current_max_speed + speed_increment)
                        consecutive_success = 0  # 속도가 승격되었으므로 새 레벨 적응을 위해 카운터 초기화
                    
                    # 베스트 랩 갱신 조건 만족 시 가중치 백업 (파일명에 출발 idx 및 maxV 명시)
                    if fastlap > lap:        
                        fastlap = lap
                        save_name = f"{work_dir}_ddqn_v2/fast-model_{lap:.2f}sec_idx{random_idx}_maxV{current_max_speed:.1f}_epi{n_epi}.pt"
                        torch.save(q.state_dict(), save_name)
                else:
                    # 억까 자리에 스폰되었든, 조향 실수든 도중에 한 번이라도 터지면 카운터 무조건 가차 없이 '리셋'
                    consecutive_success = 0

                break

        if memory.size() > train_start:
            train(q, q_target, memory, optimizer, beta)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            # 연속 완주 현황(Streak: 현재카운트/5)을 실시간 모니터링할 수 있도록 프린트문 고도화
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%, max_speed : {:.1f}m/s (Streak: {}/{})"
                  .format(n_epi, laptime, memory.size(), epsilon * 100, current_max_speed, consecutive_success, required_success))

    print('train finish')
    env.close()

def eval():
    env = gym.make('f110_gym:f110-v3',
                   map="{}/maps/{}".format(current_dir, RACETRACK),
                   map_ext=".png", num_agents=1)

    q = Qnet()
    # 📝 [테스트 가이드] 중간 점검할 때 검증하고 싶은 최신 pt 파일명을 여기에 교체해 주시면 됩니다.
    q.load_state_dict(torch.load("26-05-23_ddqn_v2/fast-model_60.52sec_idx192_maxV10.4_epi8319.pt"))

    # [안전 출발선 배치] eval 모드 실행 시 억까 당하지 않도록 트랙 선상 위의 정방향 안전 라인에서 출발시킵니다.
    start_idx = 192
    start_x = waypoints[start_idx][0]
    start_y = waypoints[start_idx][1]
    next_idx = (start_idx + 1) % num_waypoints
    dx = waypoints[next_idx][0] - start_x
    dy = waypoints[next_idx][1] - start_y
    start_theta = np.arctan2(dy, dx)
    
    poses = np.array([[start_x, start_y, start_theta]])
    
    # [테스트 가이드] 내가 가져온 모델 가중치 파일 이름에 적힌 maxV 값을 여기에 그대로 동기화해 주세요.
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
            
            # train과 완전 일치하게 연동시킨 스무딩 공식
            target_steer = (a - 2) * (np.pi / 30)
            steer = (SMOOTHING_ALPHA * target_steer) + ((1.0 - SMOOTHING_ALPHA) * prev_steer)

            # train과 동일한 비율의 감속 크루징 기어 매핑
            if a == 2:
                speed = current_eval_max
            elif a == 1 or a == 3:
                speed = current_eval_max * 0.85 # 0.85
            else:
                speed = current_eval_max * 0.6 # 0.60

            actions.append([steer, speed])
            actions = np.array(actions)
            obs, r, done, info = env.step(actions)
            
            s_prime = define_state(obs, steer)
            s = s_prime
            prev_steer = steer
            env.render(mode='human_fast')

            if done:
                lap = round(obs['lap_times'][0], 3)
                print(f"[EVAL {t}] 실제 물리 엔진 정량 랩타임: {lap:.3f} sec (완주 판단: {int(obs['lap_counts'][0]) == 2})")
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
