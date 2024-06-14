import numpy as np
import gymnasium as gym
import torch as th
import wandb
import time

wandb.init(project="minigrid", sync_tensorboard=True, monitor_gym=True, save_code=True)

batch_size = 64
lr = 0.0001
episilon = 0.9
gamma = 0.9
target_update_iter = 200

env = gym.make('MiniGrid-MemoryS9-v0', render_mode='rgb')
env = env.unwrapped

# 환경이 제대로 초기화되었는지 확인
assert env.observation_space is not None, "환경의 observation_space가 None입니다. 환경 초기화에 문제가 있습니다."
assert env.action_space is not None, "환경의 action_space가 None입니다. 환경 초기화에 문제가 있습니다."

device = th.device("cuda" if th.cuda.is_available() else "cpu")
# n_action = env.action_space.n
n_action = 3
# MiniGrid 환경의 observation_space는 Dict 타입일 수 있음
if isinstance(env.observation_space, gym.spaces.Dict):
    n_state = np.prod(env.observation_space['image'].shape)
else:
    n_state = np.prod(env.observation_space.shape)

hidden = 32

class net(th.nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.fc1 = th.nn.Linear(n_state, hidden)
        self.out = th.nn.Linear(hidden, n_action)

    def forward(self, x):
        x = self.fc1(x)
        x = th.nn.functional.relu(x)
        out = self.out(x)
        return out


class Sarsa():
    def __init__(self):
        self.net, self.target_net = net().to(device), net().to(device)
        self.iter_num = 0
        self.optimizer = th.optim.Adam(self.net.parameters(), lr=lr)
        self.loss_history = []

    def learn(self, s, a, s_, r, done):
        s = th.Tensor(s).to(device)
        s_ = th.FloatTensor(s_).to(device)
        eval_q = self.net(s)[a]
        target_q = self.target_net(s_)
        target_a = self.choose_action(target_q)
        target_q = target_q[target_a]
        if not done:
            y = gamma * target_q + r
        else:
            y = r
        loss = (y - eval_q) ** 2
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.iter_num += 1
        if self.iter_num % target_update_iter == 0:
            self.target_net.load_state_dict(self.net.state_dict())
        self.loss_history.append(loss.item())
        return target_a

    def greedy_action(self, qs):
        return th.argmax(qs)

    def random_action(self):
        return np.random.randint(0, n_action)

    def choose_action(self, qs):
        if np.random.rand() > episilon:
            return self.random_action()
        else:
            return self.greedy_action(qs).tolist()


sarsa = Sarsa()


def preprocess_state(state):
    if isinstance(state, dict):
        return state['image'].flatten()
    else:
        return state.flatten()

for episode in range(1000):
    start_time = time.time()  # 에피소드 시작 시간
    s = env.reset()
    s = preprocess_state(s[0])  # Initial state
    qs = sarsa.net(th.FloatTensor(s).to(device))
    a = sarsa.choose_action(qs)
    step = 0
    r = 0.0
    total_reward = 0.0
    episode_loss = 0
    episode_value = 0
    done = False
    while not done:
        step += 1
        s_, r, terminated, truncated, _ = env.step(a)
        s_ = preprocess_state(s_)
        a = sarsa.learn(s, a, s_, r, done)
        done = terminated or truncated
        transition = [s, a, [r], s_, [done]]

        total_reward += r
        s = s_
        print("step:", step, "reward:", r)

        if done:
            break

    episode_loss = np.mean(sarsa.loss_history[-step:]) if step > 0 else 0
    duration = time.time() - start_time  # 에피소드 소요 시간
    wandb.log({
        "episode": episode,
        "reward": total_reward,
        "average_loss": episode_loss,
        "duration": duration,  # 에피소드 소요 시간 기록
    }, step=episode)



th.save(sarsa.eval_q_net.state_dict(), "sarsa_eval_q_net_min.pth")
th.save(sarsa.target_q_net.state_dict(), "sarsa_target_q_net_min.pth")

env.close()  # 모든 에피소드 종료 후 환경 닫기
wandb.finish()
