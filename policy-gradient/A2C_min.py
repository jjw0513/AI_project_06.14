import gym
import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import gymnasium as gym
import wandb
import time

wandb.init(project="minigrid", sync_tensorboard=True, monitor_gym=True, save_code=True)

lr = 0.0001
gamma = 0.9
hidden = 32
entropy_coef = 0.01
value_loss_coef = 0.5
max_grad_norm = 0.5
env = gym.make('MiniGrid-MemoryS9-v0', render_mode="rgb")
device = th.device("cuda" if th.cuda.is_available() else "cpu")
env = env.unwrapped
n_action = 3

# MiniGrid 환경의 observation_space는 Dict 타입일 수 있음
if isinstance(env.observation_space, gym.spaces.Dict):
    obs_shape = env.observation_space['image'].shape
else:
    obs_shape = env.observation_space.shape

class actor(nn.Module):  # policy net
    def __init__(self):
        super(actor, self).__init__()
        self.fc1 = nn.Linear(n_state, hidden)
        self.fc2 = nn.Linear(hidden, n_action)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        prob = self.softmax(x)
        return prob

class Q(nn.Module):
    def __init__(self):
        super(Q, self).__init__()
        self.conv1 = nn.Conv2d(obs_shape[2], 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * obs_shape[0] * obs_shape[1], hidden)
        self.fc2 = nn.Linear(hidden, n_action)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        q = self.fc2(x)
        return q

class V(nn.Module):
    def __init__(self):
        super(V, self).__init__()
        self.conv1 = nn.Conv2d(obs_shape[2], 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * obs_shape[0] * obs_shape[1], hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        v = self.fc2(x)
        return v

class critic(nn.Module):
    def __init__(self):
        super(critic, self).__init__()
        self.v = V()
        self.q = Q()

    def forward(self, x):
        v = self.v(x)
        q = self.q(x)
        advantage = q - v.repeat(1, q.size(1))
        return advantage

class AC():
    def __init__(self):
        self.actor = actor().to(device)
        self.critic = critic().to(device)

        self.Aoptimizer = th.optim.Adam(self.actor.parameters(), lr=lr)
        self.Qoptimizer = th.optim.Adam(self.critic.q.parameters(), lr=lr)
        self.Voptimizer = th.optim.Adam(self.critic.v.parameters(), lr=lr)

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.loss_history = []  # 손실 기록을 저장할 리스트

    def choose_action(self, s):
        s = self.process_state(s)
        s = th.FloatTensor(s).to(device)
        a_prob = self.actor(s)
        dist = Categorical(a_prob)
        action = dist.sample().item()
        return action

    def actor_learn(self, s, a, A):
        s = self.process_state(s)
        s = th.FloatTensor(s).to(device)
        a_prob = self.actor(s)[0, a]
        loss = -(th.log(a_prob) * A.detach())

        self.Aoptimizer.zero_grad()
        loss.backward()
        self.Aoptimizer.step()

    def critic_learn(self, transition):  # transition=[s,[r],[a],s_,[done]]
        s = self.process_state(transition[0])
        s = th.FloatTensor(s).to(device)
        r = transition[1][0]
        s_ = self.process_state(transition[3])
        s_ = th.FloatTensor(s_).to(device)
        done = transition[4][0]

        a = transition[2][0]
        q = self.critic.q(s)[0, a]
        v = self.critic.v(s)
        A = q - v
        v_ = self.critic.v(s_) * gamma + r
        if not done:
            q_target = th.max(self.critic.q(s_)) * gamma + r
            loss_q = (q - q_target.detach()) ** 2
        else:
            q_target = r
            loss_q = (q - q_target) ** 2
        loss_v = (v - v_.detach()) ** 2

        # Calculate entropy for actor
        a_prob = self.actor(s)
        dist = Categorical(a_prob)
        entropy = dist.entropy().mean()

        # Total loss
        loss = loss_q + self.value_loss_coef * loss_v - self.entropy_coef * entropy
        self.loss_history.append(loss.item())  # 손실을 기록

        self.Qoptimizer.zero_grad()
        self.Voptimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.Qoptimizer.step()
        self.Voptimizer.step()
        return A

    def process_state(self, s):
        if isinstance(s, dict):
            s = s['image']
        return s

ac = AC()
max_steps_per_episode = 1000  # 각 에피소드의 최대 스텝 수

for episode in range(1000):
    start_time = time.time()  # 에피소드 시작 시간
    s = env.reset()
    step = 0
    total_reward = 0
    episode_loss = 0
    episode_value = 0
    reward_received = 0
    if isinstance(s, tuple):
        s = s[0]

    done = False
    while not done:
        a = ac.choose_action(s)
        s_, r, terminated, truncated, _ = env.step(a)
        step += 1
        done = terminated or truncated
        total_reward += r
        transition = [s, [r], [a], s_, [done]]

        A = ac.critic_learn(transition)
        ac.actor_learn(s, a, A)
        print("epi : ", episode, "step : ", step, "total_reward: ", total_reward)

        episode_value += ac.critic.v(th.FloatTensor(ac.process_state(s)).to(device)).item()

        if done:
            break
        s = s_
        if isinstance(s, tuple):
            s = s[0]

    episode_loss = np.mean(ac.loss_history[-step:]) if step > 0 else 0
    duration = time.time() - start_time  # 에피소드 소요 시간

    wandb.log({
        "episode": episode,
        "reward": total_reward,
        "average_loss": episode_loss,
        "average_value": episode_value,
        "duration": duration,  # 에피소드 소요 시간 기록
    }, step=episode)

th.save(ac.actor.state_dict(), "actor_model_mini.pth")
th.save(ac.critic.state_dict(), "critic_model_mini.pth")

env.close()
wandb.finish()
