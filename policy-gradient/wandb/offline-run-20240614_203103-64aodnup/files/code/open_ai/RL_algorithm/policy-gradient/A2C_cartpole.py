import gym
import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import wandb

# Initialize wandb
wandb.init(project="cartpole", sync_tensorboard=True, monitor_gym=True, save_code=True)

actor_lr = 0.0001  # Actor learning rate
critic_lr = 0.005  # Critic learning rate
gamma = 0.99 # discount factor
hidden = 32
total_episodes = 10000
max_steps_per_episode = 500

config = {
    "actor_lr": actor_lr,
    "critic_lr": critic_lr,
    "gamma": gamma,
    "hidden": hidden,
    "total_episodes": total_episodes,
    "max_steps_per_episode": max_steps_per_episode
}

env = gym.make('CartPole-v1', render_mode="rgb_array")
env = env.unwrapped

device = th.device("cuda" if th.cuda.is_available() else "cpu")
n_action = env.action_space.n
n_state = env.observation_space.shape[0]

class Actor(nn.Module):  # policy net
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_state, hidden)
        self.fc2 = nn.Linear(hidden, n_action)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        prob = self.softmax(x)
        return prob

class Q(nn.Module):
    def __init__(self):
        super(Q, self).__init__()
        self.q = nn.Sequential(nn.Linear(n_state, hidden),
                               nn.ReLU(),
                               nn.Linear(hidden, n_action))

    def forward(self, x):
        q = self.q(x)
        return q

class V(nn.Module):
    def __init__(self):
        super(V, self).__init__()
        self.v = nn.Sequential(nn.Linear(n_state, hidden),
                               nn.ReLU(),
                               nn.Linear(hidden, 1))

    def forward(self, x):
        v = self.v(x)
        return v

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.v = V()
        self.q = Q()

    def forward(self, x):
        v = self.v(x)
        q = self.q(x)
        advantage = q - v.repeat(1, q.size(1))
        return advantage

class AC():
    def __init__(self):
        self.actor = Actor().to(device)
        self.critic = Critic().to(device)
        self.Aoptimizer = th.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.Qoptimizer = th.optim.Adam(self.critic.q.parameters(), lr=critic_lr)
        self.Voptimizer = th.optim.Adam(self.critic.v.parameters(), lr=critic_lr)
        self.loss_history = []

    def choose_action(self, s):
        s = th.FloatTensor(s).to(device)
        a_prob = self.actor(s)
        dist = Categorical(a_prob)
        action = dist.sample().item()
        return action

    def actor_learn(self, s, a, A):
        s = th.FloatTensor(s).to(device)
        a_prob = self.actor(s)[a]
        loss = -(th.log(a_prob) * A.detach())
        self.Aoptimizer.zero_grad()
        loss.backward()
        self.Aoptimizer.step()
        self.loss_history.append(loss.item())

    def critic_learn(self, transition):  # transition=[s,[r],[a],s_,[done]]
        s = th.FloatTensor(transition[0]).to(device)
        r = transition[1][0]
        s_ = th.FloatTensor(transition[3]).to(device)
        done = transition[4][0]
        a = transition[2][0]
        q = self.critic.q(s)[a]
        v = self.critic.v(s)
        A = q - v

        if done:
            v_ = th.tensor(r).to(device)  # 보상을 텐서로 변환
        else:
            v_ = self.critic.v(s_) * gamma + r

        if not done:
            q_target = th.max(self.critic.q(s_)) * gamma + r
            loss_q = (q - q_target.detach()) ** 2
        else:
            q_target = r
            loss_q = (q - q_target) ** 2
        loss_v = (v - v_.detach()) ** 2
        self.Qoptimizer.zero_grad()
        loss_q.backward()
        self.Qoptimizer.step()
        self.Voptimizer.zero_grad()
        loss_v.backward()
        self.Voptimizer.step()
        return A

ac = AC()

for episode in range(total_episodes):
    s = env.reset()
    total_reward = 0
    episode_loss = 0
    episode_value = 0
    done = False
    t = 0
    while not done and t < max_steps_per_episode:
        t += 1
        a = ac.choose_action(s)
        s_, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        total_reward += r
        transition = [s, [r], [a], s_, [done]]
        A = ac.critic_learn(transition)
        ac.actor_learn(s, a, A)
        episode_value += ac.critic.v(th.FloatTensor(s).to(device)).item()
        s = s_

    episode_loss = np.mean(ac.loss_history[-t:]) if t > 0 else 0
    episode_value /= t if t > 0 else 1

    # Log metrics to wandb at the end of each episode
    wandb.log({
        "episode": episode,
        "reward": total_reward,
        "average_loss": episode_loss,
        "average_value": episode_value,
        "duration": t  # 에피소드 당 스텝 수 기록
    }, step=episode)  # 여기서 step 인자를 명시적으로 설정하여 x축을 episode로 설정

    if episode % 10 == 0:
        print(f"Episode: {episode}, Reward: {total_reward}, Avg Loss: {episode_loss}, Avg Value: {episode_value}, Duration: {t}")

    if episode % 100 == 0:  # test
        total_test_reward = 0.0
        for i in range(10):
            t_s = env.reset()
            t_r = 0.0
            time = 0
            while time < max_steps_per_episode:
                time += 1
                t_qs = ac.actor(th.FloatTensor(t_s).to(device))
                t_a = ac.choose_action(t_s)
                ts_, tr, tdone, ttruncated, _ = env.step(t_a)
                t_r += tr
                if tdone or ttruncated:
                    break
                t_s = ts_
            total_test_reward += t_r
        avg_test_reward = total_test_reward / 10
        wandb.log({"episode": episode, "test_reward": avg_test_reward})
        print(f"Episode: {episode}, Test Reward: {avg_test_reward}")

# Save models
th.save(ac.actor.state_dict(), "actor_model.pth")
th.save(ac.critic.state_dict(), "critic_model.pth")

env.close()  # 모든 에피소드 종료 후 환경 닫기

wandb.finish()
