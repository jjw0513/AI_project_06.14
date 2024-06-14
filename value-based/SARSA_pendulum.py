import numpy as np
import gym
import torch as th
import wandb

# 하이퍼파라미터 설정
gamma = 0.9
epsilon = 0.9
lr = 0.001
target_update_iter = 100
total_episodes = 500  # 총 에피소드 수
max_steps_per_episode = 200  # 에피소드당 최대 스텝 수
log_interval = 100

# wandb 초기화
wandb.init(project="pendulum")

# 환경 설정
env = gym.make('Pendulum-v1', render_mode="human")
device = th.device("cuda" if th.cuda.is_available() else "cpu")
env = env.unwrapped
n_action = 9  # 9개 행동: -2, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0
n_state = 3
hidden = 256


# 신경망 정의
class Net(th.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Net, self).__init__()
        self.fc1 = th.nn.Linear(state_dim, hidden)
        self.out = th.nn.Linear(hidden, action_dim)

    def forward(self, x):
        x = th.nn.functional.relu(self.fc1(x))
        out = self.out(x)
        return out


# SARSA 에이전트 정의
class Sarsa():
    def __init__(self):
        self.net = Net(n_state, n_action).to(device)
        self.target_net = Net(n_state, n_action).to(device)
        self.iter_num = 0
        self.optimizer = th.optim.Adam(self.net.parameters(), lr=lr)
        self.loss_history = []

    def learn(self, s, a, s_, r, done):
        s = th.Tensor(s).to(device)
        s_ = th.FloatTensor(s_).to(device)
        eval_q = self.net(s)[a]
        target_q = self.target_net(s_).max(0)[0].item()
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

    def greedy_action(self, qs):
        action = th.argmax(qs).item()
        real_action = (action - 4) / 2  # Scale to range [-2, 2]
        return action, real_action

    def random_action(self):
        action = np.random.choice([n for n in range(9)])
        real_action = (action - 4) / 2  # Scale to range [-2, 2]
        return action, real_action

    def choose_action(self, qs):
        if np.random.rand() > epsilon:
            return self.random_action()
        else:
            return self.greedy_action(qs)


sarsa = Sarsa()

for episode in range(total_episodes):
    s= env.reset()  # Get the initial observation
    step_count = 0
    episode_reward = 0.0
    done = False
    qs = sarsa.net(th.Tensor(s).to(device))
    a, real_a = sarsa.choose_action(qs)

    while not done and step_count < max_steps_per_episode:  # Add step count limit
        s_, r, done, truncated, _ = env.step([real_a])  # Pass real action as a list
        a_next, real_a_next = sarsa.choose_action(sarsa.net(th.Tensor(s_).to(device)))

        sarsa.learn(s, a, s_, r, done)
        print("action:", a, "real_action:", real_a)
        s = s_
        a = a_next
        real_a = real_a_next
        episode_reward += r
        step_count += 1

        if done or truncated:
            break

        print("episode:", episode, "reward:", episode_reward)

    avg_loss = np.mean(sarsa.loss_history[-100:]) if sarsa.loss_history else 0
    print("episode:", episode, "reward:", episode_reward)

    # wandb 기록
    wandb.log({
        "episode": episode,
        "reward": episode_reward,
        "avg_loss": avg_loss,
        "duration": step_count
    })

env.close()

