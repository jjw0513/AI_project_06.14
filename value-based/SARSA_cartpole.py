import numpy as np
import gym
import torch as th
import wandb

wandb.init(project="cartpole")

gamma = 0.9
episilon = 0.9
lr = 0.001
target_update_iter = 100
total_episodes = 10000
max_steps_per_episode = 500
log_interval = 100
env = gym.make('CartPole-v1',render_mode = "rgb_array")
device = th.device("cuda" if th.cuda.is_available() else "cpu")
env = env.unwrapped
n_action = env.action_space.n
n_state = env.observation_space.shape[0]
hidden = 256


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

for episode in range(total_episodes):
    s = env.reset()
    t = 0
    episode_reward = 0.0
    qs = sarsa.net(th.Tensor(s).to(device))
    a = sarsa.choose_action(qs)
    while t < max_steps_per_episode:
        t += 1
        s_, r, done, _,_ = env.step(a)
        a = sarsa.learn(s, a, s_, r, done)
        s = s_
        episode_reward += r
        if done:
            break

    # 에피소드 끝날 때마다 wandb에 로그 기록
    avg_loss = np.mean(sarsa.loss_history[-100:]) if sarsa.loss_history else 0
    wandb.log({
        "episode": episode,
        "reward": episode_reward,
        "average_loss": avg_loss,
        "duration": t
    })

    if episode % log_interval == 0:  # test
        total_reward = 0.0
        for i in range(10):
            t_s = env.reset()
            t_r = 0.0
            time = 0
            while time < max_steps_per_episode:
                time += 1
                qs = sarsa.net(th.Tensor(t_s).to(device))
                a = sarsa.greedy_action(qs)
                ts_, tr, tdone, _, _ = env.step(a.tolist())
                t_r += tr
                if tdone:
                    break
                t_s = ts_
            total_reward += t_r
        avg_test_reward = total_reward / 10
        wandb.log({"test_reward": avg_test_reward})
        print("episode:" + format(episode) + ", test score:" + format(total_reward / 10))

# 모델 저장
th.save(sarsa.net.state_dict(), "sarsa_net.pth")
th.save(sarsa.target_net.state_dict(), "sarsa_target_net.pth")

env.close()  # 모든 에피소드 종료 후 환경 닫기

wandb.finish()
