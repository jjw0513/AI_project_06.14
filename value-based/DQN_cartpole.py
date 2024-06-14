import gymnasium as gym
import torch as th
import numpy as np
import wandb

# wandb 초기화
wandb.init(project="cartpole")

batch_size = 50
lr = 0.001
episilon = 0.9
replay_memory_size = 10000
gamma = 0.9
target_update_iter = 100
total_episodes = 10000  #
max_steps_per_episode = 500

env = gym.make('CartPole-v1', render_mode="human")
env = env.unwrapped

# 환경이 제대로 초기화되었는지 확인
assert env.observation_space is not None, "환경의 observation_space가 None입니다. 환경 초기화에 문제가 있습니다."
assert env.action_space is not None, "환경의 action_space가 None입니다. 환경 초기화에 문제가 있습니다."

device = th.device("cuda" if th.cuda.is_available() else "cpu")
n_action = env.action_space.n
n_state = env.observation_space.shape[0]
hidden = 32

class net(th.nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.fc1 = th.nn.Linear(n_state, hidden)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = th.nn.Linear(hidden, n_action)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = th.nn.functional.relu(x)
        out = self.out(x)
        return out

class replay_memory():
    def __init__(self):
        self.memory_size = replay_memory_size
        self.memory = []
        self.cur = 0

    def size(self):
        return len(self.memory)

    def store_transition(self, trans):
        if len(self.memory) < self.memory_size:
            self.memory.append(trans)
        else:
            self.memory[self.cur] = trans
            self.cur = (self.cur + 1) % self.memory_size

    def sample(self):
        if len(self.memory) < batch_size:
            return []
        sam = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in sam]
        return np.array(batch, dtype=object)

class DQN(object):
    def __init__(self):
        self.eval_q_net, self.target_q_net = net().to(device), net().to(device)
        self.replay_mem = replay_memory()
        self.iter_num = 0
        self.optimizer = th.optim.Adam(self.eval_q_net.parameters(), lr=lr)
        self.loss = th.nn.MSELoss().to(device)
        self.loss_history = []

    def choose_action(self, qs):
        if np.random.uniform() < episilon:
            return th.argmax(qs).tolist()
        else:
            return np.random.randint(0, n_action)

    def greedy_action(self, qs):
        return th.argmax(qs)

    def learn(self):
        if self.iter_num % target_update_iter == 0:
            self.target_q_net.load_state_dict(self.eval_q_net.state_dict())
        self.iter_num += 1

        batch = self.replay_mem.sample()
        if len(batch) == 0:
            return  # 배치 크기가 작으면 학습하지 않음

        b_s = th.FloatTensor(np.vstack(batch[:, 0])).to(device)
        b_a = th.LongTensor(batch[:, 1].astype(int).tolist()).to(device)
        b_r = th.FloatTensor(np.vstack(batch[:, 2])).to(device)
        b_s_ = th.FloatTensor(np.vstack(batch[:, 3])).to(device)
        b_d = th.FloatTensor(np.vstack(batch[:, 4])).to(device)
        q_target = th.zeros((batch_size, 1)).to(device)
        q_eval = self.eval_q_net(b_s)
        q_eval = th.gather(q_eval, dim=1, index=th.unsqueeze(b_a, 1))
        q_next = self.target_q_net(b_s_).detach()
        for i in range(b_d.shape[0]):
            if int(b_d[i].tolist()[0]) == 0:
                q_target[i] = b_r[i] + gamma * th.unsqueeze(th.max(q_next[i], 0)[0], 0)
            else:
                q_target[i] = b_r[i]
        td_error = self.loss(q_eval, q_target)

        self.optimizer.zero_grad()
        td_error.backward()
        self.optimizer.step()

        # wandb 로그를 위한 손실 저장
        self.loss_history.append(td_error.item())

dqn = DQN()

for episode in range(total_episodes):
    s = env.reset()
    s = s[0]  # Initial state
    t = 0
    episode_reward = 0.0
    while t < max_steps_per_episode:
        t += 1
        env.render()  # 환경 시각화
        qs = dqn.eval_q_net(th.FloatTensor(s).to(device))
        a = dqn.choose_action(qs)
        s_, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        transition = [s.tolist(), a, [r], s_.tolist(), [done]]
        dqn.replay_mem.store_transition(transition)
        s = s_
        episode_reward += r
        if dqn.replay_mem.size() > batch_size:
            dqn.learn()
        if done:
            break

    avg_loss = np.mean(dqn.loss_history[-batch_size:]) if dqn.loss_history else 0
    # 상태만 가져와서 평균 값을 계산
    recent_states = [trans[0] for trans in dqn.replay_mem.memory[-batch_size:]]
    avg_value = np.mean([dqn.eval_q_net(th.FloatTensor(s).to(device)).max().item() for s in recent_states])
    wandb.log({
        "episode": episode,
        "reward": episode_reward,
        "average_loss": avg_loss,
        "average_value": avg_value,
        "duration": t
    }, step=episode)

    if episode % 100 == 0:  #매 100-step마다 test 진행
        total_test_reward = 0.0
        for i in range(10):
            t_s = env.reset()
            t_s = t_s[0]  # Initial state
            t_r = 0.0
            time = 0
            while time < max_steps_per_episode:
                time += 1
                t_qs = dqn.eval_q_net(th.FloatTensor(t_s).to(device))
                t_a = dqn.greedy_action(t_qs).item()
                ts_, tr, tdone, ttruncated, _ = env.step(t_a)
                t_r += tr
                if tdone or ttruncated:
                    break
                t_s = ts_
            total_test_reward += t_r
        avg_test_reward = total_test_reward / 10
        wandb.log({"episode": episode, "test_reward": avg_test_reward})
        print("episode:" + format(episode) + ", test score:" + format(avg_test_reward))

# 모델 저장
th.save(dqn.eval_q_net.state_dict(), "dqn_eval_q_net.pth")
th.save(dqn.target_q_net.state_dict(), "dqn_target_q_net.pth")

env.close()

wandb.finish()
