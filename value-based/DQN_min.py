import gymnasium as gym
import torch as th
import numpy as np
import wandb
import time

wandb.init(project="minigrid", sync_tensorboard=True, monitor_gym=True, save_code=True)

batch_size = 64
lr = 0.0001
epsilon = 0.9
replay_memory_size = 10000
gamma = 0.9
target_update_iter = 200

env = gym.make('MiniGrid-MemoryS9-v0', render_mode='rgb')
env = env.unwrapped


device = th.device("cuda" if th.cuda.is_available() else "cpu")
n_action = 3

if isinstance(env.observation_space, gym.spaces.Dict):
    n_state = np.prod(env.observation_space['image'].shape)
else:
    n_state = np.prod(env.observation_space.shape)

hidden = 32

class Net(th.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = th.nn.Linear(n_state, hidden)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = th.nn.Linear(hidden, n_action)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = th.nn.functional.relu(x)
        out = self.out(x)
        return out

class ReplayMemory:
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
            return -1
        sam = np.random.choice(len(self.memory), batch_size)
        batch = [self.memory[i] for i in sam]
        return np.array(batch, dtype=object)

class DQN:
    def __init__(self):
        self.eval_q_net, self.target_q_net = Net().to(device), Net().to(device)
        self.replay_mem = ReplayMemory()
        self.iter_num = 0
        self.optimizer = th.optim.Adam(self.eval_q_net.parameters(), lr=lr)
        self.loss_fn = th.nn.MSELoss().to(device)
        self.loss_history = []

    def choose_action(self, qs):
        if np.random.uniform() < epsilon:
            return th.argmax(qs).tolist()
        else:
            return np.random.randint(0, n_action)

    def learn(self):
        if self.iter_num % target_update_iter == 0:
            self.target_q_net.load_state_dict(self.eval_q_net.state_dict())
        self.iter_num += 1

        batch = self.replay_mem.sample()
        if len(batch) == 0:
            return

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
        td_error = self.loss_fn(q_eval, q_target)

        self.optimizer.zero_grad()
        td_error.backward()
        self.optimizer.step()

        self.loss_history.append(td_error.item())

dqn = DQN()

def preprocess_state(state):
    if isinstance(state, dict):
        return state['image'].flatten()
    else:
        return state.flatten()

for episode in range(1000):
    start_time = time.time()  # 에피소드 시작 시간
    s = env.reset()
    s = preprocess_state(s[0])  # Initial state
    step = 0
    r = 0.0
    total_reward = 0.0
    episode_loss = 0
    episode_value = 0
    done = False
    while not done:
        step += 1
        env.render()  # 환경 시각화
        qs = dqn.eval_q_net(th.FloatTensor(s).to(device))
        a = dqn.choose_action(qs)
        s_, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        s_ = preprocess_state(s_)
        transition = [s.tolist(), a, [r], s_.tolist(), [done]]

        dqn.replay_mem.store_transition(transition)
        total_reward += r
        s = s_
        print("step:", step, "reward:", r)

        if dqn.replay_mem.size() > batch_size:
            dqn.learn()

        if done:
            break

    episode_loss = np.mean(dqn.loss_history[-step:]) if step > 0 else 0
    duration = time.time() - start_time
    wandb.log({
        "episode": episode,
        "reward": total_reward,
        "average_loss": episode_loss,
        "duration": duration,
    }, step=episode)

th.save(dqn.eval_q_net.state_dict(), "dqn_eval_q_net_min.pth")
th.save(dqn.target_q_net.state_dict(), "dqn_target_q_net_min.pth")

env.close()
wandb.finish()
