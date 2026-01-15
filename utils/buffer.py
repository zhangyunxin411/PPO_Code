import torch
import numpy as np

class ReplayBuffer_LSTM:
    def __init__(self, cfg):
        self.states = []
        self.actions = []
        self.old_log_probs = []
        self.gaes = []
        self.returns = []
        self.hidden_states = []

        self.rewards = []
        self.values = []
        self.terminateds = []
        self.dones = []

        self.next_value = None

        self.seq_len = cfg.lstm_cfg.seq_len
        self.num_seq = cfg.lstm_cfg.num_seq
        self.n_state = cfg.n_states
        self.n_action = cfg.n_actions
        self.cfg = cfg

    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.old_log_probs.clear()
        self.gaes.clear()
        self.returns.clear()
        self.hidden_states.clear()

        self.rewards.clear()
        self.values.clear()
        self.terminateds.clear()
        self.dones.clear()

    def store(self, state, action, reward, log_prob, value, terminated, done):
        self.states.append(state)
        self.actions.append(action)
        self.old_log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.terminateds.append(terminated)
        self.dones.append(done)

    def size(self):
        return len(self.states)
    
    def sample(self):
        # 在转成tensor之前先计算优势估计和return
        adv, v_target = self.compute_advantage()

        states = torch.tensor(np.array(self.states), dtype=torch.float32).to(self.cfg.device)
        actions = torch.tensor(np.array(self.actions), dtype=torch.float32).to(self.cfg.device)
        adv = torch.tensor(adv, dtype=torch.float32).to(self.cfg.device)
        v_target = torch.tensor(v_target, dtype=torch.float32).to(self.cfg.device)
        old_log_probs = torch.tensor(np.array(self.old_log_probs), dtype=torch.float32).to(self.cfg.device)
        hidden_states = torch.tensor(np.array(self.hidden_states), dtype=torch.float32).to(self.cfg.device)

        # 将收集到的数据reshape成(num_seq, seq_len)
        if self.cfg.env_continuous:
            states = states.view(self.num_seq, self.seq_len, self.n_state)
            actions = actions.view(self.num_seq, self.seq_len, self.n_action)
            old_log_probs = old_log_probs.view(self.num_seq, self.seq_len)
            adv = adv.view(self.num_seq, self.seq_len)
            v_target = v_target.view(self.num_seq, self.seq_len)
        else:
            states = states.view(self.num_seq, self.seq_len, self.n_state)
            actions = actions.view(self.num_seq, self.seq_len).type(torch.long)
            old_log_probs = old_log_probs.view(self.num_seq, self.seq_len)
            adv = adv.view(self.num_seq, self.seq_len)
            v_target = v_target.view(self.num_seq, self.seq_len)

        self.samples = states, actions, old_log_probs, adv, v_target, hidden_states
        
        return self.samples
    
    def compute_advantage(self):
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        values = np.array(self.values + [self.next_value])
        terminated = np.array(self.terminateds)
        
        gae = np.zeros_like(rewards)
        last_gae = 0
        for t in reversed(range(len(rewards))):
            td_error = rewards[t] + self.cfg.gamma * values[t + 1] * (1 - terminated[t]) - values[t]
            gae[t] = td_error + self.cfg.gamma * self.cfg.lamda * (1 - dones[t]) * last_gae
            last_gae = gae[t]
        returns = gae + values[:-1]
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)           # 优势估计标准化
            
        return gae, returns


class ReplayBuffer_on_policy:
    def __init__(self, cfg):
        self.cfg = cfg
        self.buffer = []
        self.samples = None
        
    def store(self, transitions):
        assert self.samples is None, 'Need to clear the buffer before storing new transitions.'
        self.buffer.append(transitions)
        
    def clear(self):
        self.buffer = []
        self.samples = None
        
    def size(self):
        return len(self.buffer)
    
    def compute_advantage(self, rewards, terminated, dones, values, next_values):
        with torch.no_grad():
            td_error = rewards + self.cfg.gamma * next_values * (1 - terminated) - values
            td_error = td_error.cpu().detach().numpy()

            dones = dones.cpu().detach().numpy()
            adv, gae = [], 0.0
            for delta, d in zip(td_error[::-1], dones[::-1]):
                gae = self.cfg.gamma * self.cfg.lamda * gae * (1 - d) + delta   # 这里的(1 - d)起作用前提done=true出现在中间元素，此时gae!=0
                adv.append(gae)
            adv.reverse()
            adv = torch.tensor(np.array(adv), device=self.cfg.device, dtype=torch.float32).view(-1, 1)
            v_target = adv + values                                 # td_target(lamada)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)           # 优势估计标准化

        return adv, v_target
        
    def sample(self):
        if self.samples is None:
            # 先将每一类数据解压成一个元组，再转成tensor
            states, actions, rewards, log_probs, values, next_values, terminated, dones = map(
                lambda x: torch.tensor(np.array(x), dtype=torch.float32, device=self.cfg.device), 
                zip(*self.buffer)
            )
            # 改变张量形状(batch_size,1)
            if self.cfg.env_continuous:
                actions, rewards, terminated, dones, log_probs, values, next_values = actions.view(-1, self.cfg.n_actions), rewards.view(-1, 1),\
                terminated.view(-1, 1), dones.view(-1, 1), log_probs.view(-1, 1), values.view(-1, 1), next_values.view(-1, 1)
            else:
                actions, rewards, terminated, dones, log_probs, values, next_values = actions.view(-1, 1).type(torch.long), \
                rewards.view(-1, 1), terminated.view(-1, 1), dones.view(-1, 1), log_probs.view(-1, 1), values.view(-1, 1), next_values.view(-1, 1)
            
            adv, v_target = self.compute_advantage(rewards, terminated, dones, values, next_values)
            self.samples = states, actions, log_probs, adv, v_target
        
        return self.samples


class ReplayBuffer_on_policy_v2:
    def __init__(self, cfg):
        self.cfg = cfg
        self.clear()

    def clear(self):
        episode_max_steps = self.cfg.max_steps
        self.buffer = {
            's': np.zeros([self.cfg.batch_size, episode_max_steps] + list(self.cfg.state_shape), dtype=np.float32),
            'a': np.zeros([self.cfg.batch_size, episode_max_steps], dtype=np.int64),
            'a_logprob': np.zeros([self.cfg.batch_size, episode_max_steps], dtype=np.float32),
            'r': np.zeros([self.cfg.batch_size, episode_max_steps], dtype=np.float32),
            'd': np.zeros([self.cfg.batch_size, episode_max_steps], dtype=np.float32),
            'dw': np.ones([self.cfg.batch_size, episode_max_steps], dtype=np.float32),
            'v': np.zeros([self.cfg.batch_size, episode_max_steps], dtype=np.float32),
            'v_': np.zeros([self.cfg.batch_size, episode_max_steps], dtype=np.float32),
            'active': np.zeros([self.cfg.batch_size, episode_max_steps], dtype=np.int8),
        }
        self.size = np.zeros(self.cfg.batch_size, dtype=int)
        self.episode_num = 0

    def store(self, transitions):
        s, a, r, d, dw, a_logprob, v, v_ = transitions
        self.buffer['s'][self.episode_num, self.size[self.episode_num]] = s
        self.buffer['a'][self.episode_num, self.size[self.episode_num]] = a
        self.buffer['a_logprob'][self.episode_num, self.size[self.episode_num]] = a_logprob
        self.buffer['r'][self.episode_num, self.size[self.episode_num]] = r
        self.buffer['d'][self.episode_num, self.size[self.episode_num]] = d
        self.buffer['dw'][self.episode_num, self.size[self.episode_num]] = dw
        self.buffer['v'][self.episode_num, self.size[self.episode_num]] = v
        self.buffer['v_'][self.episode_num, self.size[self.episode_num]] = v_
        self.buffer['active'][self.episode_num, self.size[self.episode_num]] = 1
        self.size[self.episode_num] += 1

    def next_episode(self):
        self.episode_num += 1

    def sample(self):
        max_episode_len = self.size.max()
        return (
            torch.tensor(self.buffer['s'][:, :max_episode_len], dtype=torch.float32, device=self.cfg.device),
            torch.tensor(self.buffer['a'][:, :max_episode_len], dtype=torch.long, device=self.cfg.device),
            torch.tensor(self.buffer['a_logprob'][:, :max_episode_len], dtype=torch.float32, device=self.cfg.device),
            torch.tensor(self.buffer['r'][:, :max_episode_len], dtype=torch.float32, device=self.cfg.device),
            torch.tensor(self.buffer['d'][:, :max_episode_len], dtype=torch.float32, device=self.cfg.device),
            torch.tensor(self.buffer['dw'][:, :max_episode_len], dtype=torch.float32, device=self.cfg.device),
            torch.tensor(self.buffer['v'][:, :max_episode_len], dtype=torch.float32, device=self.cfg.device),
            torch.tensor(self.buffer['v_'][:, :max_episode_len], dtype=torch.float32, device=self.cfg.device),
            torch.tensor(self.buffer['active'][:, :max_episode_len], dtype=torch.float32, device=self.cfg.device),
        )


class ReplayBuffer_off_policy:
    def __init__(self, cfg):
        self.buffer = np.empty(cfg.memory_capacity, dtype=object)
        self.is_full = False
        self.pointer = 0
        self.capacity = cfg.memory_capacity
        self.batch_size = cfg.batch_size
        self.device = cfg.device

    def store(self, transitions):
        self.buffer[self.pointer] = transitions
        self.pointer = (self.pointer + 1) % self.capacity
        if self.pointer == 0:
            self.is_full = True

    def clear(self):
        self.buffer = np.empty(self.capacity, dtype=object)
        self.pointer = 0
        self.is_full = False

    def sample(self):
        batch_size = min(self.batch_size, self.size())
        indices = np.random.choice(self.size(), batch_size, replace=False)
        samples = map(lambda x: torch.tensor(np.array(x), dtype=torch.float32,
                                             device=self.device), zip(*self.buffer[indices]))
        return samples
    
    def size(self):
        if self.is_full:
            return self.capacity
        return self.pointer
    
    
# numpy实现环形队列存储
class Queue:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = np.empty(buffer_size, dtype=object)
        self.index = 0
        self.filled = False

    def put(self, item):
        self.buffer[self.index] = item
        self.index = (self.index + 1) % self.buffer_size
        if self.index == 0:
            self.filled = True

    def sample(self):
        if not self.filled and self.index == 0:
            raise ValueError('Queue is empty!')
        max_index = self.buffer_size if self.filled else self.index
        idx = np.random.randint(0, max_index)
        return self.buffer[idx]

    def is_empty(self):
        return not self.filled and self.index == 0

    def is_full(self):
        return self.filled
    
    def size(self):
        return self.buffer_size if self.filled else self.index
    
    def capacity(self):
        return self.buffer_size
