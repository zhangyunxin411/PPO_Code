import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import BatchSampler, SubsetRandomSampler
from utils.model import *
from utils.buffer import ReplayBuffer_on_policy as ReplayBuffer
from utils.buffer import ReplayBuffer_LSTM
from utils.runner import *


class Config(BasicConfig):
    def __init__(self):
        super(Config, self).__init__()
        self.env_name = 'HalfCheetah-v5'
        self.env_continuous = True
        self.render_mode = 'rgb_array'
        self.algo_name = 'PPO'
        self.net = 'LSTM'          # 'MLP' or 'LSTM'
        self.batch_size = 2048      # 每次更新网络需要采样的最少数据
        self.total_step = 2000000
        self.mini_batch = 64
        self.epochs = 10
        self.clip = 0.2
        self.gamma = 0.99
        self.lamda = 0.95
        self.dual_clip = 3.0
        self.val_coef = 0.5         # 价值损失系数
        self.log_std_init = -0.5      # log_std的初始值
        self.lr = Schedule("linear_schedule", init = 5e-4, final = 0.0)                  # 5e-4
        # self.lr = 3e-4
        self.ent_coef = 0.0         # 熵损失系数
        self.grad_clip = 0.5        # 梯度裁剪
        self.load_model = False     # 是否加载模型
        self.seed = 42
        self.test_eps = 5
        if self.net == 'LSTM':
            self.lstm_cfg = LSTM_Config(self.batch_size)
            

class LSTM_Config:
    def __init__(self, batch_size):
        self.chunk_num = 2
        self.hidden_dim = 64
        self.seq_len = 8
        self.num_seq = batch_size // self.seq_len


class Schedule:
    def __init__(self, schedual:str, init:float, final:float):
        self.schedual = schedual
        self.init = init
        self.final = final

    def __call__(self, progress):
        if self.schedual == "linear_schedule" :
            return self.linear_schedule(progress)

    def linear_schedule(self, progress):
        return self.init + (self.final - self.init) * progress


class PPO(ModelLoader):
    def __init__(self, cfg, net_arch = None, hidden_dim = None):
        super().__init__(cfg)
        self.net = ActorCritic(cfg, net_arch, hidden_dim)
        self.cfg = cfg
        if callable(self.cfg.lr):
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.cfg.lr(0.0), eps=1e-5)
        else:
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.cfg.lr, eps=1e-5)
        
        if cfg.net == 'MLP':
            self.memory = ReplayBuffer(cfg)
        elif cfg.net == 'LSTM':
            self.memory = ReplayBuffer_LSTM(cfg)
        self.learn_step = 0

    @torch.no_grad()
    def choose_action(self, state, hidden_state=None):
        state = torch.tensor(state, device=self.cfg.device, dtype=torch.float).unsqueeze(0) # 添加batch维度
        if self.cfg.net == 'MLP':
            prob, value = self.net(state)
        elif self.cfg.net == 'LSTM':
            assert hidden_state is not None, "hidden_state can't be None when using LSTM net"
            if state.dim() == 2 :
                # state的形状应为(batch_size, seq_len, input_size)
                state = state.unsqueeze(1)      # 添加 seq_len 维度
            if hidden_state.dim() == 2:
                # hidden_state的形状为(hidden_layer_num, mini_batch, 2*hidden_dim)
                hidden_state = hidden_state.unsqueeze(0)    # 添加 hidden_layer_num 维度
            prob, value, hidden_state = self.net(state, hidden_state)
        action = prob.sample()

        if self.cfg.net == 'MLP':
            if self.cfg.env_continuous:
                log_prob = prob.log_prob(action).sum(dim=1, keepdim=True)         # 连续动作空间彼此独立
                return action.tolist()[0], log_prob.item(), value.item()       # 动作空间不唯一
            else:
                log_prob = prob.log_prob(action)
                return action.item(), log_prob.item(), value.item()
            
        elif self.cfg.net == 'LSTM':
            assert hidden_state is not None, "hidden_state can't be None when using LSTM net"
            if self.cfg.env_continuous:
                log_prob = prob.log_prob(action).sum(dim=1, keepdim=True)
                return action.tolist()[0], log_prob.item(), value.item(), hidden_state
            else:
                log_prob = prob.log_prob(action)
                return action.item(), log_prob.item(), value.item(), hidden_state
            
    @torch.no_grad()
    def evaluate(self, state, hidden_state=None):
        state = torch.tensor(state, device=self.cfg.device, dtype=torch.float).unsqueeze(0)
        if self.cfg.net == 'MLP':
            prob, _ = self.net(state)
        elif self.cfg.net == 'LSTM':
            assert hidden_state is not None, "hidden_state can't be None when using LSTM net"
            hidden_state = hidden_state.unsqueeze(0)    # 添加 hidden_layer_num 维度
            state = state.unsqueeze(1)      # 添加 seq_len 维度
            prob, _, new_hidden_state = self.net(state, hidden_state)

        if self.cfg.env_continuous:
            action = prob.mean.tolist()[0]            # 选择概率最大的那个动作，纯贪婪
        else:
            action = prob.mode.tolist()[0]            # 选择众数，也就是概率最大的动作
        return action, new_hidden_state
    
    def update_lr(self, progress):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.cfg.lr(progress)
    
    def update(self):
        states, actions, old_probs, adv, v_target = self.memory.sample()
        losses = np.zeros(5)
        progress = (self.learn_step * self.cfg.batch_size) / self.cfg.total_step
        if isinstance(self.cfg.lr, Schedule):
            self.update_lr(progress)
        for _ in range(self.cfg.epochs):
            # 批处理：随机采样，不放回，每次采样mini_batch个元素，得到的是序号组成的序列，也就是将memory中的数据分成一个个mini_batch大小序列
            for indices in BatchSampler(SubsetRandomSampler(range(self.memory.size())), self.cfg.mini_batch, drop_last=False):
                actor_prob, value = self.net(states[indices])
                if self.cfg.env_continuous:
                    log_probs = actor_prob.log_prob(actions[indices]).sum(dim=-1,keepdim=True)     # 新策略
                    #old_probs = old_probs.squeeze(dim=-1)
                else:
                    log_probs = actor_prob.log_prob(actions[indices].squeeze(dim=-1))   # Categorical分布输入期望形状为[mini_batch,]
                    log_probs = log_probs.unsqueeze(-1)     # 但是old_probs形状为[mini_batch, 1]需要保持一致
                assert old_probs[indices].shape == log_probs.shape , \
                    f"old_probs.shape is {old_probs[indices].shape}, but log_probs.shape is {log_probs.shape}"
                
                ratio = torch.exp(log_probs - old_probs[indices])   # [mini_batch,]
                assert ratio.shape == adv[indices].shape, \
                    f"ratio.shape is {ratio.shape}, but adv.shape is {adv[indices].shape}"
                
                surr1 = ratio * adv[indices]
                surr2 = torch.clamp(ratio, 1 - self.cfg.clip, 1 + self.cfg.clip) * adv[indices]
                min_surr = torch.min(surr1, surr2).mean()

                # dual_clip
                clip_loss = -torch.mean(torch.where(
                    adv[indices] < 0,
                    torch.max(min_surr, self.cfg.dual_clip * adv[indices]),
                    min_surr
                ))

                # TODO:熵系数衰减
                

                value_loss = F.mse_loss(v_target[indices], value)
                if self.cfg.env_continuous:
                    entropy_loss = -actor_prob.entropy().sum(dim=1).mean()
                else:
                    entropy_loss = -actor_prob.entropy().mean()
                loss = clip_loss + self.cfg.val_coef * value_loss + self.cfg.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
                self.optimizer.step()

                losses[0] += loss.item()
                losses[1] += clip_loss.item()
                losses[2] += value_loss.item()
                losses[3] += entropy_loss.item()
                
        self.memory.clear()
        self.learn_step += 1

        return {
            'total_loss': losses[0] / self.cfg.epochs,
            'clip_loss': losses[1] / self.cfg.epochs,
            'value_loss': losses[2] / self.cfg.epochs,
            'entropy_loss': losses[3] / self.cfg.epochs / (self.cfg.batch_size // self.cfg.mini_batch),
            'advantage': adv.mean().item(),
            'lr': self.optimizer.param_groups[0]['lr'],
        }
    
    def lstm_update(self):
        # s和a的形状为(num_seq, seq_len, dim)其他形状为(num_seq, seq_len)hidden_states形状为(2048,128)
        states, actions, old_log_probs, adv, v_target, hidden_states = self.memory.sample()
        losses = np.zeros(5)
        progress = (self.learn_step * self.cfg.batch_size) / self.cfg.total_step
        if isinstance(self.cfg.lr, Schedule):
            self.update_lr(progress)
        
        init_hidden_states = hidden_states[::self.cfg.lstm_cfg.seq_len]  # 每个序列的初始隐藏状态
        assert len(init_hidden_states) == self.cfg.lstm_cfg.num_seq, "len(init_hidden_states) not equal num_seq"
        # 采样到的数据被分成256条长为8的序列
        for _ in range(self.cfg.epochs):
            perm = torch.randperm(self.cfg.lstm_cfg.num_seq, device=self.cfg.device) # 随机打乱序列顺序
            for start in range(0, self.cfg.lstm_cfg.num_seq, self.cfg.mini_batch):
                end = start + self.cfg.mini_batch
                indices = perm[start:end]

                hidden_states_batch = init_hidden_states[indices]
                states_batch = states[indices]
                action_batch = actions[indices]
                old_log_probs_batch = old_log_probs[indices]
                adv_batch = adv[indices]
                v_target_batch = v_target[indices]

                hidden_states_batch = hidden_states_batch.unsqueeze(0)  # 添加 hidden_layer_num 维度
                assert hidden_states_batch.shape == (1, self.cfg.mini_batch, 2*self.cfg.lstm_cfg.hidden_dim), \
                    f"hidden_states_batch.shape={hidden_states_batch.shape}, expected shape={(1, self.cfg.mini_batch, 2*self.cfg.lstm_cfg.hidden_dim)}"
                actor_prob, value, _ = self.net(states_batch, hidden_states_batch)

                # 展平以计算损失
                value_flat = value.view(-1)
                actions_flat = action_batch.view(-1, self.cfg.n_actions)
                old_log_probs_flat = old_log_probs_batch.view(-1)
                old_log_probs_flat = old_log_probs_flat.unsqueeze(-1)
                adv_flat = adv_batch.view(-1)
                adv_flat = adv_flat.unsqueeze(-1)
                v_target_flat = v_target_batch.view(-1)

                if self.cfg.env_continuous:
                    log_probs = actor_prob.log_prob(actions_flat).sum(dim=-1,keepdim=True)     # 新策略
                else:
                    log_probs = actor_prob.log_prob(actions_flat.squeeze(dim=-1))   # Categorical分布输入期望形状为[mini_batch,]
                    log_probs = log_probs.unsqueeze(-1)     # 但是old_probs形状为[mini_batch*seq_len, 1]需要保持一致
                assert old_log_probs_flat.shape == log_probs.shape , \
                    f"old_probs.shape is {old_log_probs_flat.shape}, but log_probs.shape is {log_probs.shape}"
                
                ratio = torch.exp(log_probs - old_log_probs_flat)   # [mini_batch*seq_len,]
                assert ratio.shape == adv_flat.shape, \
                    f"ratio.shape is {ratio.shape}, but adv.shape is {adv_flat.shape}"
                
                surr1 = ratio * adv_flat
                surr2 = torch.clamp(ratio, 1 - self.cfg.clip, 1 + self.cfg.clip) * adv_flat
                min_surr = torch.min(surr1, surr2).mean()

                # dual_clip
                clip_loss = -torch.mean(torch.where(
                    adv_flat < 0,
                    torch.max(min_surr, self.cfg.dual_clip * adv_flat),
                    min_surr
                ))

                # TODO:熵系数衰减

                assert v_target_flat.shape == value_flat.shape, \
                    f"v_target_flat.shape={v_target_flat.shape}, but value_flat.shape={value_flat.shape}"
                value_loss = F.mse_loss(v_target_flat, value_flat)
                if self.cfg.env_continuous:
                    entropy_loss = -actor_prob.entropy().sum(dim=1).mean()
                else:
                    entropy_loss = -actor_prob.entropy().mean()
                loss = clip_loss + self.cfg.val_coef * value_loss + self.cfg.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
                self.optimizer.step()

                losses[0] += loss.item()
                losses[1] += clip_loss.item()
                losses[2] += value_loss.item()
                losses[3] += entropy_loss.item()
                
        self.memory.clear()
        self.learn_step += 1

        return {
            'total_loss': losses[0] / self.cfg.epochs,
            'clip_loss': losses[1] / self.cfg.epochs,
            'value_loss': losses[2] / self.cfg.epochs,
            'entropy_loss': losses[3] / self.cfg.epochs / (self.cfg.batch_size // self.cfg.mini_batch),
            'advantage': adv.mean().item(),
            'lr': self.optimizer.param_groups[0]['lr'],
        }

if __name__ == '__main__':
    # BenchMark.train(PPO, Config)
    BenchMark.train_lstm(PPO, Config)
    # BenchMark.test(PPO, Config)

