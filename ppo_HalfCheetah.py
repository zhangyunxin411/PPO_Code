import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import BatchSampler, SubsetRandomSampler
from utils.model import *
from utils.buffer import ReplayBuffer_on_policy as ReplayBuffer
from utils.runner import *


class Config(BasicConfig):
    def __init__(self):
        super(Config, self).__init__()
        self.env_name = 'HalfCheetah-v5'
        self.env_continuous = True
        self.render_mode = 'rgb_array'
        self.algo_name = 'PPO'
        # self.train_eps = 2000       # episode_num
        self.batch_size = 2048      # 每次更新网络需要采样的最少数据
        self.total_step = 2000000
        self.mini_batch = 64
        self.epochs = 10
        self.clip = 0.2
        self.gamma = 0.99
        self.lamda = 0.95
        self.dual_clip = 3.0
        self.val_coef = 0.5         # 价值损失系数
        self.log_std_init = -0.5    # log_std的初始值
        #self.lr = Schedule("linear_schedule", init = 5e-4, final = 0.0)                  # 5e-4
        self.lr = 3e-4
        self.ent_coef = 0.0         # 熵损失系数
        self.grad_clip = 0.5        # 梯度裁剪
        self.load_model = False     # 是否加载模型
        self.seed = 42


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
    def __init__(self, cfg, net_arch):
        super().__init__(cfg)
        self.net = ActorCritic(cfg, net_arch)
        self.cfg = cfg
        # self.net = torch.jit.script(ActorCritic(cfg).to(cfg.device))
        if callable(self.cfg.lr):
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.cfg.lr(0.0), eps=1e-5)
        else:
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.cfg.lr, eps=1e-5)
        self.memory = ReplayBuffer(cfg)
        self.learn_step = 0

    @torch.no_grad()
    def choose_action(self, state):
        state = torch.tensor(state, device=self.cfg.device, dtype=torch.float).unsqueeze(0)
        prob, value = self.net(state)
        action = prob.sample()
        if self.cfg.env_continuous:
            log_prob = prob.log_prob(action).sum(dim=1, keepdim=True)         # 连续动作空间彼此独立
            return action.tolist()[0], log_prob.item(), value.item()       # 动作空间不唯一
        else:
            log_prob = prob.log_prob(action)
            return action.item(), log_prob.item(), value.item()

    @torch.no_grad()
    def evaluate(self, state):
        state = torch.tensor(state, device=self.cfg.device, dtype=torch.float).unsqueeze(0)
        prob, _ = self.net(state)
        if self.cfg.env_continuous:
            action = prob.mean.tolist()[0]            # 选择概率最大的那个动作，纯贪婪
        else:
            action = prob.mode.tolist()[0]            # 选择众数，也就是概率最大的动作
        return action
    
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

if __name__ == '__main__':
    BenchMark.train(PPO, Config)

