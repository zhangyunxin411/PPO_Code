from torch import nn
import torch
import numpy as np
from torch.nn import functional as F
from torch.distributions import Normal, Categorical
import os
from utils.buffer import Queue
from loguru import logger


def initialize_weights(layer, init_type='kaiming', nonlinearity='leaky_relu'):
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        if init_type == 'kaiming':                  # kaiming初始化，适合激活函数为ReLU, LeakyReLU, PReLU
            nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)
        elif init_type == 'xavier':
            nn.init.xavier_uniform_(layer.weight)   # xavier初始化, 适合激活函数为tanh和sigmoid
        elif init_type == 'orthogonal':
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))       # 正交初始化，适合激活函数为ReLU
        else:       
            raise ValueError(f"Unknown initialization type: {init_type}")
        
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
    return layer


# 全连接层
class MLP(nn.Module):
    def __init__(self,
                 dim_list,
                 activation=nn.PReLU(),
                 last_act=False,
                 use_norm=False,
                 linear=nn.Linear,
                 *args, **kwargs
                 ):
        super(MLP, self).__init__()
        assert dim_list, "Dim list can't be empty!"
        layers = []
        for i in range(len(dim_list) - 1):
            layer = initialize_weights(linear(dim_list[i], dim_list[i + 1], *args, **kwargs))
            layers.append(layer)
            if i < len(dim_list) - 2:
                if use_norm:
                    layers.append(nn.LayerNorm(dim_list[i + 1]))
                layers.append(activation)
        if last_act:
            if use_norm:
                layers.append(nn.LayerNorm(dim_list[-1]))
            layers.append(activation)
        # 将上述的网络连接起来
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


# 一种兼顾宽度和深度的全连接层，提取信息效率更高
class PSCN(nn.Module):
    def __init__(self, input_dim, output_dim, depth=4, linear=nn.Linear):
        super(PSCN, self).__init__()
        min_dim = 2 ** (depth - 1)
        assert depth >= 1, "depth must be at least 1"
        assert output_dim >= min_dim, f"output_dim must be >= {min_dim} for depth {depth}"
        assert output_dim % min_dim == 0, f"output_dim must be divisible by {min_dim} for depth {depth}"
        
        self.layers = nn.ModuleList()
        self.output_dim = output_dim
        in_dim, out_dim = input_dim, output_dim
        
        for i in range(depth):
            self.layers.append(MLP([in_dim, out_dim], last_act=True, linear=linear))
            in_dim = out_dim // 2
            out_dim //= 2 

    def forward(self, x):
        out_parts = []
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                split_size = int(self.output_dim // (2 ** (i + 1)))
                part, x = torch.split(x, [split_size, split_size], dim=-1)  # 按照参数中的列表大小，划分成指定数目的块
                out_parts.append(part)
            else:
                out_parts.append(x)

        out = torch.cat(out_parts, dim=-1)
        return out


class MlpExtractor(nn.Module):
    def __init__(self, net_arch : dict, input_dim, activation_fn, device):
        super(MlpExtractor, self).__init__()

        actor_net: list[nn.Module] = []
        critic_net: list[nn.Module] = []

        actor_arch = net_arch.get("actor")
        critic_arch = net_arch.get("critic")
        actor_last_dim = input_dim
        critic_last_dim = input_dim
        for i in actor_arch:
            actor_net.append(nn.Linear(actor_last_dim, i))
            actor_net.append(activation_fn())
            actor_last_dim = i
        for i in critic_arch:
            critic_net.append(nn.Linear(critic_last_dim, i))
            critic_net.append(activation_fn())
            critic_last_dim = i
        
        self.actor_last_dim = actor_last_dim
        self.critic_last_dim = critic_last_dim

        self.actor_net = nn.Sequential(*actor_net).to(device)
        self.critic_net = nn.Sequential(*critic_net).to(device)

    def forward(self, states):
        return self.actor_net(states), self.critic_net(states)


class ActorCritic(nn.Module):
    def __init__(self, cfg, net_arch):
        super(ActorCritic, self).__init__()
        # self.fc_head = PSCN(cfg.n_states, 256, depth=4)
        # self.critic_fc = MLP([256, 64, 1])
        # self.actor_fc = MLP([256, 64, cfg.n_actions])
        self.mlp_extractor = MlpExtractor(net_arch, cfg.n_states, nn.Tanh, cfg.device)
        self.critic_fc = nn.Linear(self.mlp_extractor.critic_last_dim, 1)
        self.actor_fc = nn.Linear(self.mlp_extractor.actor_last_dim, cfg.n_actions)
        self.env_continuous = cfg.env_continuous
        if cfg.env_continuous:      # 标准差为状态无关的，可随梯度更新的参数
            self.log_std = nn.Parameter(torch.ones(cfg.n_actions) * cfg.log_std_init, requires_grad=True)

    # 输入标准化后的状态，输出动作分布，价值
    def forward(self, s):
        actor_out, critic_out = self.mlp_extractor(s)         # s--PSCN-->x
        if self.env_continuous:
            mean = 1.0 * torch.tanh(self.actor_fc(actor_out))
            std = torch.exp(self.log_std.clamp(min=-5, max=1e-4))
            prob = Normal(mean, std)
        else : 
            prob = F.softmax(self.actor_fc(actor_out), dim=-1)
            prob = Categorical(prob)
        value = self.critic_fc(critic_out)
        return prob, value


# 管理模型加载与存储
class ModelLoader:
    def __init__(self, cfg):
        cfg.save_path = f'./checkpoints/{cfg.algo_name}_{cfg.env_name.replace("/", "-")}.pth'
        self.cfg = cfg
        if not os.path.exists(os.path.dirname(cfg.save_path)):
            os.makedirs(os.path.dirname(cfg.save_path))

    def save_model(self):
        state = {}
        for key, value in self.__dict__.items():
            exclude_keys = ['state_buffer', 'cfg', 'memory']
            if key in exclude_keys:
                continue
            logger.debug(f"Save {key}")
            if hasattr(value, 'state_dict'):
                state[f'{key}_state_dict'] = value.state_dict()
            else:
                state[key] = value
        torch.save(state, self.cfg.save_path)
        self._print_model_summary()
        logger.info(f"Save model to {self.cfg.save_path}")

    def load_model(self):
        with logger.catch(message="Model loading failed."):
            checkpoint = torch.load(self.cfg.save_path, map_location=self.cfg.device)
            for key, value in checkpoint.items():
                exclude_keys = ['state_buffer', 'cfg', 'memory']
                if key in exclude_keys:
                    continue
                logger.debug(f"Load {key}")
                if key.endswith('_state_dict'):
                    attr_name = key.replace('_state_dict', '')
                    if hasattr(self, attr_name):
                        getattr(self, attr_name).load_state_dict(value)
                else:
                    setattr(self, key, value)
            logger.info(f"Load model： {self.cfg.save_path}")
            
            
    def _print_model_summary(self):
        if hasattr(self, 'net'):
            num_params = sum(p.numel() for p in self.net.parameters())
            message = f"Model Summary: Number of parameters: {num_params}\n"
            for name, param in self.net.named_parameters():
                message += f"{name}: {param.numel()} parameters\n"
            logger.debug(message)
                

class StateManager:
    def __init__(self, buffer_size=100):
        self.state_buffer = Queue(buffer_size)

    def save_state(self, *args):
        self.state_buffer.put(args)

    def load_state(self):
        return self.state_buffer.sample()
  
