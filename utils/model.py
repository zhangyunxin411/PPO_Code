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


class LSTM(nn.Module):
    def __init__(self, cfg, hidden_dim, *args, **kwargs):
        super(LSTM, self).__init__()
        self.input_dim = cfg.n_states
        self.hidden_dim = hidden_dim
        self.mini_batch = cfg.mini_batch
        self.device = cfg.device

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True, device=cfg.device, *args, **kwargs)
        # 初始化网络参数，权重使用正交初始化，偏置初始化为0
        """for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)"""
        self.chunk_size = cfg.lstm_cfg.chunk_num

    # 注意：hidden_states是隐藏状态和细胞状态的合并
    def forward(self, input, hidden_states):
        # input形状为(batch_size, seq_len, input_size)
        # hidden_states为h+c,形状为(hidden_layer_num, mini_batch, 2*hidden_dim)
        if hidden_states is None:
            hidden_states = torch.zeros(1, self.mini_batch, self.chunk_size*self.hidden_dim, device=self.device)
        assert hidden_states.dim() == 3, f"hidden_states.dim()={hidden_states.dim()}, expected dim=3"
        assert input.dim() == 3, f"input.dim()={input.dim()}, expected dim=3"
        h_in = torch.chunk(hidden_states, self.chunk_size, dim=-1)   # 包括隐藏状态h和细胞状态c，将其且分开，输出(h,c)
        h_in = tuple(h.contiguous() for h in h_in)  # 转换成连续内存布局，提高性能
        output, new_hidden_state = self.lstm(input, h_in)
        new_hidden_state = torch.cat(new_hidden_state, dim=-1)  # 再将新的h和c拼接成一个张量
        new_hidden_state = new_hidden_state.squeeze(0)  # 去掉hidden_layer_num维度，变成(mini_batch, 2*hidden_dim)
        return output, new_hidden_state


class ActorCritic(nn.Module):
    def __init__(self, cfg, net_arch = None, hidden_dim = None):
        super(ActorCritic, self).__init__()
        self.cfg = cfg
        if cfg.net == 'MLP':
            assert net_arch!=None, "if use MLP, net_arch can't be None"
            self.mlp_extractor = MlpExtractor(net_arch, cfg.n_states, nn.Tanh, cfg.device)
            critic_dim = self.mlp_extractor.critic_last_dim
            actor_dim = self.mlp_extractor.actor_last_dim
        elif cfg.net == 'LSTM':
            assert hidden_dim!=None, "if use LSTM, hidden_dim can't be None"
            self.lstm = LSTM(cfg, hidden_dim)
            critic_dim = hidden_dim
            actor_dim = hidden_dim

        self.critic_fc = nn.Linear(critic_dim, 1)
        self.actor_fc = nn.Linear(actor_dim, cfg.n_actions)
        if cfg.env_continuous:      # 标准差为状态无关的，可随梯度更新的参数
            self.log_std = nn.Parameter(torch.ones(cfg.n_actions) * cfg.log_std_init, requires_grad=True)

    # 输入标准化后的状态，输出动作分布，价值
    def forward(self, s, hidden_state=None):
        if self.cfg.net == 'MLP':
            actor_in, critic_in = self.mlp_extractor(s)
        elif self.cfg.net == 'LSTM':
            lstm_out, new_hidden_state = self.lstm(s, hidden_state)
            actor_in = lstm_out
            critic_in = lstm_out

        if self.cfg.env_continuous:
            mean = 1.0 * torch.tanh(self.actor_fc(actor_in))
            if self.cfg.net == 'LSTM':
                mean = mean.view(-1, self.cfg.n_actions)
            std = torch.exp(self.log_std.clamp(min=-5, max=1e-4))
            prob = Normal(mean, std)
        else : 
            prob = F.softmax(self.actor_fc(actor_in), dim=-1)
            prob = Categorical(prob)
        value = self.critic_fc(critic_in)

        if self.cfg.net == 'MLP':
            return prob, value
        elif self.cfg.net == 'LSTM':
            return prob, value, new_hidden_state


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
            if key in exclude_keys:     # 跳过这几个属性不检查
                continue
            logger.debug(f"Save {key}")
            # 构建一个字典state包含网络和优化器的可学习参数
            if hasattr(value, 'state_dict'):
                state[f'{key}_state_dict'] = value.state_dict()
            else:
                state[key] = value
        torch.save(state, self.cfg.save_path)
        self._print_model_summary()
        logger.info(f"Save model to {self.cfg.save_path}")

    def load_model(self):
        with logger.catch(message="Model loading failed."):
            checkpoint = torch.load(self.cfg.save_path, map_location=self.cfg.device, weights_only=False)
            for key, value in checkpoint.items():
                exclude_keys = ['state_buffer', 'cfg', 'memory']
                if key in exclude_keys:
                    continue
                logger.debug(f"Load {key}")
                if key.endswith('_state_dict'):     # 匹配后缀是否相同
                    attr_name = key.replace('_state_dict', '')      # 去掉后缀
                    if hasattr(self, attr_name):
                        getattr(self, attr_name).load_state_dict(value)     # 由于只保存网络参数，因此将参数导入到网络和优化器
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
  
