import gymnasium as gym
import numpy as np
import torch
import time, sys
import random
from torch.utils.tensorboard import SummaryWriter
from gymnasium.wrappers import AtariPreprocessing
from utils.normalization import Normalization, RewardScaling
from utils.env_wrappers import PyTorchFrame
from utils.buffer import *
from loguru import logger

np.random.seed(int(time.time()))
logger.remove()
logger.add(sys.stdout, level='INFO', format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{message}</level>")

class BasicConfig:
    def __init__(self):
        self.render_mode = 'rgb_array'
        # self.train_eps = 500
        self.test_eps = 3
        self.eval_freq = 10
        self.max_steps = 20000
        self.total_step = 1000
        self.lr = 1e-4
        self.gamma = 0.99
        self.lamda = 0.95
        self.log_std_init = 0.0
        self.n_states = None
        self.n_actions = None
        self.action_bound = None
        self.env_continuous = False
        self.use_atari = False
        self.unwrapped = False
        self.load_model = False
        self.save_freq = 50
        self.on_policy = None
        self.seed = 42
        self.save_path = './checkpoints/model.pth'
        self.device = torch.device('cuda') \
            if torch.cuda.is_available() else torch.device('cpu')
        self.seed_set()
        
    def seed_set(self):
        random.seed(self.seed)        # Python内置随机
        np.random.seed(self.seed)     # NumPy
        torch.manual_seed(self.seed)  # PyTorch
        torch.cuda.manual_seed_all(self.seed)  # 如果使用GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def show(self):
        print('-' * 30 + 'Parameters' + '-' * 30)
        for k, v in vars(self).items():
            print(k, '=', v)
        print('-' * 60)

    
def log_monitors(writer, monitors, agent, phase, step):
    for key, value in monitors.items():
        if not np.isnan(value):
            writer.add_scalar(f'{phase}/{key}', value, global_step=step)

# 构建环境
def make_env(cfg, **kwargs):
    env = gym.make(cfg.env_name, render_mode=cfg.render_mode, **kwargs)
    s = env.observation_space.shape
    # 用于判断观测空间是否是一个单通道(灰度)或三通道(RGB)图像
    use_rgb = (len(s) == 3 and s[2] in [1, 3]) if s is not None else False
    if cfg.use_atari:
        frame_skip = 4 if 'NoFrameskip' in cfg.env_name else 1
        env = AtariPreprocessing(env, grayscale_obs=False, terminal_on_life_loss=True,
                                 scale_obs=True, frame_skip=frame_skip)
    if use_rgb:
        env = PyTorchFrame(env)

    if cfg.unwrapped:
        env = env.unwrapped
        
    logger.info(f'Observation Space = {env.observation_space}')
    logger.info(f'Action Space = {env.action_space}')
    
    cfg.state_shape = env.observation_space.shape
    cfg.n_states = int(env.observation_space.shape[0])
    # 根据动作空间 连续/离散 获取动作空间的形状
    env_continuous = isinstance(env.action_space, gym.spaces.Box)
    if env_continuous:
        cfg.env_continuous = env_continuous
        cfg.action_bound = env.action_space.high[0]
        cfg.n_actions = int(env.action_space.shape[0])
    else:
        cfg.n_actions = int(env.action_space.n)
    # 获取环境截止最大的时间步数，对于lunnarlander是None，取max_steps
    cfg.max_steps = int(env.spec.max_episode_steps or cfg.max_steps)
    return env


def train(env, agent, cfg):
    logger.info('Start training!')
    
    if cfg.load_model:
        agent.load_model()
    
    if not hasattr(agent, "state_norm"):
        agent.state_norm = Normalization(shape=env.observation_space.shape) # 状态标准化
        logger.debug('Add state normalization!')
    if not hasattr(agent, "reward_scaler"):
        agent.reward_scaler = RewardScaling(shape=1, gamma=cfg.gamma)       # 奖励缩放
        logger.debug('Add reward scaling!')

    cfg.on_policy = (
        isinstance(agent.memory, ReplayBuffer_on_policy) or 
        isinstance(agent.memory, ReplayBuffer_on_policy_v2) or
        isinstance(agent.memory, list) and isinstance(agent.memory[0], ReplayBuffer_on_policy)
    )
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # 创建SummaryWriter，用于tensorboard添加记录数据
    writer = SummaryWriter(f'./exp/{cfg.algo_name}_{cfg.env_name.replace("/", "-")}_{timestamp}')
    # 显示超参数
    cfg.show()
    
    while (agent.learn_step * cfg.batch_size) < cfg.total_step:
        ep_reward = 0.0
        agent.reward_scaler.reset()

        # 复位环境
        state, _ = env.reset(seed=cfg.seed)  
        state = agent.state_norm(state)

        if cfg.on_policy:
            action, log_prob, value = agent.choose_action(state)
        else:
            action = agent.choose_action(state)        
        
        while agent.memory.size() < cfg.batch_size:
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward             # 累积奖励，不打折扣

            reward = agent.reward_scaler(reward)[0]     # 缩放奖励
            next_state = agent.state_norm(next_state)

            # 存储一个时间步数据
            if cfg.on_policy:
                _action, _log_prob, _value = agent.choose_action(next_state)
                if terminated:
                    _value = 0.0
                transitions = (state, action, reward, log_prob, value, _value, terminated, done)
                agent.memory.store(transitions)     # 保存一个时间步数据
                if done:
                    next_state, _ = env.reset(seed=cfg.seed)  
                    next_state = agent.state_norm(next_state)
                    _action, _log_prob, _value = agent.choose_action(next_state)
                action, log_prob, value = _action, _log_prob, _value
            else:
                agent.memory.store((state, action, reward, next_state, done))
                action = agent.choose_action(next_state)
            
            state = next_state
        # update
        monitors = agent.update()

        log_monitors(writer, monitors, agent, 'train', agent.learn_step)
        # TODO：监控A网络的std，但是输出是多个动作，所以不好处理
        if cfg.env_continuous:
            log_monitors(writer, {'std_mean': torch.mean(torch.exp(agent.net.log_std)).item()}, agent, 'train', agent.learn_step)

        log_monitors(writer, {'reward': ep_reward}, agent, 'train', agent.learn_step)
        logger.info(f'Step:{agent.learn_step*cfg.batch_size}  Episode:{agent.learn_step}  Reward:{ep_reward:.0f}')

        # 每更新10次评估一次
        if (agent.learn_step + 1) % cfg.eval_freq == 0:
            tools = {'writer': writer}
            evaluate(env, agent, cfg, tools)
        # 每更新10次保存一次模型
        if (agent.learn_step + 1) % cfg.save_freq == 0:
            agent.save_model()    
            
    logger.info('Finish training!')
    agent.save_model()
    env.close()
    writer.close()

# 评估当前的策略，纯贪婪，不探索，选择概率最大的动作，以奖励累加（不打折）作为标准
def evaluate(env, agent, cfg, tools):
    ep_reward, ep_step, done = 0.0, 0, False
    # state, _ = env.reset(seed=np.random.randint(1, 2**31 - 1))
    state, _ = env.reset(seed=cfg.seed)
    writer = tools['writer']
    state = agent.state_norm(state, update=False)
    while not done:
        ep_step += 1
        action = agent.evaluate(state)      # 选动作贪婪
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = agent.state_norm(next_state, update=False)
        state = next_state
        ep_reward += reward
        done = terminated or truncated
    log_monitors(writer, {'reward': ep_reward}, agent, 'eval', agent.learn_step)   # 可能出现2条episode才会更新一次的状况


def test(env, agent, cfg):
    logger.info('Start test!')
    agent.load_model()
    for i in range(cfg.test_eps):
        ep_reward, ep_step, done = 0.0, 0, False
        state, _ = env.reset(seed=np.random.randint(1, 2**31 - 1))
        state = agent.state_norm(state, update=False)
        while not done:
            ep_step += 1
            action = agent.evaluate(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = agent.state_norm(next_state, update=False)
            state = next_state
            ep_reward += reward
            done = terminated or truncated
        logger.info(f'Episode:{i + 1}/{cfg.train_eps}  Reward:{ep_reward:.0f}  Step:{ep_step:.0f}')
    logger.info('Finish test!')
    env.close()
    
    
class BenchMark:
    @logger.catch(reraise=True)
    @staticmethod
    def train(algo, config):
        cfg = config()
        env = make_env(cfg)     # must be called before agent is created
        agent = algo(cfg, {"actor" : [64, 64], "critic" : [64, 64]})
        print(agent.net)
        train(env, agent, cfg)
    
    @logger.catch(reraise=True) 
    @staticmethod
    def test(algo, config):
        cfg = config()
        cfg.render_mode = 'human'
        env = make_env(cfg)
        agent = algo(cfg)
        test(env, agent, cfg)
        
    
 