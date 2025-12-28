# reference: https://github.com/CORE-Robotics-Lab/ICCT/blob/main/icct/runfiles/train.py
# modified to leverage PPO instead of SAC or TD3, directly apply to MLPs+PPO+action chunking.

import copy
import argparse
import random
import os
import time

import torch as th
import gymnasium as gym
import highway_env
import numpy as np
import wandb

from typing import Callable
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CallbackList
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.torch_layers import CombinedExtractor, FlattenExtractor
from temp_interp.rl_helpers.save_after_ep_callback import EpCheckPointCallback
from temp_interp.rl_helpers.ppo import PPO
import temp_interp.envs.lunar_lander_hard
#### might have issues b/c hpc isn't ubuntu, figure out later
# from flow.utils.registry import make_create_env
# from temp_interp.envs.accel_ring import ring_accel_params
# from temp_interp.envs.accel_ring_multilane import ring_accel_lc_params
# from temp_interp.envs.accel_figure8 import fig8_params
####

def make_env(env_id, gamma=None):
    def thunk():
        if env_id == 'figure8':
            create_env, _ = make_create_env(params=fig8_params, version=0)
            env = create_env()
        elif env_id == 'LunarLanderHard': 
            env = gym.make(env_id, continuous=True, enable_wind=True, wind_power=20.0, turbulence_power=2.0)
        else: env = gym.make(env_id)
        if gamma: env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        return env
    return thunk

# https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value
    return func

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ICCT Training')
    parser.add_argument('--env_name', help='environment to run on', type=str, default='lunar')
    parser.add_argument('--num_envs', help='Number of parallel environments to run', type=int, default=1)
    parser.add_argument('--seed', help='the seed number to use', type=int, default=42)
    # PPO kwargs
    parser.add_argument('--abstraction_type', help='temp-ensemble or temp-pred', default=None)
    parser.add_argument('--gpu', help='if run on a GPU', action='store_true', default=False)
    parser.add_argument('--n_steps', help='Number of steps per batch', type=int, default=2048)
    parser.add_argument('--lr', help='learning rate', type=float, default=3e-4)
    parser.add_argument('--batch_size', help='batch size', type=int, default=256)
    parser.add_argument('--gamma', help='the discount factor', type=float, default=0.9999)
    parser.add_argument('--clip-range', help='the clip coef of the gradient update', type=float, default=0.2)
    parser.add_argument('--clip-range-vf', help='the clip range of the value function, must be tuned depending on the env rewards', type=float, default=None)
    parser.add_argument('--ent-coef', help='the entropy coefficient in PPO', type=float, default=0.1)
    parser.add_argument('--training_steps', help='total steps for training the model', type=int, default=500000)
    # evaluation and model saving
    parser.add_argument('--min_reward', help='minimum reward to save the model', type=int)
    parser.add_argument('--save_path', help='the path of saving the model', type=str, default='test')
    parser.add_argument('--n_eval_episodes', help='the number of episodes for each evaluation during training', type=int, default=5)
    parser.add_argument('--eval_freq', help='evaluation frequence of the model', type=int, default=1500)
    parser.add_argument('--log_interval', help='the number of episodes before logging', type=int, default=4)
    parser.add_argument('--use_wandb', help='whether to log using wandb instead of raw tensorboard', type=bool, default=True)

    args = parser.parse_args()
    assert args.abstraction_type is not None, print("ERROR: Abstraction type not set.")
    set_random_seed(args.seed) # can add: using_cuda=True
    if args.env_name == 'lunar': env_id = 'LunarLanderContinuous-v3'
    elif args.env_name == 'lunar-hard': env_id = 'LunarLanderHard'
    elif args.env_name == 'cart': env_id = 'InvertedPendulum-v5' # update to v5 for compatibility
    elif args.env_name == 'lane_keeping': env_id = 'lane-keeping-v0'
    elif args.env_name == 'figure8': env_id = 'figure8'
    envs = make_vec_env(make_env(env_id, gamma=args.gamma), n_envs=args.num_envs)
    envs.seed(seed=args.seed)
    eval_env = make_env(env_id)()
    eval_env.reset(seed=args.seed)

    sweep_config = {
        'method': 'bayes',
        'parameters': {
            'ddt_lr': {
                'distribution': 'uniform',
                'min': 1e-4,
                'max': 9e-4
            },
            'lr': {
                'distribution': 'uniform',
                'min': 1e-4,
                'max': 9e-4
            },
            'clip_range': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 0.3
            },
            'time_horizon': {
                'values': [10]
            },
            'num_leaves': {
                'values': [2, 4, 8, 16, 20]
            },
        }
    }
    if args.abstraction_type == 'temp-ensemble':
        sweep_config['metric'] = {'name': 'eval/mean_reward', 'goal': 'maximize'}
        sweep_config['parameters']['decay'] = {'distribution': 'uniform', 'min': 0.5, 'max': 1.0}
    elif args.abstraction_type == 'temp-pred':
        sweep_config['metric'] = {'name': 'eval/mean_total', 'goal': 'maximize'}
        sweep_config['parameters']['curriculum_coef'] = {'distribution': 'uniform', 'min': 0.1, 'max': 1.0}

    def train():
        ## wandb setup
        run_name = f"{env_id}__{args.seed}__{int(time.time())}"
        run = wandb.init(name=run_name, sync_tensorboard=True)
        config = wandb.config

        log_dir = args.save_path
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        eval_monitor_file_path = log_dir + 'eval_mlp_' + f'_seed{args.seed}'
        monitor_eval_env = Monitor(eval_env, eval_monitor_file_path)
        callback = EpCheckPointCallback(eval_env=monitor_eval_env, best_model_save_path=log_dir, n_eval_episodes=args.n_eval_episodes,
                                        eval_freq=args.eval_freq, minimum_reward=args.min_reward)
        wandb_callback = WandbCallback(model_save_freq=args.eval_freq, model_save_path=log_dir, verbose=2)
        callback = CallbackList([callback, wandb_callback])

        if args.gpu:
            args.device = 'cuda'
        else:
            args.device = 'cpu'
            
        if args.env_name == 'lane_keeping':
            features_extractor = CombinedExtractor
        else:
            features_extractor = FlattenExtractor

        if args.env_name == 'cart':
            args.fs_submodel_version = 1
        else:
            args.fs_submodel_version = 0

        policy_kwargs = {
            'features_extractor_class': features_extractor,
            # 'ddt_kwargs': ddt_kwargs,
            'net_arch': {'vf': [64, 64], 'pi': [64, 64]},
            'activation_fn': th.nn.Tanh,
            'time_horizon': config.time_horizon,
            'abstraction_type': args.abstraction_type,
        }
        if args.abstraction_type == 'temp-ensemble': policy_kwargs['decay'] = config.decay
        policy_name = 'MLPPolicy'
        model = PPO(policy_name, envs,
                    learning_rate=config.lr,
                    n_steps=args.n_steps,
                    batch_size=args.batch_size,
                    gamma=args.gamma,
                    ent_coef=args.ent_coef,
                    clip_range=config.clip_range,
                    clip_range_vf=args.clip_range_vf,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=log_dir+f"{run.id}",
                    verbose=1,
                    device=args.device,
                    curriculum_coef=0 if args.abstraction_type == 'temp-ensemble' else config.curriculum_coef,
                    seed=args.seed)
        model.learn(total_timesteps=args.training_steps, log_interval=args.log_interval, callback=callback)
        run.finish()
    
    sweep_id = wandb.sweep(sweep_config, project=args.abstraction_type + 'mlp')
    wandb.agent(sweep_id, function=train, count=10)