# reference: https://github.com/CORE-Robotics-Lab/ICCT/blob/main/icct/runfiles/train.py
# modified to leverage PPO instead of SAC or TD3, directly apply PPO+ICCTs, and PPO+ICCT+action chunking.

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
import temp_interp.envs.lunar_lander_hard
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CallbackList
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.torch_layers import CombinedExtractor, FlattenExtractor
from temp_interp.rl_helpers.save_after_ep_callback import EpCheckPointCallback
from temp_interp.rl_helpers.ppo import PPO
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
    parser.add_argument('--gpu', help='if run on a GPU', action='store_true', default=False)
    parser.add_argument('--n_steps', help='Number of steps per batch', type=int, default=2048)
    parser.add_argument('--lr', help='learning rate', type=float, default=3e-4)
    parser.add_argument('--batch_size', help='batch size', type=int, default=256)
    parser.add_argument('--gamma', help='the discount factor', type=float, default=0.9999)
    parser.add_argument('--clip-range', help='the clip coef of the gradient update', type=float, default=0.2)
    parser.add_argument('--clip-range-vf', help='the clip range of the value function, must be tuned depending on the env rewards', type=float, default=None)
    parser.add_argument('--ent-coef', help='the entropy coefficient in PPO', type=float, default=0.1)
    parser.add_argument('--training_steps', help='total steps for training the model', type=int, default=500000)
    # DDT kwargs
    parser.add_argument('--num_leaves', help='number of leaves used in ddt (2^n)', type=int, default=16)
    parser.add_argument('--submodels', help='if use sub-models in ddt', action='store_true', default=False)
    parser.add_argument('--sparse_submodel_type', help='the type of the sparse submodel, 1 for L1 regularization, 2 for feature selection, other values for not sparse', type=int, default=0)
    parser.add_argument('--hard_node', help='if use differentiable crispification', action='store_true', default=False)
    parser.add_argument('--argmax_tau', help='the temperature of the diff_argmax function', type=float, default=1.0)
    parser.add_argument('--ddt_lr', help='the learning rate of the ddt', type=float, default=3e-4)
    parser.add_argument('--use_individual_alpha', help='if use different alphas for different nodes', action='store_true', default=False)
    parser.add_argument('--l1_reg_coeff', help='the coefficient of the l1 regularization when using l1-reg submodels', type=float, default=5e-3)
    parser.add_argument('--l1_reg_bias', help='if consider biases in the l1 loss when using l1-reg submodels', action='store_true', default=False)
    parser.add_argument('--l1_hard_attn', help='if only sample one linear controller to perform L1 regularization for each update when using l1-reg submodels', action='store_true', default=False)
    parser.add_argument('--num_sub_features', help='the number of chosen features for submodels', type=int, default=1)
    parser.add_argument('--use_gumbel_softmax', help='if use gumble softmax instead of the differentiable argmax proposed in the paper', action='store_true', default=False)
    # evaluation and model saving
    parser.add_argument('--min_reward', help='minimum reward to save the model', type=int)
    parser.add_argument('--save_path', help='the path of saving the model', type=str, default='test')
    parser.add_argument('--n_eval_episodes', help='the number of episodes for each evaluation during training', type=int, default=5)
    parser.add_argument('--eval_freq', help='evaluation frequence of the model', type=int, default=1500)
    parser.add_argument('--log_interval', help='the number of episodes before logging', type=int, default=4)
    parser.add_argument('--use_wandb', help='whether to log using wandb instead of raw tensorboard', type=bool, default=True)

    args = parser.parse_args()
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
        'metric': {
            # ### general performance sweep
            # 'name': 'eval/mean_reward', # 'rollout/ep_rew_mean'
            # 'goal': 'maximize'

            # ### pred loss sweep
            # 'name': 'eval/mean_pred_error',
            # 'goal': 'minimize'

            ### total sweep
            'name': 'eval/mean_total',
            'goal': 'maximize'
        },
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
            'curriculum_coef': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 1.0,
            },
            'time_horizon': {
                'values': [10]
            },
            'num_leaves': {
                'values': [8, 16, 18, 20, 22]
            },
        }
    }

    def train():
        ## wandb setup
        run_name = f"{env_id}__{args.seed}__{int(time.time())}"
        run = wandb.init(name=run_name, sync_tensorboard=True)
        config = wandb.config

        log_dir = args.save_path
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        if not args.submodels and not args.hard_node:
            method = 'm1'
        elif args.submodels and not args.hard_node:
            method = 'm2'
            if args.sparse_submodel_type == 1 or args.sparse_submodel_type == 2:
                raise Exception('Not a method we want to test')
        elif not args.submodels and args.hard_node:
            method = 'm3'    
        else:
            if args.sparse_submodel_type != 1 and args.sparse_submodel_type != 2:
                method = 'm4'
            elif args.sparse_submodel_type == 1:
                method = 'm5a'
            else:
                method = f'm5b_{args.num_sub_features}'
        
        eval_monitor_file_path = log_dir + 'eval_' + method + f'_seed{args.seed}'
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
        
        ddt_kwargs = {
            'num_leaves': config.num_leaves,
            'submodels': args.submodels,
            'hard_node': args.hard_node,
            'device': args.device,
            'argmax_tau': args.argmax_tau,
            'ddt_lr': config.ddt_lr,
            'use_individual_alpha': args.use_individual_alpha,
            'sparse_submodel_type': args.sparse_submodel_type,
            'fs_submodel_version': args.fs_submodel_version,
            'l1_reg_coeff': args.l1_reg_coeff,
            'l1_reg_bias': args.l1_reg_bias,
            'l1_hard_attn': args.l1_hard_attn,
            'num_sub_features': args.num_sub_features,
            'use_gumbel_softmax': args.use_gumbel_softmax,
            'alg_type': 'ppo'
        }
        policy_kwargs = {
            'features_extractor_class': features_extractor,
            'ddt_kwargs': ddt_kwargs,
            'net_arch': {'vf': [64, 64]},
            'activation_fn': th.nn.Tanh,
            'time_horizon': config.time_horizon,
        }
        policy_name = 'ICCTPolicy'
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
                    seed=args.seed,
                    curriculum_coef=config.curriculum_coef,
                    time_horizon=config.time_horizon
                    )
        model.learn(total_timesteps=args.training_steps, log_interval=args.log_interval, callback=callback)
        run.finish()
    
    sweep_id = wandb.sweep(sweep_config, project="temp-pred")
    wandb.agent(sweep_id, function=train, count=5)