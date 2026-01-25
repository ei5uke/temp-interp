# Robustness verification on the tree

# 1. Run a DDT-pred tree on LL-hard a bunch of times.
# 2. Keep track of the number of positive-positives; false-positives; negative-negatives; and false-negatives
# 3. Have a way to trace the path taken in the tree

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
from temp_interp.algos.icct_helpers import convert_to_crisp
import temp_interp.envs.lunar_lander_hard

def make_env(env_id, gamma=None):
    def thunk():
        env = gym.make(env_id, continuous=True, enable_wind=True, wind_power=20.0, turbulence_power=2.0)
        if gamma: env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.FlattenObservation(env) # change Dict to nparray
        return env
    return thunk

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DDT-Pred Robustness Verification')
    parser.add_argument('--seed', help='random seed', type=int, default=13)
    parser.add_argument('--load_path', help='the path of saving the model', type=str, default='test')
    parser.add_argument('--num_episodes', help='number of episodes to test', type=int, default=10000)
    parser.add_argument('--load_file', help='which model file to load and test', type=str, default='best_model')

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
    args = parser.parse_args()
    
    set_random_seed(args.seed)
    env_id = 'LunarLanderHard'
    env = make_env(env_id, gamma=args.gamma)()
    closed_model_env = make_env(env_id, gamma=args.gamma)()

    if args.gpu:
        args.device = 'cuda'
    else:
        args.device = 'cpu'
        
    # debug
    # features_extractor = FlattenExtractor
    # args.fs_submodel_version = 0
    # ddt_kwargs = {
    #     'num_leaves': 8,
    #     'submodels': args.submodels,
    #     'hard_node': args.hard_node,
    #     'device': args.device,
    #     'argmax_tau': args.argmax_tau,
    #     'ddt_lr': 5e-4,
    #     'use_individual_alpha': args.use_individual_alpha,
    #     'sparse_submodel_type': args.sparse_submodel_type,
    #     'fs_submodel_version': args.fs_submodel_version,
    #     'l1_reg_coeff': args.l1_reg_coeff,
    #     'l1_reg_bias': args.l1_reg_bias,
    #     'l1_hard_attn': args.l1_hard_attn,
    #     'num_sub_features': args.num_sub_features,
    #     'use_gumbel_softmax': args.use_gumbel_softmax,
    #     'alg_type': 'ppo'
    # }
    # policy_kwargs = {
    #     'features_extractor_class': features_extractor,
    #     'ddt_kwargs': ddt_kwargs,
    #     'net_arch': {'vf': [64, 64]},
    #     'activation_fn': th.nn.Tanh,
    #     'time_horizon': 10,
    #     'abstraction_type': args.abstraction_type,
    # }
    # policy_name = 'ICCTPolicy'
    # model = PPO(policy_name, env,
    #             learning_rate=5e-4,
    #             n_steps=args.n_steps,
    #             batch_size=args.batch_size,
    #             gamma=args.gamma,
    #             ent_coef=args.ent_coef,
    #             clip_range=0.2,
    #             clip_range_vf=args.clip_range_vf,
    #             policy_kwargs=policy_kwargs,
    #             verbose=1,
    #             device=args.device,
    #             curriculum_coef=1,
    #             seed=args.seed,
    #             epsilon=5e-2)

    # load model
    model = PPO.load(args.load_path + "/" + args.load_file, env=env)

    model.policy.action_net.debug()

    # deployment loop
    obs = env.reset(seed=args.seed)[0]
    _ = closed_model_env.reset(seed=args.seed)[0] # has slight random shift
    render = False
    false_positives = 0 # the current method doesn't ccoutn for positive-falses and false-falses
    positive_positives = 0
    for i in range(args.num_episodes):
        done = False
        pred_crash = None
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            next_action = action[:2]
            pred_actions = action[2:]
            obs, reward, done, _, _ = env.step(next_action)
            if not pred_crash:
                closed_model_env.step(next_action)
                for j in range(len(pred_actions) // 2):
                    _, _, c_done, _, _ = closed_model_env.step(pred_actions[2*j:2*j+1])
                    if c_done:
                        pred_crash = j+1
            elif pred_crash > 0:
                pred_crash -= 1
            else:
                if done: positive_positives += 1
                else: false_positives += 1

            if render:
                env.render()
            if done:
                obs = env.reset(seed=args.seed+i)[0]
                _ = closed_model_env.reset(seed=args.seed+i)[0] # has slight random shift
                break
    print(f"FP: {false_positives}")
    print(f"PP: {positive_positives}")