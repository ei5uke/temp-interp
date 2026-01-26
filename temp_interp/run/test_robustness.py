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

def make_env(env_id):
    def thunk():
        env = gym.make(env_id, continuous=True, enable_wind=True, wind_power=20.0, turbulence_power=2.0)
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
    env = make_env(env_id)()

    # debug
    # if args.gpu:
    #     args.device = 'cuda'
    # else:
    #     args.device = 'cpu'
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
    # model.policy.action_net.debug()

    # deployment loop to check for confusion matrix
    true_positives = false_positives = false_negatives = true_negatives = 0
    A = B = 0
    no_guess = 0
    late_guess = 0
    done_timestep = []
    all_rewards = []
    for i in range(args.num_episodes):
        seed = args.seed + i
        done = False
        pred_finish = None
        pred_reward = None
        seen_obs = []
        taken_actions = []
        episode_reward = 0
        step = 0
        while not done or step < 1000:
            step += 1
            env = make_env(env_id)()
            obs = env.reset(seed=seed)[0]
            if len(seen_obs) == 0: seen_obs.append(obs)

            for j, taken_action in enumerate(taken_actions):
                obs, reward, done, trunc, _ = env.step(taken_action)

            action, _ = model.predict(obs, deterministic=True)
            next_action = action[:2]
            taken_actions.append(next_action)
            pred_actions = action[2:]

            obs, reward, done, trunc, _ = env.step(next_action)
            episode_reward += reward
            seen_obs.append(obs)

            if pred_finish is None:
                for k in range(len(pred_actions) // 2):
                    pred_obs, pred_reward, pred_done, pred_trunc, _ = env.step(pred_actions[2*k:2*k+2])
                    if pred_done:
                        done_timestep.append(k)
                        pred_finish = k
                        break
            elif pred_finish > 0:
                pred_finish -= 1
                if done: 
                    late_guess += 1
                    env.close()
                    all_rewards.append(episode_reward.item())
                    break
            elif pred_finish == 0:
                if done:
                    A += 1
                    if pred_reward == reward:
                        if pred_reward == 100:
                            true_positives += 1
                        elif pred_reward == -100:
                            true_negatives += 1
                    else:
                        if pred_reward == -100:
                            false_negatives += 1
                        elif pred_reward == 100:
                            false_positives += 1
                else:
                    B += 1
                    if pred_reward == 100: false_positives += 1
                    elif pred_reward == -100: false_negatives += 1
                all_rewards.append(episode_reward.item())
                env.close()
                break

            env.close()

            if done and pred_finish is None:
                no_guess += 1
                all_rewards.append(episode_reward.item())
                break
            elif done and pred_finish==0:
                late_guess += 1
                env.close()
                all_rewards.append(episode_reward.item())
                break
            elif done:
                import ipdb; ipdb.set_trace()

    print(f"TP: {true_positives}")
    print(f"FP: {false_positives}")
    print(f"FN: {false_negatives}")
    print(f"TN: {true_negatives}")
    print(f"NG: {no_guess}")
    print(f"LG: {late_guess}")
    print(f"Avg/S.e. timestep: {np.mean(done_timestep)}, {np.std(done_timestep) / len(done_timestep)}")
    print(np.mean(all_rewards), np.std(all_rewards) / len(all_rewards))