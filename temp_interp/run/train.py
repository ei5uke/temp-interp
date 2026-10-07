# reference: https://github.com/CORE-Robotics-Lab/ICCT/blob/main/icct/runfiles/train.py
# modified to leverage PPO instead of SAC or TD3, directly apply PPO+ICCTs, and PPO+ICCT+action chunking.

import copy
import json
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
from temp_interp.rl_helpers.evaluation import evaluate_policy_A
from temp_interp.algos.cart import collect_teacher_data, fit_cart, cart_num_params, cart_to_icct
import temp_interp.envs.lunar_lander_hard

def make_env(env_id, gamma=None):
    def thunk():
        if env_id == 'figure8':
            create_env, _ = make_create_env(params=fig8_params, version=0)
            env = create_env()
        elif env_id == 'LunarLanderHard': 
            env = gym.make(env_id, continuous=True, enable_wind=True, wind_power=20.0, turbulence_power=2.0)
        else: env = gym.make(env_id)
        if gamma: env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.FlattenObservation(env) # change Dict to nparray
        return env
    return thunk

# Per-environment DDT and ITTR settings (Table 5 of the paper). The remaining ITTR keys are passed to PPO.
ITTR_CONFIGS = {
    'lane_keeping': dict(num_leaves=2, epsilon=5e-2, min_timesteps=200, min_headstart=2000, min_evaluations=2,
                         min_visitations=1e5, low_return_stat='mean', low_return_threshold=-0.3, prune_first=False),
    'cart': dict(num_leaves=2, epsilon=2e-1, min_timesteps=600, min_headstart=6000, min_evaluations=3,
                 min_visitations=1e5, low_return_stat='mean', low_return_threshold=0.0, prune_first=False),
    'lunar': dict(num_leaves=2, epsilon=5e-2, min_timesteps=1000, min_headstart=10000, min_evaluations=4,
                  min_visitations=1e5, low_return_stat='mean', low_return_threshold=0.0, prune_first=False),
    'lunar-hard': dict(num_leaves=4, epsilon=5e-3, min_timesteps=1000, min_headstart=10000, min_evaluations=4,
                       min_visitations=1e5, low_return_stat='frac_positive', low_return_threshold=0.5, prune_first=True),
}

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
    parser.add_argument('--num_search', help='the number of hyperparameter searches', type=int, default=1)
    parser.add_argument('--time_horizon', help='the time horizon of the temporal abstraction', type=int, default=10)
    parser.add_argument('--no_sweep', help='run once with the first value of each sweep parameter, without a W&B sweep (works offline)', action='store_true', default=False)
    parser.add_argument('--pred_coef', help='scaling coefficient lambda of the temporal prediction objective', type=float, default=1.0)
    # CART distillation and warm start
    parser.add_argument('--cart_teacher', help='path of a trained (MLP-Big) model to distill into a CART tree', type=str, default=None)
    parser.add_argument('--cart_samples', help='number of teacher state / action pairs used to fit CART', type=int, default=100000)
    parser.add_argument('--cart_max_leaves', help='maximum number of CART leaves (default: the starting number of DDT leaves)', type=int, default=None)
    parser.add_argument('--cart_eval_episodes', help='number of episodes used to evaluate the CART tree', type=int, default=100)
    parser.add_argument('--warm_start', help='after distillation, train the CART tree with RL as a DDT', action='store_true', default=False)
    # ITTR overrides of the per-environment presets in ITTR_CONFIGS
    parser.add_argument('--decay', help='EMA coefficient of temporal ensemble (default: the sweep value)', type=float, default=None)
    parser.add_argument('--epsilon', help='ITTR: minimum average policy complexity gradient', type=float, default=None)
    parser.add_argument('--min_timesteps', help='ITTR: minimum steps since the last restructuring (n)', type=int, default=None)
    parser.add_argument('--min_headstart', help='ITTR: minimum global steps before restructuring (m)', type=int, default=None)
    parser.add_argument('--min_visitations', help='ITTR: prune a leaf visited fewer than k times (k)', type=float, default=None)

    args = parser.parse_args()
    assert args.abstraction_type is not None, print("ERROR: Abstraction type not set.")
    assert not args.warm_start or args.cart_teacher is not None, "--warm_start requires --cart_teacher"
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

    ittr_kwargs = dict(ITTR_CONFIGS[args.env_name])
    num_leaves, epsilon = ittr_kwargs.pop('num_leaves'), ittr_kwargs.pop('epsilon')
    if args.epsilon is not None: epsilon = args.epsilon
    for key in ['min_timesteps', 'min_headstart', 'min_visitations']:
        if getattr(args, key) is not None: ittr_kwargs[key] = getattr(args, key)

    sweep_config = {
        'method': 'bayes',
        'parameters': {
            'ddt_lr': {
                'values': [5e-4]
            },
            'lr': {
                'values': [5e-4]
            },
            'clip_range': {
                'values': [0.2]
            },
            'time_horizon': {
                'values': [args.time_horizon]
            },
            'num_leaves': {
                'values': [num_leaves]
            },
            'epsilon': {
                'values': [epsilon]
            },
        }
    }
    if args.abstraction_type == 'temp-ensemble':
        sweep_config['metric'] = {'name': 'eval/mean_reward', 'goal': 'maximize'}
        # sweep_config['parameters']['decay'] = {'distribution': 'uniform', 'min': 0.5, 'max': 1.0}
        sweep_config['parameters']['decay'] = {'values': [0.75 if args.decay is None else args.decay]}
    elif args.abstraction_type == 'temp-pred':
        sweep_config['metric'] = {'name': 'eval/mean_total', 'goal': 'maximize'}
        # sweep_config['parameters']['curriculum_coef'] = {'distribution': 'uniform', 'min': 0.1, 'max': 1.0}
        sweep_config['parameters']['curriculum_coef'] = {'values': [0.5]}

    def train():
        ## wandb setup
        run_name = f"{env_id}__{args.seed}__ICCT_{args.time_horizon}_{int(time.time())}"
        if args.no_sweep:
            run = wandb.init(project=args.abstraction_type+"ICCT", name=run_name, sync_tensorboard=True,
                             config={k: v['values'][0] for k, v in sweep_config['parameters'].items()})
        else:
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

        if args.gpu and th.cuda.is_available():
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
            'abstraction_type': args.abstraction_type,
        }
        if args.abstraction_type == 'temp-ensemble': policy_kwargs['decay'] = config.decay
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
                    curriculum_coef=0 if args.abstraction_type == 'temp-ensemble' else config.curriculum_coef,
                    seed=args.seed,
                    epsilon=config.epsilon,
                    pred_coef=args.pred_coef,
                    ittr_kwargs=ittr_kwargs)

        if args.cart_teacher is not None:
            # CART: distill the teacher into a tree with linear leaves and evaluate it as an ICCT
            teacher = PPO.load(args.cart_teacher, env=envs, device=args.device)
            assert teacher.policy.abstraction_type == args.abstraction_type and teacher.policy.action_dim == model.policy.action_dim, \
                "The teacher must use the same abstraction type and time horizon"
            teacher_env = make_vec_env(make_env(env_id), n_envs=1, seed=args.seed)
            states, targets = collect_teacher_data(teacher, teacher_env, args.cart_samples)
            cart, leaf_models = fit_cart(states, targets, args.cart_max_leaves or num_leaves, args.seed)
            model.set_action_net(cart_to_icct(cart, leaf_models, model.policy.action_net))
            rewards, _, _ = evaluate_policy_A(model, monitor_eval_env, n_eval_episodes=args.cart_eval_episodes, return_episode_rewards=True)
            cart_results = {'cart/mean_reward': float(np.mean(rewards)), 'cart/std_reward': float(np.std(rewards)),
                            'cart/num_leaves': int(cart.get_n_leaves()), 'cart/num_params': int(cart_num_params(cart, model.policy.action_dim))}
            wandb.log(cart_results)
            with open(log_dir + f'cart_results_seed{args.seed}.json', 'w') as f:
                json.dump(cart_results, f)
            model.save(log_dir + f'cart_seed{args.seed}')
            if not args.warm_start:
                run.finish()
                return

        model.learn(total_timesteps=args.training_steps, log_interval=args.log_interval, callback=callback)
        run.finish()

    if args.no_sweep:
        train()
    else:
        sweep_id = wandb.sweep(sweep_config, project=args.abstraction_type+"ICCT")
        wandb.agent(sweep_id, function=train, count=args.num_search)