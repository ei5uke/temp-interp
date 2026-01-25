# references: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py
# and https://github.com/CORE-Robotics-Lab/ICCT/blob/a7887bfd824a86381599dc576b9e4c0aeac61092/icct/rl_helpers/sac.py
# modified PPO to include action chunking training.

import io
import pathlib
import warnings
import copy
from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from collections import deque, defaultdict
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, TensorDict
from stable_baselines3.common.utils import FloatSchedule, explained_variance, obs_as_tensor, check_for_correct_spaces
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.save_util import load_from_zip_file, recursive_getattr, recursive_setattr, save_to_zip_file
from stable_baselines3.common.vec_env.patch_gym import _convert_space

from temp_interp.rl_helpers.policies import BasePolicy, ICCTPolicy
from temp_interp.rl_helpers.buffers import TemporalRolloutBuffer
from temp_interp.algos.icct_helpers import deepen_icct, prune_icct

SelfPPO = TypeVar("SelfPPO", bound="PPO")

class PPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (ICCTPolicy, MLPPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`ppo_policies`
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "ICCTPolicy": ICCTPolicy,
    }

    def __init__(
        self,
        policy: Union[str, type[ICCTPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[TemporalRolloutBuffer]] = TemporalRolloutBuffer,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        method: str = 'pred',
        curriculum_coef: Union[float, Schedule] = 1.0,
        epsilon: float = 0.05, # the minimum average gradient that must be observed to incentivize morph.
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `TemporalRolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self.method = method
        self.curriculum_coef = curriculum_coef

        if _init_setup_model:
            self._setup_model()
            # project actions, states, and leaf probabilities to the same dimension (the average of the dimension spaces)
            self.dim_common = (self.policy.flattened_obs_dim + self.policy.action_dim + self.policy.action_net.num_leaves) // 3
            self.state_proj = th.nn.Linear(self.policy.flattened_obs_dim, self.dim_common).to(device)
            self.action_chunk_proj = th.nn.Linear(self.policy.action_dim, self.dim_common).to(device)
            self.action_proj = th.nn.Linear(self.policy.og_action_dim, self.dim_common).to(device)
            self.leaf_proj = th.nn.Linear(self.policy.action_net.num_leaves, self.dim_common).to(device)
            self.policy_complexities = []

            # only add if we are storing the flattened action to the buffer instead 
            # of the temporal ensemble action
            self.rollout_buffer.action_dim = self.policy.action_dim
            self.rollout_buffer.og_action_dim = self.policy.og_action_dim
            if self.policy.abstraction_type == 'temp-pred': self.rollout_buffer.log_prob_dim = self.policy.action_dim 

        # deepening / pruning variables
        self.last_morph_timestep = 0
        self.epsilon = epsilon

    def _setup_model(self) -> None:
        super()._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = FloatSchedule(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = FloatSchedule(self.clip_range_vf)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: TemporalRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Clear policy's past information, then collect rollouts
        """
        # we can either clear this before each rollout, or implicitly clear it when the tree gets replaced by the deepened/pruned one
        # self.policy.action_net.visitations = np.zeros(self.policy.action_net.num_leaves)

        if self.policy.abstraction_type == 'temp-ensemble': self.policy.clear_lists()

        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)  # type: ignore[arg-type]
                if self.policy.abstraction_type == 'temp-ensemble':
                    actions, values, log_probs, flattened_actions = self.policy(obs_tensor)
                elif self.policy.abstraction_type == 'temp-pred':
                    actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    if self.policy.abstraction_type == 'temp-ensemble':
                        clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
                    elif self.policy.abstraction_type == 'temp-pred':
                        clipped_actions = np.clip(actions, np.repeat(self.action_space.low, self.policy.action_dim // 
                            self.policy.og_action_dim), np.repeat(self.action_space.high, self.policy.action_dim // 
                            self.policy.og_action_dim))[:, :self.policy.og_action_dim]

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            # print("New Obs: ", new_obs)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                flattened_actions.cpu() if self.policy.abstraction_type == 'temp-ensemble' else actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs.reshape(self.env.num_envs, -1),
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]
        # Update curriculum coefficient
        # lamb = self.curriculum_coef(self._current_progress_remaining)
        # curr_level = 1 + int((1 - self._current_progress_remaining) * self.policy.time_horizon) # 1->0

        entropy_losses = []
        pg_losses, value_losses = [], []
        if self.policy.abstraction_type == 'temp-pred': pred_losses = []
        ratios = []
        clip_fractions = []

        # deepen / prune the tree
        method = self._check_morph()
        if method is not None:
            self._morph(method)

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            # approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):

                actions = rollout_data.actions

                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()

                # Policy loss
                losses = self._policy_loss_helper(self.policy.action_net, rollout_data, log_prob, clip_range)
                policy_loss = losses['policy_loss']
                ratio = losses['ratio']
                if self.policy.abstraction_type == 'temp-pred' and self.policy.time_horizon > 1: temp_pred_loss = losses['temp_pred_loss']

                # Logging
                pg_losses.append(policy_loss.item())
                ratios.append(ratio.mean().item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)
                if self.policy.abstraction_type == 'temp-pred' and self.policy.time_horizon > 1: 
                    pred_losses.append(temp_pred_loss.item())

                # Value loss using the TD(gae_lambda) target
                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                if self.policy.abstraction_type == 'temp-pred' and self.policy.time_horizon > 1: loss += temp_pred_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                # with th.no_grad():
                #     log_ratio = log_prob - rollout_data.old_log_prob
                #     approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                #     approx_kl_divs.append(approx_kl_div)
                # if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                #     continue_training = False
                #     if self.verbose >= 1:
                #         print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                #     break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        # explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        ### mutual information (mi) analysis
        mi_batch = next(self.rollout_buffer.get(self.batch_size))
        obs_batch = mi_batch.observations
        action_batch = mi_batch.actions
        with th.no_grad():
            # Estimate policy complexity, for curr action and action chunk
            s_proj = F.normalize(self.state_proj(obs_batch.to(th.float32)), dim=1)
            a_proj = F.normalize(self.action_proj(action_batch[:, :self.policy.og_action_dim]), dim=1)
            ac_proj = F.normalize(self.action_chunk_proj(action_batch), dim=1)
            policy_complexity = self._estimate_mutual_info(s_proj, a_proj)
            self.policy_complexities.append(policy_complexity)
            self.logger.record(f"train/I(S;A)", policy_complexity)
            self.logger.record(f"train/I(S;AC)", self._estimate_mutual_info(s_proj, ac_proj))

            # Information bottleneck
            leaf_probs = self.policy.forward_info_bottleneck(obs_batch)
            t_proj = F.normalize(self.leaf_proj(leaf_probs), dim=1)
            self.logger.record(f"train/I(S;T)", self._estimate_mutual_info(s_proj, t_proj))
            self.logger.record(f"train/I(T;A)", self._estimate_mutual_info(t_proj, a_proj))
            self.logger.record(f"train/I(T;AC)", self._estimate_mutual_info(t_proj, ac_proj))

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        if self.policy.abstraction_type == 'temp-pred': self.logger.record("train/pred_loss", np.mean(pred_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        # self.logger.record("train/ratios", np.mean(ratios)) # debugging
        # self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        # self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        # self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/num_leaves", int(self.policy.action_net.num_leaves))

        # self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        # self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def _policy_loss_helper(self, action_net, rollout_data, log_prob, clip_range):
        """
        Calculates policy gradient loss, modularized code.
        """
        
        # Normalize advantage
        advantages = rollout_data.advantages
        # Normalization does not make sense if mini batchsize == 1, see GH issue #325
        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ratio between old and new policy, should be one at the first iteration
        if self.policy.abstraction_type == 'temp-ensemble':
            ratio = th.exp(log_prob - rollout_data.old_log_prob)
        elif self.policy.abstraction_type == 'temp-pred':
            if self.policy.time_horizon > 1: old_log_prob = rollout_data.old_log_prob[:, 0, :].sum(dim=1)
            else: old_log_prob = rollout_data.old_log_prob.reshape_as(log_prob).sum(dim=1)
            ratio = th.exp(log_prob[:, :self.policy.og_action_dim].sum(dim=1) - old_log_prob)

        # clipped surrogate loss
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

        # DDT addition
        if type(self.policy) is ICCTPolicy:
            if action_net.use_submodels and action_net.sparse_submodel_type == 1:
                attn = action_net.leaf_attn.repeat_interleave(2)
                l1_reg_loss = 0
                if action_net_kwargs['l1_reg_bias']:
                    for i, (name, p) in enumerate(action_net.lin_models.named_parameters()):
                        l1_reg_loss += th.sum(abs(p)) * attn[i]
                else:
                    for i, (name, p) in enumerate(action_net.lin_models.named_parameters()):
                        if not 'bias' in name:
                            l1_reg_loss += th.sum(abs(p)) * attn[i]
                l1_reg_loss *= self.policy.ddt_kwargs['l1_reg_coeff'] * self.policy.ddt.leaf_attn.size(0)
                l1_reg_losses.append(l1_reg_loss.item())
                policy_loss += l1_reg_loss

        losses = {'policy_loss': policy_loss, 'ratio': ratio}

        # temporal loss only if abstraction is temp-pred
        if self.policy.abstraction_type == 'temp-pred' and self.policy.time_horizon > 1:
            # temporal prediction loss 1 (log prob)
            curr_action = rollout_data.actions.clone()
            curr_action = curr_action[:, :self.policy.og_action_dim].repeat(1, self.policy.time_horizon)
            _, pred_log_prob, _ = self.policy.evaluate_actions(rollout_data.past_observations, 
                curr_action.repeat_interleave(self.policy.time_horizon - 1, dim=0))
            pred_log_prob = pred_log_prob.reshape(pred_log_prob.shape[0], self.policy.time_horizon, self.policy.og_action_dim).sum(dim=-1)
            idcs = th.tensor(th.arange(self.policy.time_horizon-1, 0, step=-1).reshape(-1, 1).tolist()*self.batch_size).to(pred_log_prob.device)
            pred_log_prob = th.gather(pred_log_prob, 1, idcs).reshape(-1, self.policy.time_horizon - 1)
            # curr_level is a curriculum parameter. Increase horizon length for prediction later on in training.
            temp_pred_loss = (1 - self._current_progress_remaining) * -pred_log_prob.mean()
            losses['temp_pred_loss'] = temp_pred_loss
        return losses

    def _estimate_mutual_info(self, proj_A, proj_B, temp=0.1):
        """
        Estimate Mutual information using InfoNCELoss.
        
        :param proj_A: the projection of the first vector to a common space.
        :param proj_B: the projection of the second vector to a common space.
        """
        labels = th.arange(self.batch_size).to(proj_A.device)  # positive pairs on diagonal
        similarity = th.matmul(proj_A, proj_B.T) / temp
        loss_A_B = F.cross_entropy(similarity, labels)
        loss_B_A = F.cross_entropy(similarity.T, labels)
        loss = (loss_A_B + loss_B_A) / 2
        return np.log(self.batch_size) - loss.cpu().item()

    def _check_morph(self, min_headstart=10000, min_timesteps=1000, min_evaluations=4):
    # def _check_morph(self, min_headstart=1000, min_timesteps=500, min_evaluations=1): # debugging
        """
        Check whether we should morph (deepen or prune) the tree. Return whether to deepen or prune if enough timesteps 
        have passed and if the policy complexities have, on-average, been increasing / decreasing by more than epsilon.

        :param timesteps: the current number of timesteps completed in PPO since the last morph step.
        :param min_headstart: the minimum number of timesteps before we can even allow morph.
        :param min_timesteps: the minimum number of timesteps between each morph session.
        :param min_evaluations: the minimum number of policy complexity evaluations performed before morphing
        """
        print(f"visitations: {self.policy.action_net.visitations}", flush=True)
        if self.num_timesteps > min_headstart and (self.num_timesteps - min_timesteps) > self.last_morph_timestep and len(self.policy_complexities) > min_evaluations:
            trend = np.mean(np.gradient(self.policy_complexities))
            print(f"trend: {trend}", flush=True)
            if trend - self.epsilon > 0:
                self.policy_complexities = []
                return 'deepen'
            # elif trend + self.epsilon < 0 and self.policy.action_net.num_leaves > 2 and np.min(self.policy.action_net.visitations) == 0:
            elif self.policy.action_net.num_leaves > 2 and np.min(self.policy.action_net.visitations) == 0: # prune anytime we have unvisited leaves and we don't deepen.
                self.policy_complexities = []
                return 'prune'

            # # debugging
            # self.policy_complexities = []
            # return 'deepen'
        return None

    def _morph(self, method):
        """
        Deepen or prune the tree.

        :param method: whether to deepen or prune.
        """

        if method == 'deepen':
            print("***Deepening***", flush=True)
            rollout_data = next(self.rollout_buffer.get(self.batch_size)) # get a random batch to test
            deepened_tree = deepen_icct(self.policy.action_net, rollout_data, self.device).to(self.device)
            assert deepened_tree is not None
            self.policy.new_action_net = deepened_tree
        elif method == 'prune':
            print("***Pruning***", flush=True)
            pruned_tree = prune_icct(self.policy.action_net, self.device).to(self.device)
            self.policy.new_action_net = pruned_tree
        else:
            print("Error in _morph", flush=True)
            exit()
        self.policy.action_net = self.policy.new_action_net
        self.policy.new_action_net = None
        self.last_morph_timestep = self.num_timesteps
        self.leaf_proj = th.nn.Linear(self.policy.action_net.num_leaves, self.dim_common).to(self.device)
        curr_optimizer_lr = self.policy.optimizer.param_groups[0]['lr']
        self.policy.optimizer = type(self.policy.optimizer)(self.policy.parameters(), **self.policy.optimizer.defaults)
        self.policy.optimizer.param_groups[0]['lr'] = curr_optimizer_lr
        print("***Completed.***", flush=True)

    def learn(
        self: SelfPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfPPO:
        # total_params = sum(param.numel() for param in self.policy.action_net.parameters() if param.requires_grad)
        # print(f"Total trainable parameters: {total_params}")
        # exit()
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def set_parameters(
        self,
        load_path_or_dict: Union[str, TensorDict],
        exact_match: bool = True,
        device: Union[th.device, str] = "auto",
    ) -> None:
        """
        Load parameters from a given zip-file or a nested dictionary containing parameters for
        different modules (see ``get_parameters``).

        :param load_path_or_iter: Location of the saved data (path or file-like, see ``save``), or a nested
            dictionary containing nn.Module parameters used by the policy. The dictionary maps
            object names to a state-dictionary returned by ``torch.nn.Module.state_dict()``.
        :param exact_match: If True, the given parameters should include parameters for each
            module and each of their parameters, otherwise raises an Exception. If set to False, this
            can be used to update only specific parameters.
        :param device: Device on which the code should run.
        """
        params = {}
        if isinstance(load_path_or_dict, dict):
            params = load_path_or_dict
        else:
            _, params, _ = load_from_zip_file(load_path_or_dict, device=device, load_data=False)

        # Keep track which objects were updated.
        # `_get_torch_save_params` returns [params, other_pytorch_variables].
        # We are only interested in former here.
        objects_needing_update = set(self._get_torch_save_params()[0])
        updated_objects = set()

        for name in params:
            attr = None
            try:
                attr = recursive_getattr(self, name)
            except Exception as e:
                # What errors recursive_getattr could throw? KeyError, but
                # possible something else too (e.g. if key is an int?).
                # Catch anything for now.
                raise ValueError(f"Key {name} is an invalid object name.") from e

            if isinstance(attr, th.optim.Optimizer):
                # # Optimizers do not support "strict" keyword...
                # # Seems like they will just replace the whole
                # # optimizer state with the given one.
                # # On top of this, optimizer state-dict
                # # seems to change (e.g. first ``optim.step()``),
                # # which makes comparing state dictionary keys
                # # invalid (there is also a nesting of dictionaries
                # # with lists with dictionaries with ...), adding to the
                # # mess.
                # #
                # # TL;DR: We might not be able to reliably say
                # # if given state-dict is missing keys.
                # #
                # # Solution: Just load the state-dict as is, and trust
                # # the user has provided a sensible state dictionary.
                # attr.load_state_dict(params[name])  # type: ignore[arg-type]
                pass
            else:
                # Assume attr is th.nn.Module
                attr.load_state_dict(params[name], strict=exact_match)
                updated_objects.add(name)
    
    @classmethod
    def load(  # noqa: C901
        cls: type[SelfPPO],
        path: str,
        env: GymEnv,
        device: str = "auto",
        custom_objects = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs,
    ) -> SelfPPO:
        """
        Load the model from a zip-file.
        Warning: ``load`` re-creates the model from scratch, it does not update it in-place!
        For an in-place load use ``set_parameters`` instead.

        :param path: path to the file (or a file-like) where to
            load the agent from
        :param env: the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param device: Device on which the code should run.
        :param custom_objects: Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            ``keras.models.load_model``. Useful when you have an object in
            file that can not be deserialized.
        :param print_system_info: Whether to print system info from the saved model
            and the current system info (useful to debug loading issues)
        :param force_reset: Force call to ``reset()`` before training
            to avoid unexpected behavior.
            See https://github.com/DLR-RM/stable-baselines3/issues/597
        :param kwargs: extra arguments to change the model when loading
        :return: new model instance with loaded parameters
        """
        data, params, pytorch_variables = load_from_zip_file(
            path,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
        )

        assert data is not None, "No data found in the saved file"
        assert params is not None, "No params found in the saved file"

        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            data["policy_kwargs"]["ddt_kwargs"]["num_leaves"] = 3 # this should be whatever the final number of leaves is
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]
            # backward compatibility, convert to new format
            saved_net_arch = data["policy_kwargs"].get("net_arch")
            if saved_net_arch and isinstance(saved_net_arch, list) and isinstance(saved_net_arch[0], dict):
                data["policy_kwargs"]["net_arch"] = saved_net_arch[0]

        if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError("The observation_space and action_space were not given, can't verify new environments")

        # Gym -> Gymnasium space conversion
        for key in {"observation_space", "action_space"}:
            data[key] = _convert_space(data[key])

        if env is not None:
            # Wrap first if needed
            env = cls._wrap_env(env, data["verbose"])
            # Check if given env is valid
            check_for_correct_spaces(env, data["observation_space"], data["action_space"])
            # Discard `_last_obs`, this will force the env to reset before training
            # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
            if force_reset and data is not None:
                data["_last_obs"] = None
            # `n_envs` must be updated. See issue https://github.com/DLR-RM/stable-baselines3/issues/1018
            if data is not None:
                data["n_envs"] = env.num_envs
        else:
            # Use stored env, if one exists. If not, continue as is (can be used for predict)
            if "env" in data:
                env = data["env"]

        model = cls(
            policy=data["policy_class"],
            env=env,
            # policy_kwargs=data["policy_kwargs"],
            device=device,
            _init_setup_model=False,  # type: ignore[call-arg]
        )

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        try:
            # put state_dicts back in place
            model.set_parameters(params, exact_match=True, device=device)
        except RuntimeError as e:
            # Patch to load policies saved using SB3 < 1.7.0
            # the error is probably due to old policy being loaded
            # See https://github.com/DLR-RM/stable-baselines3/issues/1233
            if "pi_features_extractor" in str(e) and "Missing key(s) in state_dict" in str(e):
                model.set_parameters(params, exact_match=False, device=device)
                warnings.warn(
                    "You are probably loading a A2C/PPO model saved with SB3 < 1.7.0, "
                    "we deactivated exact_match so you can save the model "
                    "again to avoid issues in the future "
                    "(see https://github.com/DLR-RM/stable-baselines3/issues/1233 for more info). "
                    f"Original error: {e} \n"
                    "Note: the model should still work fine, this only a warning."
                )
            else:
                raise e
        except ValueError as e:
            # Patch to load DQN policies saved using SB3 < 2.4.0
            # The target network params are no longer in the optimizer
            # See https://github.com/DLR-RM/stable-baselines3/pull/1963
            saved_optim_params = params["policy.optimizer"]["param_groups"][0]["params"]  # type: ignore[index]
            n_params_saved = len(saved_optim_params)
            n_params = len(model.policy.optimizer.param_groups[0]["params"])
            if n_params_saved == 2 * n_params:
                # Truncate to include only online network params
                params["policy.optimizer"]["param_groups"][0]["params"] = saved_optim_params[:n_params]  # type: ignore[index]

                model.set_parameters(params, exact_match=True, device=device)
                warnings.warn(
                    "You are probably loading a DQN model saved with SB3 < 2.4.0, "
                    "we truncated the optimizer state so you can save the model "
                    "again to avoid issues in the future "
                    "(see https://github.com/DLR-RM/stable-baselines3/pull/1963 for more info). "
                    f"Original error: {e} \n"
                    "Note: the model should still work fine, this only a warning."
                )
            else:
                raise e

        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                # Skip if PyTorch variable was not defined (to ensure backward compatibility).
                # This happens when using SAC/TQC.
                # SAC has an entropy coefficient which can be fixed or optimized.
                # If it is optimized, an additional PyTorch variable `log_ent_coef` is defined,
                # otherwise it is initialized to `None`.
                if pytorch_variables[name] is None:
                    continue
                # Set the data attribute directly to avoid issue when using optimizers
                # See https://github.com/DLR-RM/stable-baselines3/issues/391
                recursive_setattr(model, f"{name}.data", pytorch_variables[name].data)

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if model.use_sde:
            model.policy.reset_noise()  # type: ignore[operator]
        return model