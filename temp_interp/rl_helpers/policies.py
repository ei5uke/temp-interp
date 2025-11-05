# reference: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/policies.py
# modified to include new ICCTPolicy definition

from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, MultiInputActorCriticPolicy
MlpPolicy = ActorCriticPolicy
CnnPolicy = ActorCriticCnnPolicy
MultiInputPolicy = MultiInputActorCriticPolicy

import numpy as np
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from collections import deque
from gymnasium import spaces
import torch as th
from torch import nn
from stable_baselines3.common.distributions import Distribution, DiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    MlpExtractor,
)
from temp_interp.algos.icct import ICCT
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule

class ICCTPolicy(BasePolicy):
    """
    Actor-critic for PPO with ICCT.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        ddt_kwargs: Dict[str, Any] = None,
        time_horizon: int = 3
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
            normalize_images=normalize_images,
        )
        
        self.observation_space = observation_space
        self.flattened_obs_dim = get_flattened_obs_dim(self.observation_space)
        new_shape = (action_space.shape[0] * time_horizon,)
        new_low = np.tile(action_space.low, time_horizon)
        new_high = np.tile(action_space.high, time_horizon)

        self.action_space = spaces.Box(low=new_low, high=new_high, shape=new_shape, dtype=action_space.dtype)
        # self.action_space = action_space
        self.action_dim = get_action_dim(self.action_space)
        self.ddt_kwargs = ddt_kwargs
        self.rpo_alpha = 0.5 # Robust Policy Optimization addition
        self.time_horizon = time_horizon
        self.past_actions = deque(maxlen=self.time_horizon)
        self.past_log_probs = deque(maxlen=self.time_horizon)

        if isinstance(net_arch, list) and len(net_arch) > 0 and isinstance(net_arch[0], dict):
            warnings.warn(
                (
                    "As shared layers in the mlp_extractor are removed since SB3 v1.8.0, "
                    "you should now pass directly a dictionary and not a list "
                    "(net_arch=dict(pi=..., vf=...) instead of net_arch=[dict(pi=..., vf=...)])"
                ),
            )
            net_arch = net_arch[0]

        # Default network architecture, from stable-baselines
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = dict(pi=[64, 64], vf=[64, 64])

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        self.share_features_extractor = share_features_extractor
        self.features_extractor = self.make_features_extractor()
        self.features_dim = self.features_extractor.features_dim
        if self.share_features_extractor:
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = self.features_extractor
        else:
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = self.make_features_extractor()

        self.log_std_init = log_std_init
        dist_kwargs = None

        assert not (squash_output and not use_sde), "squash_output=True is only available when using gSDE (use_sde=True)"
        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": False,
            }

        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        # Action distribution
        # self.action_dist = make_proba_distribution(action_space, use_sde=use_sde, dist_kwargs=dist_kwargs)
        # self.action_chunk_dist = SquashedDiagGaussianDistribution(self.action_dim)
        # self.action_dist = SquashedDiagGaussianDistribution(self.action_dim // self.time_horizon)
        self.action_chunk_dist = DiagGaussianDistribution(self.action_dim)

        self.og_action_dim = self.action_dim // self.time_horizon
        self._build(lr_schedule)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()

        default_none_kwargs = self.dist_kwargs or collections.defaultdict(lambda: None)  # type: ignore[arg-type, return-value]
        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                squash_output=default_none_kwargs["squash_output"],
                full_std=default_none_kwargs["full_std"],
                use_expln=default_none_kwargs["use_expln"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                ortho_init=self.ortho_init,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def reset_noise(self, n_envs: int = 1) -> None:
        """
        Sample new weights for the exploration matrix.

        :param n_envs:
        """
        assert isinstance(self.action_chunk_dist, StateDependentNoiseDistribution), "reset_noise() is only available when using gSDE"
        self.action_chunk_dist.sample_weights(self.log_std, batch_size=n_envs)

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()
        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        self.action_net = ICCT(input_dim=latent_dim_pi,
                        output_dim=self.action_dim,
                        weights=None,
                        comparators=None,
                        leaves=self.ddt_kwargs['num_leaves'],
                        alpha=None,
                        use_individual_alpha=self.ddt_kwargs['use_individual_alpha'],
                        device=self.ddt_kwargs['device'],
                        use_submodels=self.ddt_kwargs['submodels'],
                        hard_node=self.ddt_kwargs['hard_node'],
                        argmax_tau=self.ddt_kwargs['argmax_tau'],
                        sparse_submodel_type=self.ddt_kwargs['sparse_submodel_type'],
                        fs_submodel_version = self.ddt_kwargs['fs_submodel_version'],
                        l1_hard_attn=self.ddt_kwargs['l1_hard_attn'],
                        num_sub_features=self.ddt_kwargs['num_sub_features'],
                        use_gumbel_softmax=self.ddt_kwargs['use_gumbel_softmax'],
                        alg_type=self.ddt_kwargs['alg_type']).to(self.ddt_kwargs['device']).to(self.device)

        self.log_std = nn.Parameter(th.ones(self.action_dim) * self.log_std_init, requires_grad=True)

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=self.ddt_kwargs['ddt_lr'], **self.optimizer_kwargs) # this should probs be ddt_kwargs itself

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        # # get n_envs from obs
        # n_envs = obs.shape[0]
        # batch_size = n_envs * self.time_horizon

        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        actions = actions.reshape((-1, *self.action_space.shape))  # not sure if necessary
        # print("Actions for 2 envs: ", actions)

        chunked_action = actions.reshape(-1, self.time_horizon, self.og_action_dim) # to create chunk like [n_envs, chunk_size, action_dim]

        # print(chunked_action.transpose(0,1))
        # log_prob_dist = self._get_action_dist_from_latent(latent_pi, actions)
        # log_prob = log_prob_dist.log_prob(chunked_action.transpose(0,1)) # send it [chunk_size, n_envs, action_dim]
        # log_prob = log_prob.reshape(n_envs, self.time_horizon)

        action_batches = chunked_action.transpose(0, 1)

        # Get log probs for each action
        # log_prob = th.stack([
        #     distribution.distribution.log_prob(action_batches[i])  # Each is shape (n_env, action_dim) -> log_prob shape (n_env,)
        #     for i in range(self.time_horizon)
        # ])
        
        print("Actions: ", actions)
        log_prob = distribution.distribution.log_prob(actions)
        print("Log prob: ", log_prob)
        # print(log_prob)
        # Aggregate Actions
        self.past_actions.append(action_batches)
        self.past_log_probs.append(log_prob)
        # print("Past Actions: ", self.past_actions)
        # print("Past log_prob: ", self.past_log_probs)
        (ensemble_action, ensemble_log) = self.temporal_ensemble()
        
        # print("Ensemble action: ", ensemble_action)
        # print("Ensemble log: ", ensemble_log)
        return ensemble_action, values, ensemble_log

    def temporal_ensemble(self) -> tuple[th.Tensor, th.Tensor]:
        stacked_actions = th.stack(list(self.past_actions))
        n_tensors, time_steps, n_envs, action_dim = stacked_actions.shape
        time_indices = th.arange(n_tensors - 1, -1, -1)
        # print(time_indices)
        # print(stacked_actions.shape)
        # print("ensemble")
        # print(self.past_actions)
        selected_actions = stacked_actions[
            th.arange(n_tensors),  # tensor dimension
            time_indices,              # time_step dimension (reversed)
            :,                         # all environments
            :                          # all action dims
        ]
        stacked_logs = th.stack(list(self.past_log_probs))
        selected_logs = stacked_logs[
            th.arange(n_tensors),  # tensor dimension
            time_indices,              # time_step dimension (reversed)
            :                          # all environments
        ]
        print("Selected logs: ", selected_logs)
        # print()
        return (selected_actions.mean(dim=0), selected_logs.mean(dim=0))

    def forward_info_bottleneck(self, obs: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Return the leaf probabilities outputted by the DDT to use as a sufficient statistic for the information bottleneck

        :param obs: Observation
        :return: The batch of leaf probabilities.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, _ = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
        input_compressions = self.action_net.forward_input_compressions(latent_pi)
        return input_compressions

    def extract_features(  # type: ignore[override]
        self, obs: PyTorchObs, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> Union[th.Tensor, tuple[th.Tensor, th.Tensor]]:
        """
        Preprocess the observation if needed and extract features.

        :param obs: Observation
        :param features_extractor: The features extractor to use. If None, then ``self.features_extractor`` is used.
        :return: The extracted features. If features extractor is not shared, returns a tuple with the
            features for the actor and the features for the critic.
        """
        if self.share_features_extractor:
            return super().extract_features(obs, self.features_extractor if features_extractor is None else features_extractor)
        else:
            if features_extractor is not None:
                warnings.warn(
                    "Provided features_extractor will be ignored because the features extractor is not shared.",
                    UserWarning,
                )

            pi_features = super().extract_features(obs, self.pi_features_extractor)
            vf_features = super().extract_features(obs, self.vf_features_extractor)
            return pi_features, vf_features

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor, actions: th.Tensor = None) -> Distribution:
        """
        Retrieve action distribution given the latent codes.
        If actions is given, then evaluate at the action-level, not the action-chunk-level.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)
        if actions is not None:
            # new to RPO: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/rpo_continuous_action.py
            z = th.FloatTensor(mean_actions.shape).uniform_(-self.rpo_alpha, self.rpo_alpha).to(self.device)
            mean_actions = mean_actions + z
        return self.action_chunk_dist.proba_distribution(mean_actions, self.log_std)
    
    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        return self.get_distribution(observation).get_actions(deterministic=deterministic)

    def evaluate_actions(self, obs: PyTorchObs, actions: th.Tensor) -> tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """

        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi, actions)
        log_prob = distribution.distribution.log_prob(actions)[:,:self.action_dim].sum(dim=1)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        features = super().extract_features(obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)