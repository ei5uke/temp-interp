# CART distillation of a (chunked) teacher policy into a decision tree with linear models at its leaves,
# and conversion of that tree into an ICCT so it can be evaluated as is (CART) or fine-tuned with RL (Warm).

import numpy as np
import torch as th
from torch import nn
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from stable_baselines3.common.vec_env import VecEnv

from temp_interp.algos.icct import ICCT

# ICCT leaves output tanh(linear(x)), so the leaf models are fit on atanh of the clipped teacher actions
TANH_CLIP = 0.999


def collect_teacher_data(teacher, env: VecEnv, n_samples: int):
    """
    Roll out the teacher deterministically and record the states it visits with its mean action (chunk).

    :param teacher: trained PPO model whose policy is a ``TemporalActorCriticPolicy``
    :param env: environment to roll out in
    :param n_samples: number of state / action-chunk pairs to collect
    :return: states [n_samples, obs_dim] and mean action chunks [n_samples, action_dim]
    """
    policy = teacher.policy
    if policy.abstraction_type == 'temp-ensemble':
        policy.past_eval_actions.clear()
        policy.past_eval_valid.clear()
    device = next(policy.parameters()).device
    states, targets = [], []
    obs = env.reset()
    while len(states) * env.num_envs < n_samples:
        obs_tensor = th.as_tensor(obs, dtype=th.float32, device=device)
        with th.no_grad():
            targets.append(policy.get_distribution(obs_tensor).distribution.mean.cpu().numpy())
            actions = policy(obs_tensor, deterministic=True).cpu().numpy()
        states.append(np.asarray(obs, dtype=np.float32).reshape(env.num_envs, -1))
        obs, _, dones, _ = env.step(actions[:, :policy.og_action_dim])
        if policy.abstraction_type == 'temp-ensemble': policy.reset_ensemble(dones, deterministic=True)
    return np.concatenate(states)[:n_samples], np.concatenate(targets)[:n_samples]


def fit_cart(states: np.ndarray, targets: np.ndarray, max_leaf_nodes: int, seed: int):
    """
    Fit a CART regression tree with at most ``max_leaf_nodes`` leaves (grown best-first),
    then a linear model on each leaf, both on atanh of the clipped targets.

    :return: the fitted sklearn tree and one ``LinearRegression`` per sklearn leaf id
    """
    link_targets = np.arctanh(np.clip(targets, -TANH_CLIP, TANH_CLIP))
    cart = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, min_samples_leaf=2 * (states.shape[1] + 1),
                                 random_state=seed).fit(states, link_targets)
    leaf_ids = cart.apply(states)
    leaf_models = {leaf: LinearRegression().fit(states[leaf_ids == leaf], link_targets[leaf_ids == leaf])
                   for leaf in np.unique(leaf_ids)}
    return cart, leaf_models


def cart_num_params(cart: DecisionTreeRegressor, output_dim: int) -> int:
    """
    Number of parameters of the distilled tree: a feature index and a threshold per decision node,
    and a linear model (weights and bias) per leaf.
    """
    n_leaves = cart.get_n_leaves()
    return 2 * (n_leaves - 1) + n_leaves * (cart.n_features_in_ + 1) * output_dim


def cart_to_icct(cart: DecisionTreeRegressor, leaf_models: dict, template: ICCT) -> ICCT:
    """
    Build an ICCT computing the same function as the CART tree with linear leaves.
    The ICCT settings (crisp nodes, submodels, ...) are copied from ``template``.

    A CART node goes left when x_f <= threshold. The ICCT node with one-hot weight e_f and comparator
    ``threshold`` evaluates TRUE (its left branch) when x_f > threshold, so CART's right child becomes
    the ICCT's left branch. Nodes keep CART's pre-order, so every node index is larger than its ancestors',
    as ``prune_icct`` expects.
    """
    assert template.use_submodels and template.sparse_submodel_type != 2, "Only ICCT-complete trees are supported"
    tree = cart.tree_
    assert cart.get_n_leaves() >= 2, "CART found no split, an ICCT needs at least two leaves"
    decision_nodes = [n for n in range(tree.node_count) if tree.children_left[n] != -1]
    node_index = {n: i for i, n in enumerate(decision_nodes)}

    weights, comparators = [], []
    for n in decision_nodes:
        weight = np.zeros(template.input_dim)
        weight[tree.feature[n]] = 1.0
        weights.append(weight)
        comparators.append([tree.threshold[n]])

    # leaf_info[i] = [nodes where the path to leaf i goes left (TRUE), nodes where it goes right]
    leaf_info, submodels = [], []
    stack = [(0, [], [])]
    while stack:
        n, left_path, right_path = stack.pop()
        if tree.children_left[n] == -1:
            linear = nn.Linear(template.input_dim, template.output_dim)
            with th.no_grad():
                linear.weight.copy_(th.as_tensor(leaf_models[n].coef_.reshape(template.output_dim, -1)))
                linear.bias.copy_(th.as_tensor(np.atleast_1d(leaf_models[n].intercept_)))
            leaf_info.append([sorted(left_path), sorted(right_path)])
            submodels.append(linear)
            continue
        i = node_index[n]
        stack.append((tree.children_left[n], left_path, right_path + [i]))
        stack.append((tree.children_right[n], left_path + [i], right_path))

    n_nodes = len(decision_nodes)
    alpha = [[1.0]] * n_nodes if template.use_individual_alpha else [1.0]
    return ICCT(input_dim=template.input_dim,
                output_dim=template.output_dim,
                weights=np.array(weights),
                comparators=np.array(comparators),
                leaves=leaf_info,
                alpha=alpha,
                paths=None,
                submodels=nn.ModuleList(submodels),
                use_individual_alpha=template.use_individual_alpha,
                device=template.device,
                use_submodels=template.use_submodels,
                hard_node=template.hard_node,
                argmax_tau=template.argmax_tau,
                sparse_submodel_type=template.sparse_submodel_type,
                fs_submodel_version=template.fs_submodel_version,
                l1_hard_attn=template.l1_hard_attn,
                num_sub_features=template.num_sub_features,
                use_gumbel_softmax=template.use_gumbel_softmax,
                alg_type=template.alg_type)
