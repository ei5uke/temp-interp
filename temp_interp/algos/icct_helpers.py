# reference: https://github.com/CORE-Robotics-Lab/ICCT/blob/main/icct/core/icct_helpers.py
# and: https://github.com/CORE-Robotics-Lab/Team-Development-with-Transparent-Policies/blob/4ed586005ddfefa6634f216e700cbefeda804ab0/ipm/models/idct_helpers.py#L267

import numpy as np
import sys
from temp_interp.algos.icct import ICCT
import torch
import copy
from collections import defaultdict

def deepen_icct(tree, rollout_data, device='cpu'):
    """
    Deepens the tree by computing the leaf that has the highest probability and output variance,
    in other words, a leaf that is picked often but is not confident in its answer.
    :param tree: the tree
    :param rollout_data: a batch of rollout data used to evaluate the tree
    :param device: device used for pytorch
    """
    # find the leaf to deepen
    deepened_leaf = -1
    # Method 1: find leaf w/ the highest prob and policy gradient variance
    dictionary = tree.forward_input_compressions(rollout_data.observations.to(torch.float32), deepening=True)
    probs = dictionary['probs']
    leaf_action = dictionary['leaf_action'].transpose(0, 1)
    action_diff = rollout_data.actions.unsqueeze(1).expand(-1, tree.num_leaves, -1) - leaf_action
    scaled_action_diff = action_diff * rollout_data.advantages.unsqueeze(1).expand(-1, tree.num_leaves).unsqueeze(-1)
    action_variance = 1 / scaled_action_diff.std(dim=-1)
    score = (probs * action_variance).mean(dim=0)
    deepened_leaf = torch.argmax(score).item()
    # Method 2: find leaf w/ max visitations
    # deepened_leaf = int(np.argmax(tree.visitations))

    old_leaf_info = copy.deepcopy(tree.leaf_init_information)
    old_weights = tree.layers  # Get the weights out
    old_comparators = tree.comparators  # get the comparator values out
    old_alphas = tree.alpha
    old_submodels = tree.lin_models

    leaf_information = old_leaf_info[deepened_leaf]  # get the old leaf init info out
    left_path = leaf_information[0]
    right_path = leaf_information[1]

    new_weight = np.random.normal(scale=0.2, size=old_weights[0].size()[0])
    new_comp = np.random.normal(scale=0.2, size=old_comparators[0].size()[0])
    new_alpha = np.random.normal(scale=1, size=1)
    new_submodel1 = old_submodels[deepened_leaf].cpu()
    new_submodel2 = old_submodels[deepened_leaf].cpu()

    new_weights = [weight.detach().clone().data.cpu().numpy() for weight in old_weights]
    new_weights.append(new_weight)  # Add it to the list of nodes
    new_comps = [comp.detach().clone().data.cpu().numpy() for comp in old_comparators]
    new_comps.append(new_comp)
    new_alphas = [alpha.detach().clone().data.cpu().numpy() for alpha in old_alphas]
    new_alphas.append(new_alpha)
    new_submodels = [old_submodels[i].cpu() for i in range(len(old_submodels))]
    new_submodels.append(new_submodel1)
    new_submodels.append(new_submodel2)
    # Remove the old leaf
    del new_submodels[deepened_leaf]
    # Add it to the list of nodes

    new_weights = np.array(new_weights)
    new_comps = np.array(new_comps)
    new_alphas = np.array(new_alphas)
    new_submodels = torch.nn.ModuleList(new_submodels)

    new_node_ind = len(new_weights) - 1  # Remember where we put it

    # Create the paths, which are copies of the old path but now with a left / right at the new node
    new_leaf1_left = left_path.copy()
    new_leaf1_right = right_path.copy()
    new_leaf2_left = left_path.copy()
    new_leaf2_right = right_path.copy()
    # Leaf 1 goes left at the new node, leaf 2 goes right
    new_leaf1_left.append(new_node_ind)
    new_leaf2_right.append(new_node_ind)

    new_leaf_information = old_leaf_info
    new_leaf_information.append([new_leaf1_left, new_leaf1_right])
    new_leaf_information.append([new_leaf2_left, new_leaf2_right])
    # Remove the old leaf
    del new_leaf_information[deepened_leaf]

    new_network = ICCT(input_dim=tree.input_dim, 
                    output_dim=tree.output_dim, 
                    weights=new_weights, 
                    comparators=new_comps,
                    leaves=new_leaf_information, 
                    alpha=new_alphas, 
                    paths=None,
                    submodels=new_submodels,
                    use_individual_alpha=tree.use_individual_alpha, 
                    device=device,
                    use_submodels=tree.use_submodels,
                    hard_node=tree.hard_node,
                    argmax_tau=tree.argmax_tau,
                    sparse_submodel_type=tree.sparse_submodel_type,
                    fs_submodel_version=tree.fs_submodel_version,
                    l1_hard_attn=tree.l1_hard_attn,
                    num_sub_features=tree.num_sub_features,
                    use_gumbel_softmax=tree.use_gumbel_softmax,
                    alg_type=tree.alg_type,)
                    # morph_debug_flag=[tree.left_path_sigs, tree.right_path_sigs, deepened_leaf, "deepening"]) # debugging
    return new_network

def prune_icct(tree, device='cpu'):
    """
    Prune the tree based on the leaf with zero visitations.
    :param tree: the tree
    :param device: device used for pytorch
    """
    leaf_info = copy.deepcopy(tree.leaf_init_information)

    # get the leaf/node to prune
    pruned_leaf = int(np.argmin(tree.visitations))
    _, leaf_2_node = node_leaf_map(leaf_info)
    decision_node_index = leaf_2_node[pruned_leaf]
    if decision_node_index in leaf_info[pruned_leaf][0]:
        prune_left = True
    else:
        prune_left = False
    nodes_to_prune_indices = [decision_node_index]

    # prune the leaves that have the decision node in their ancestors
    new_leaf_info_pruned = []
    for leaf_idx, leaf in enumerate(leaf_info):
        left = set(leaf[0])
        right = set(leaf[1])

        if decision_node_index != max(left | right):
            new_leaf_info_pruned.append(leaf)
        else:
            if prune_left:
                if decision_node_index not in left:
                    new_leaf_info_pruned.append(leaf)
            else:
                if decision_node_index not in right:
                    new_leaf_info_pruned.append(leaf)

    # adjust the indices so that they are correct after pruning
    n_decision_nodes, _ = tree.comparators.shape
    old_idx_to_new_idx = {idx: idx for idx in range(n_decision_nodes)}  # map old indices to new ones after pruning
    for idx in range(n_decision_nodes):
        for descendant in nodes_to_prune_indices:
            if idx > descendant:
                old_idx_to_new_idx[idx] -= 1

    for descendant in nodes_to_prune_indices:
        del old_idx_to_new_idx[descendant]

    # new leaf info with adjusted ancestors
    new_leaf_info_adjusted_ancestors = []
    for leaf in new_leaf_info_pruned:

        # populate left ancestors
        # we want to remove the decision node from the ancestors
        # and also adjust the indices
        left_ancestors = []
        for i in range(len(leaf[0])):
            old_node_idx = leaf[0][i]
            if old_node_idx == decision_node_index:
                continue  # don't add the decision node to the ancestors
            adjusted_node_idx = old_idx_to_new_idx[old_node_idx]
            left_ancestors.append(adjusted_node_idx)

        # populate right ancestors
        # we want to remove the decision node from the ancestors
        # and also adjust the indices
        right_ancestors = []
        for j in range(len(leaf[1])):
            old_node_idx = leaf[1][j]
            if old_node_idx == decision_node_index:
                continue  # don't add the decision node to the ancestors
            adjusted_node_idx = old_idx_to_new_idx[old_node_idx]
            right_ancestors.append(adjusted_node_idx)

        new_leaf_info_adjusted_ancestors.append([left_ancestors, right_ancestors])

    # we need to filter out the decision nodes that we are pruning
    # from the prior weights, comparators, and alphas
    old_weights = tree.layers
    old_comparators = tree.comparators
    old_alpha = tree.alpha
    old_submodels = tree.lin_models

    new_weights = [old_weights[i].detach().clone().data.cpu().numpy() \
                    for i in range(len(old_weights)) if i != decision_node_index]
    new_comps = [old_comparators[i].detach().clone().data.cpu().numpy() \
                        for i in range(len(old_comparators)) if i != decision_node_index]

    n_alphas = old_alpha.shape
    is_individual_alpha = not len(n_alphas) == 0 and n_alphas[0] > 1
    if is_individual_alpha:
        new_alpha = [old_alpha[i].detach().clone().data.cpu().numpy() \
                        for i in range(len(old_alpha)) if i != decision_node_index]
    else:
        new_alpha = [old_alpha.data.item()]
    new_submodels = [old_submodels[i].cpu() \
                       for i in range(len(old_submodels)) if i != pruned_leaf]

    new_weights = np.array(new_weights)
    new_comps = np.array(new_comps)
    new_alpha = np.array(new_alpha)
    new_submodels = torch.nn.ModuleList(new_submodels)

    new_network = ICCT(input_dim=tree.input_dim, 
                    output_dim=tree.output_dim, 
                    weights=new_weights, 
                    comparators=new_comps,
                    leaves=new_leaf_info_adjusted_ancestors, 
                    alpha=new_alpha, 
                    paths=None,
                    submodels=new_submodels,
                    use_individual_alpha=tree.use_individual_alpha, 
                    device=device,
                    use_submodels=tree.use_submodels,
                    hard_node=tree.hard_node,
                    argmax_tau=tree.argmax_tau,
                    sparse_submodel_type=tree.sparse_submodel_type,
                    fs_submodel_version=tree.fs_submodel_version,
                    l1_hard_attn=tree.l1_hard_attn,
                    num_sub_features=tree.num_sub_features,
                    use_gumbel_softmax=tree.use_gumbel_softmax,
                    alg_type=tree.alg_type,)
                    # morph_debug_flag=[tree.left_path_sigs, tree.right_path_sigs, pruned_leaf, "pruning"])
    return new_network

def node_leaf_map(leaves):
    node_2_leaf = defaultdict(list)
    leaf_2_node = {}
    for i, leaf in enumerate(leaves):
        left = copy.deepcopy(leaf[0])
        right = copy.deepcopy(leaf[1])
        left.extend(right)
        leaf_2_node[i] = max(left)
        node_2_leaf[max(left)].append(i)
    return node_2_leaf, leaf_2_node