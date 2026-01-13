# reference: https://github.com/CORE-Robotics-Lab/ICCT/blob/main/icct/core/icct_helpers.py

import numpy as np
import sys
from temp_interp.algos.icct import ICCT
import torch
import copy
from collections import defaultdict

# TODO: unused
# def convert_to_crisp(fuzzy_model, training_data):
#     new_weights = []
#     new_comps = []
#     device = fuzzy_model.device

#     weights = np.abs(fuzzy_model.layers.cpu().detach().numpy())
#     most_used = np.argmax(weights, axis=1)
#     for comp_ind, comparator in enumerate(fuzzy_model.comparators):
#         comparator = comparator.item()
#         divisor = abs(fuzzy_model.layers[comp_ind][most_used[comp_ind]].item())
#         if divisor == 0:
#             divisor = 1
#         comparator /= divisor
#         new_comps.append([comparator])
#         max_ind = most_used[comp_ind]
#         new_weight = np.zeros(len(fuzzy_model.layers[comp_ind].data))
#         new_weight[max_ind] = fuzzy_model.layers[comp_ind][most_used[comp_ind]].item() / divisor
#         new_weights.append(new_weight)

#     new_input_dim = fuzzy_model.input_dim
#     new_weights = np.array(new_weights)
#     new_comps = np.array(new_comps)
#     new_alpha = fuzzy_model.alpha
#     new_alpha = 9999999. * new_alpha.cpu().detach().numpy() / np.abs(new_alpha.cpu().detach().numpy())
#     crispy_model = ICCT(input_dim=new_input_dim,
#                         output_dim=fuzzy_model.output_dim,
#                         weights=new_weights,
#                         comparators=new_comps,
#                         leaves=fuzzy_model.leaf_init_information,
#                         alpha=new_alpha,
#                         use_individual_alpha=fuzzy_model.use_individual_alpha,
#                         use_submodels=fuzzy_model.use_submodels,
#                         hard_node=False,
#                         sparse_submodel_type=fuzzy_model.sparse_submodel_type,
#                         l1_hard_attn=False,
#                         num_sub_features=fuzzy_model.num_sub_features,
#                         use_gumbel_softmax=fuzzy_model.use_gumbel_softmax,
#                         device=device).to(device)
#     if hasattr(fuzzy_model, 'action_mus'):
#         crispy_model.action_mus.data = fuzzy_model.action_mus.data
#     crispy_model.action_stds.data = fuzzy_model.action_stds.data
#     if fuzzy_model.use_submodels:
#         if fuzzy_model.sparse_submodel_type != 2:
#             crispy_model.lin_models = fuzzy_model.lin_models
#         else:
#             crispy_model.sub_scalars = fuzzy_model.sub_scalars
#             crispy_model.sub_weights = fuzzy_model.sub_weights
#             crispy_model.sub_biases = fuzzy_model.sub_biases

#     return crispy_model

# Below code was modified from https://github.com/CORE-Robotics-Lab/Team-Development-with-Transparent-Policies/blob/4ed586005ddfefa6634f216e700cbefeda804ab0/ipm/models/idct_helpers.py#L267

class Node:
    def __init__(self, idx: int, node_depth: int, is_leaf: bool=False,
                 left_child=None, right_child=None, domain_range=None):
        self.idx = idx
        self.node_depth = node_depth
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = is_leaf
        self.domain_range = domain_range

def find_ancestors(root, node_idx):
    q = [(root, [], [])]
    while q:
        node, curr_left_ancestors, curr_right_ancestors = q.pop(0)
        if node.idx == node_idx:
            return node, curr_left_ancestors, curr_right_ancestors
        if node.left_child and not node.left_child.is_leaf:
            q.append((node.left_child, curr_left_ancestors + [node], curr_right_ancestors))
        if node.right_child and not node.right_child.is_leaf:
            q.append((node.right_child, curr_left_ancestors, curr_right_ancestors + [node]))
    raise ValueError(f'Node with idx {node_idx} not found in tree')

def find_root(leaves):
    root_node = 0
    nodes_in_leaf_path = []
    for leaf in leaves:
        combined_ancestors = leaf[1][0] + leaf[1][1] # these are both lists, concat operation
        nodes_in_leaf_path.append(combined_ancestors)
    for node in nodes_in_leaf_path[0]:
        found_root = True
        for nodes in nodes_in_leaf_path:
            if node not in nodes:
                found_root = False
        if found_root:
            root_node = node
            break
    return root_node

def find_children(node, leaves, current_depth):
    # dfs
    left_subtree = [leaf for leaf in leaves if node.idx in leaf[1][0]]
    right_subtree = [leaf for leaf in leaves if node.idx in leaf[1][1]]

    for _, leaf in left_subtree:
        leaf[0].remove(node.idx)

    for _, leaf in right_subtree:
        leaf[1].remove(node.idx)

    left_child_is_leaf = len(left_subtree) == 1
    right_child_is_leaf = len(right_subtree) == 1

    if not left_child_is_leaf:
        left_child = find_root(left_subtree)
    else:
        left_child = left_subtree[0][0]
    if not right_child_is_leaf:
        right_child = find_root(right_subtree)
    else:
        right_child = right_subtree[0][0]

    left_child = Node(left_child, current_depth, left_child_is_leaf)
    right_child = Node(right_child, current_depth, right_child_is_leaf)
    node.left_child = left_child
    node.right_child = right_child

    if not left_child_is_leaf:
        find_children(left_child, left_subtree, current_depth + 1)
    if not right_child_is_leaf:
        find_children(right_child, right_subtree, current_depth + 1)

def compute_entropy(input: []):
    """
    Computes the entropy of a list of probabilities
    :param input: list of probabilities
    :return: entropy
    """
    return -np.sum([p * np.log(p) for p in input])

def logits_to_probs(logits): # TODO: this seems to be for discrete actions -> yes it's for overcooked
    """
    Converts logits to probabilities
    :param logits: list of logits
    :return: list of probabilities
    """
    return [np.exp(logit) / np.sum(np.exp(logits)) for logit in logits]

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
    action_variance = scaled_action_diff.std(dim=-1)
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
    # if len(set(new_leaf1_left) & set(new_leaf1_right)) > 0: import ipdb; ipdb.set_trace()
    # if len(set(new_leaf2_left) & set(new_leaf2_right)) > 0: import ipdb; ipdb.set_trace()
    # Remove the old leaf
    del new_leaf_information[deepened_leaf]

    # if len(new_leaf_information) != len(tree.lin_models) + 1:
    #     import ipdb; ipdb.set_trace()
    # for leaf in new_leaf_information:
    #     left = set(leaf[0])
    #     right = set(leaf[1])
    #     if len(left & right) > 0:
    #         import ipdb; ipdb.set_trace()

    new_network = ICCT(input_dim=tree.input_dim, 
                    output_dim=tree.output_dim, 
                    weights=new_weights, 
                    comparators=new_comps,
                    leaves=new_leaf_information, 
                    alpha=new_alphas, 
                    paths=None, # may not need
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
    # for leaf in tree.leaf_init_information:
    #     left = set(leaf[0])
    #     right = set(leaf[1])
    #     if len(left & right) > 0:
    #         import ipdb; ipdb.set_trace()

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
        # if len(left & right) > 0:
        #     import ipdb; ipdb.set_trace()

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
        # left = set(leaf[0])
        # right = set(leaf[1])
        # if len(left & right) > 0:
        #     import ipdb; ipdb.set_trace()

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

    # if len(new_leaf_info_adjusted_ancestors) != len(tree.lin_models) - 1:
    #     import ipdb; ipdb.set_trace()

    # for leaf in new_leaf_info_adjusted_ancestors:
    #     left = set(leaf[0])
    #     right = set(leaf[1])
    #     if len(left & right) > 0:
    #         import ipdb; ipdb.set_trace()

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
                    paths=None, # may not need
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