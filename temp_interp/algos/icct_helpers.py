# reference: https://github.com/CORE-Robotics-Lab/ICCT/blob/main/icct/core/icct_helpers.py

import numpy as np
import sys
from temp_interp.algos.icct import ICCT
import torch
import copy

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
    leaf_index = -1
    # Method 1: find leaf w/ the highest prob and policy gradient variance
    dictionary = tree.forward_input_compressions(rollout_data.observations, deepening=True)
    probs = dictionary['probs']
    leaf_action = dictionary['leaf_action'].transpose(0, 1)
    action_diff = rollout_data.actions.unsqueeze(1).expand(-1, tree.num_leaves, -1) - leaf_action
    scaled_action_diff = action_diff * rollout_data.advantages.unsqueeze(1).expand(-1, tree.num_leaves).unsqueeze(-1)
    action_variance = scaled_action_diff.std(dim=-1)
    score = (probs * action_variance).mean(dim=0)
    leaf_idx = torch.argmax(score).item()
    # Method 2: find leaf w/ max visitations
    # leaf_index = int(np.argmax(tree.visitations))
    if leaf_index == -1: return

    old_leaf_info = copy.deepcopy(tree.leaf_init_information)
    old_weights = tree.layers  # Get the weights out
    old_comparators = tree.comparators  # get the comparator values out
    old_alphas = tree.alpha
    old_submodels = tree.lin_models

    leaf_information = old_leaf_info[leaf_index]  # get the old leaf init info out
    left_path = leaf_information[0]
    right_path = leaf_information[1]

    new_weight = np.random.normal(scale=0.2, size=old_weights[0].size()[0])
    new_comp = np.random.normal(scale=0.2, size=old_comparators[0].size()[0])
    new_alpha = np.random.normal(scale=1, size=1)
    new_submodel = torch.nn.Linear(tree.input_dim, tree.output_dim)

    new_leaf1 = np.random.normal(scale=0.2, size=tree.output_dim).tolist()
    new_leaf2 = np.random.normal(scale=0.2, size=tree.output_dim).tolist()

    new_weights = [weight.detach().clone().data.cpu().numpy() for weight in old_weights]
    new_weights.append(new_weight)  # Add it to the list of nodes
    new_comps = [comp.detach().clone().data.cpu().numpy() for comp in old_comparators]
    new_comps.append(new_comp)
    new_alphas = [alpha.detach().clone().data.cpu().numpy() for alpha in old_alphas]
    new_alphas.append(new_alpha)
    new_submodels = [old_submodels[i].cpu() for i in range(len(old_submodels))]
    new_submodels.append(new_submodel)
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
    new_leaf_information.append([new_leaf1_left, new_leaf1_right, new_leaf1])
    new_leaf_information.append([new_leaf2_left, new_leaf2_right, new_leaf2])
    # Remove the old leaf
    del new_leaf_information[leaf_index]

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
                    alg_type=tree.alg_type)
    return new_network

def prune_icct(tree, device='cpu'):
    """
    Prune the tree based on the leaf with zero visitations.
    :param tree: the tree
    :param device: device used for pytorch
    """
    leaf_info = tree.leaf_init_information
    leaves_with_idx = copy.deepcopy([(leaf_idx, leaf_info[leaf_idx]) for leaf_idx in range(len(leaf_info))])
    root = Node(find_root(leaves_with_idx), 0)
    find_children(root, leaves_with_idx, current_depth=1)

    # get the leaf/node to prune
    pruned_leaf = int(np.argmin(tree.visitations))
    node_2_leaf, leaf_2_node = node_leaf_map(len(leaf_info))
    decision_node_index = leaf_2_node[pruned_leaf]
    pruned_node_children = node_2_leaf[decision_node_index]
    if len(pruned_node_children) == 1:
        prune_left = True
    else:
        if pruned_node_children[0] == pruned_leaf:
            prune_left = True
        else:
            prune_left = False
    
    # run BFS to find the node that we would like to prune
    # which also contains pointers to all of its children
    node_to_prune = root
    q = [root]
    while len(q) > 0:
        node_to_prune = q.pop(0)
        # keep traversing until we find the node we want to prune
        if node_to_prune.idx == decision_node_index:
            break
        if node_to_prune.left_child is not None and node_to_prune.left_child.is_leaf is False:
            q.append(node_to_prune.left_child)
        if node_to_prune.right_child is not None and node_to_prune.right_child.is_leaf is False:
            q.append(node_to_prune.right_child)

    # populate the list of descendants of the node we want to prune
    nodes_to_prune_indices = [node_to_prune.idx]
    q = []
    if prune_left:
        if node_to_prune.left_child is not None and node_to_prune.left_child.is_leaf is False:
            q = [node_to_prune.left_child]
    else:
        if node_to_prune.right_child is not None and node_to_prune.right_child.is_leaf is False:
            q = [node_to_prune.right_child]

    while len(q) > 0:
        node = q.pop(0)
        nodes_to_prune_indices.append(node.idx)
        if node.left_child is not None and node.left_child.is_leaf is False:
            q.append(node.left_child)
        if node.right_child is not None and node.right_child.is_leaf is False:
            q.append(node.right_child)

    _, pruned_node_left_ancestors, pruned_node_right_ancestors = find_ancestors(root, decision_node_index)
    pruned_node_left_ancestors = [node.idx for node in pruned_node_left_ancestors]
    pruned_node_right_ancestors = [node.idx for node in pruned_node_right_ancestors]

    # prune the leaves that have the decision node in their ancestors
    new_leaf_info_pruned = []
    for leaf_idx, leaf in enumerate(leaf_info):
        left_ancestors = leaf_info[leaf_idx][0]
        right_ancestors = leaf_info[leaf_idx][1]
        if prune_left:
            if decision_node_index not in left_ancestors:
                new_leaf_info_pruned.append(leaf)
        else:
            if decision_node_index not in right_ancestors:
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

        new_leaf_info_adjusted_ancestors.append([left_ancestors, right_ancestors, leaf[2]])

    # we need to filter out the decision nodes that we are pruning
    # from the prior weights, comparators, and alphas
    old_weights = tree.layers
    old_comparators = tree.comparators
    old_alpha = tree.alpha
    old_submodels = tree.lin_models

    new_weights = [old_weights[i].detach().clone().data.cpu().numpy() \
                    for i in range(len(old_weights)) if i not in nodes_to_prune_indices]
    new_comps = [old_comparators[i].detach().clone().data.cpu().numpy() \
                        for i in range(len(old_comparators)) if i not in nodes_to_prune_indices]

    n_alphas = old_alpha.shape
    is_individual_alpha = not len(n_alphas) == 0 and n_alphas[0] > 1
    if is_individual_alpha:
        new_alpha = [old_alpha[i].detach().clone().data.cpu().numpy() \
                        for i in range(len(old_alpha)) if i not in nodes_to_prune_indices]
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
                    alg_type=tree.alg_type)
    return new_network

def node_leaf_map(num_leaves):
    stack = []
    i = 0
    depth = int(np.floor(np.log2(num_leaves)))
    while len(stack) < num_leaves - 1:
        level = []
        for j in range(2**i):
            level.append(2**i-1+j)
            if len(level) + len(stack) == num_leaves - 1: break
        stack = level + stack
        i += 1
    
    node_2_leaf = {}
    leaf_2_node = {}
    num_nodes_before_single = 2*int(num_leaves - 2**np.floor(np.log2(num_leaves)))
    popped_nodes = set()
    i = 0
    while i < num_leaves:
        node = stack.pop(0)
        # check if all of node's children have been popped. If they have, we can skip to next node.
        if node * 2 + 1 in popped_nodes and node * 2 + 2 in popped_nodes: continue
        popped_nodes.add(node)
        if i == num_nodes_before_single and num_leaves % 2 == 1:
            node_2_leaf[node] = [i]
            leaf_2_node[i] = node
            i += 1
        else:
            node_2_leaf[node] = [i, i+1]
            leaf_2_node[i] = node
            leaf_2_node[i+1] = node
            i += 2
    return node_2_leaf, leaf_2_node