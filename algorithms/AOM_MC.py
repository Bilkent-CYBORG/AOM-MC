import time

import math
import pickle
import numpy as np
from tqdm import tqdm

from problem_models.ProblemModel import ProblemModel
from model_classes.UcbNode import UcbNode


# def find_node_containing_context(context, leaves):
#     for leaf in leaves:
#         if leaf.contains_context(context):
#             return leaf

def find_node_containing_context(context, root_node):
    curr_node: UcbNode = root_node
    is_leaf = curr_node.children is None
    while not is_leaf:
        child: UcbNode
        for child in curr_node.children:
            if child.contains_context(context):
                curr_node = child
                is_leaf = curr_node.children is None
                break
    return curr_node


"""
This class represents the AOM-MC algorithm that is presented in the paper.
"""


class AOMMC:
    problem_model: ProblemModel

    def __init__(self, problem_model: ProblemModel, v1, v2, N, rho, initial_hypercube, exploration_factor=1,
                 scale=False):
        self.scale = scale
        self.exploration_factor = exploration_factor
        self.initial_hypercube = initial_hypercube
        dim = initial_hypercube.get_dimension()
        if not math.log(2 ** dim, N).is_integer():
            print('2^D={} MUST be a power of N={}'.format(2 ** dim, N))
            exit(1)
        self.N = N
        self.num_rounds = problem_model.num_rounds
        self.rho = rho
        self.v2 = v2
        self.v1 = v1
        self.problem_model = problem_model

    def run_algorithm(self):
        max_index = 1
        nodes_played_list = []
        self.num_rounds = self.problem_model.num_rounds
        total_reward_arr = np.zeros(self.num_rounds)
        regret_arr = np.zeros(self.num_rounds)
        root_node = UcbNode(None, 0, [self.initial_hypercube])
        leaves = {root_node}
        node_split_list = []  # stores tuple of (round,node) of split nodes
        node_played_counter_dict = {}
        avg_reward_dict = {}
        time_taken_arr = np.zeros(self.num_rounds)

        for t in tqdm(range(1, self.num_rounds + 1)):
            starting_time = time.time()
            arm_node_dict = {}
            available_workers = self.problem_model.get_available_arms(t)
            index_list = np.zeros(len(available_workers))
            # THE LINE BELOW ASSUMES ALL WORKERS HAVE FIXED COST, which is the case in our simulations
            budget = int(self.problem_model.get_task_budget(t) / available_workers[0].cost)

            # Check if only root node is available
            if len(leaves) == 1:
                arm_indices_to_play = self.problem_model.get_random_arm_indices(t, budget)
            else:
                # rand_work_indices = np.random.choice(len(available_workers), size=len(available_workers), replace=False)
                rand_work_indices = np.arange(len(available_workers))  # removed randomness cause TIM
                for i, worker_ind in enumerate(rand_work_indices):
                    available_worker = available_workers[worker_ind]
                    node = find_node_containing_context(available_worker.context, root_node)
                    arm_node_dict[available_worker] = node
                    index_list[i] = self.get_worker_index(node, node_played_counter_dict, avg_reward_dict)

                if self.scale:
                    # scale indices to [0, 1]
                    if index_list.max() == index_list.min():
                        index_list = np.ones(len(available_workers))
                    else:
                        index_list = (index_list - index_list.min()) / (index_list.max() - index_list.min())

                    # max_index = max(max_index, index_list.max())
                    # index_list /= max_index
                arm_indices_to_play = self.problem_model.oracle(t, budget, index_list,
                                                                available_workers[rand_work_indices])
                arm_indices_to_play = rand_work_indices[arm_indices_to_play]

            selected_nodes = set()
            slate = []
            for index in arm_indices_to_play:
                selected_arm = available_workers[index]
                slate.append(selected_arm)
                if len(leaves) == 1:
                    selected_nodes.add(root_node)
                else:
                    selected_nodes.add(arm_node_dict[selected_arm])
            rewards = self.problem_model.play_arms(t, slate)  # Returns a list of Reward objects

            # Store reward obtained
            total_reward_arr[t - 1] = self.problem_model.get_total_reward(rewards, t)
            regret_arr[t - 1] = self.problem_model.get_regret(t, budget, slate)

            # Update the counters
            played_nodes = []
            for reward in rewards:
                if len(leaves) == 1:
                    node_with_context = next(iter(leaves))
                else:
                    node_with_context = arm_node_dict[reward.worker]
                played_nodes.append(node_with_context)
                new_counter = node_played_counter_dict[node_with_context] = node_played_counter_dict.get(
                    node_with_context, 0) + 1
                avg_reward_dict[node_with_context] = (avg_reward_dict.get(node_with_context, 0) * (
                        new_counter - 1) + reward.performance) / new_counter
            nodes_played_list.append(played_nodes)

            for selected_node in selected_nodes:
                # Split the node if needed
                if self.calc_confidence_radius(node_played_counter_dict[selected_node]) \
                        <= 1 * self.v1 * self.rho ** selected_node.h:
                    produced_nodes = selected_node.reproduce(self.N)
                    node_split_list.append((t, selected_node))
                    leaves.remove(selected_node)
                    leaves.update(produced_nodes)
            time_taken_arr[t - 1] = time.time() - starting_time

        with open('AOM-MC-leaves', 'wb') as output:
            pickle.dump((node_played_counter_dict, avg_reward_dict), output, pickle.HIGHEST_PROTOCOL)
        return {
            'time_taken_arr': time_taken_arr,
            'nodes_played_list': nodes_played_list,
            'node_split_list': node_split_list,
            'root_node': root_node,
            'total_reward_arr': total_reward_arr,
            'regret_arr': regret_arr,
            'leaves': leaves,
            'node_played_counter_dict': node_played_counter_dict,
            'avg_reward_dict': avg_reward_dict,
        }

    def get_worker_index(self, node, node_played_counter_dict, avg_reward_dict):
        num_times_node_played = node_played_counter_dict.get(node, 0)
        avg_reward_of_node = avg_reward_dict.get(node, 0)
        num_times_parent_node_played = node_played_counter_dict.get(node.parent_node, 0)
        avg_reward_of_parent_node = avg_reward_dict.get(node.parent_node, 0)

        node_index = min(avg_reward_of_node + self.calc_confidence_radius(num_times_node_played),
                         avg_reward_of_parent_node + self.calc_confidence_radius(num_times_parent_node_played) +
                         self.exploration_factor * self.v1 * self.rho ** (node.h - 1)) + self.exploration_factor * self.v1 * self.rho ** node.h

        return node_index + self.exploration_factor * self.N * self.v1 / self.v2 * self.v1 * self.rho ** node.h

    def calc_confidence_radius(self, num_times_node_played):
        if num_times_node_played == 0:
            return float('inf')
        return self.exploration_factor * np.sqrt(
            2 * np.log(self.num_rounds * np.sqrt(self.problem_model.get_max_budget() * np.sqrt(2*self.N))) / num_times_node_played)
