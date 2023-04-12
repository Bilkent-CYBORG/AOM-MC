import numpy as np
from tqdm import tqdm

from problem_models import ProblemModel


def find_node_containing_context(context, leaves):
    for leaf in leaves:
        if leaf.contains_context(context):
            return leaf


"""
This class represents a greedy benchmark that picks the K arms with highest means.
"""


class Benchmark:
    problem_model: ProblemModel

    def __init__(self, problem_model: ProblemModel):
        self.num_rounds = problem_model.num_rounds
        self.problem_model = problem_model

    def run_algorithm(self):
        total_reward_arr = np.zeros(self.num_rounds)
        regret_arr = np.zeros(self.num_rounds)
        slate_list = []

        for t in tqdm(range(1, self.num_rounds + 1)):
            available_arms = self.problem_model.get_available_arms(t)
            budget = int(self.problem_model.get_task_budget(t) / available_arms[0].cost)
            true_means = [arm.true_mean for arm in available_arms]
            slate_indices = self.problem_model.oracle(t, budget, true_means, available_arms)
            slate = [available_arms[idx] for idx in slate_indices]
            slate_list.append(slate)
            rewards = self.problem_model.play_arms(t, slate)  # Returns a list of Reward objects

            # Store reward obtained
            total_reward_arr[t - 1] = self.problem_model.get_total_reward(rewards, t)
            regret_arr[t - 1] = 0

        return {
            "bench_slate_list": slate_list,
            'total_reward_arr': total_reward_arr,
            'regret_arr': regret_arr,
        }