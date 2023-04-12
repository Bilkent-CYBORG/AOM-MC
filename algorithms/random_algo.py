import random
import time

import numpy as np
import pickle

from tqdm import tqdm

from problem_models import ProblemModel

"""
This class represents an algorithm that picks arms randomly.
"""


class RandomAlgo:
    problem_model: ProblemModel

    def __init__(self, problem_model: ProblemModel):
        self.num_rounds = problem_model.num_rounds
        self.problem_model = problem_model

    def run_algorithm(self):
        total_reward_arr = np.zeros(self.num_rounds)
        regret_arr = np.zeros(self.num_rounds)
        time_taken_arr = np.zeros(self.num_rounds)

        for t in tqdm(range(1, self.num_rounds + 1)):
            available_workers = self.problem_model.get_available_arms(t)

            # THE LINE BELOW ASSUMES ALL WORKERS HAVE FIXED COST, which is the case in our simulations
            budget = int(self.problem_model.get_task_budget(t) / available_workers[0].cost)

            arm_indices = self.problem_model.get_random_arm_indices(t, budget)
            superarm = [available_workers[idx] for idx in arm_indices]

            rewards = self.problem_model.play_arms(t, superarm)  # Returns a list of Reward objects

            # Store reward obtained
            total_reward_arr[t - 1] = self.problem_model.get_total_reward(rewards, t)
            regret_arr[t - 1] = self.problem_model.get_regret(t, budget, superarm)

        return {
            'time_taken_arr': time_taken_arr,
            'total_reward_arr': total_reward_arr,
            'regret_arr': regret_arr,
        }
