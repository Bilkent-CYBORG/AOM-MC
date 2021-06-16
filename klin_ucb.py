# Adapted from https://github.com/etiennekintzler/bandits_algorithm/blob/master/linUCB.ipynb
import time

import numpy as np
from tqdm import tqdm

from problem_models.ProblemModel import ProblemModel


class KLinUCB:
    problem_model: ProblemModel

    def __init__(self, problem_model: ProblemModel, delta, context_dim, max_budget):
        self.delta = delta
        self.context_dim = context_dim
        self.num_rounds = problem_model.num_rounds
        self.problem_model = problem_model
        self.alpha = np.sqrt(0.5 * np.log(2 * self.num_rounds * max_budget / delta))

    def run_algorithm(self):
        A = np.diag(np.ones(self.context_dim))
        b = np.zeros(self.context_dim)
        total_reward_arr = np.zeros(self.num_rounds)
        regret_arr = np.zeros(self.num_rounds)
        time_taken_arr = np.zeros(self.num_rounds)

        for t in tqdm(range(1, self.num_rounds + 1)):
            starting_time = time.time()
            available_workers = self.problem_model.get_available_arms(t)

            # THE LINE BELOW ASSUMES ALL WORKERS HAVE FIXED COST, which is the case in our simulations
            budget = int(self.problem_model.get_task_budget(t) / available_workers[0].cost)

            p_arr = np.empty(len(available_workers))
            inv_A = np.linalg.inv(A)
            theta = inv_A @ b
            for i, worker in enumerate(available_workers):
                p_arr[i] = theta.dot(worker.context) + self.alpha * np.sqrt(
                    worker.context.dot(inv_A).dot(worker.context))

            worker_indices_to_play = p_arr.argsort()[-budget:]
            workers_to_play = [available_workers[i] for i in worker_indices_to_play]

            rewards = self.problem_model.play_arms(t, workers_to_play)  # Returns a list of Reward objects
            for reward in rewards:
                A += np.outer(reward.context, reward.context)
                b += reward.performance * reward.context

            # Store reward obtained
            total_reward_arr[t - 1] = self.problem_model.get_total_reward(rewards, t)
            regret_arr[t - 1] = self.problem_model.get_regret(t, budget, workers_to_play)
            time_taken_arr[t - 1] = time.time() - starting_time

        return {
            'time_taken_arr': time_taken_arr,
            'total_reward_arr': total_reward_arr,
            'regret_arr': regret_arr
        }
