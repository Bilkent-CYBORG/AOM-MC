import random
import time

import numpy as np
import pickle

from tqdm import tqdm

from problem_models import ProblemModel

"""
This class represents the CUCB algorithm with discretized context regions.
"""


class CUCB:
    problem_model: ProblemModel

    def __init__(self, problem_model: ProblemModel, context_dim, num_cubes_per_dim, exploration_factor=1,
                 save_cubes=True):
        self.save_cubes = save_cubes
        self.num_cubes_per_dim = num_cubes_per_dim
        self.exploration_factor = exploration_factor
        self.context_dim = context_dim
        self.num_rounds = problem_model.num_rounds
        self.cube_length = 1 / num_cubes_per_dim
        self.problem_model = problem_model

    def get_hypercube_of_context(self, context):
        return tuple((context / self.cube_length).astype(int))

    def run_algorithm(self):
        total_reward_arr = np.zeros(self.num_rounds)
        regret_arr = np.zeros(self.num_rounds)
        played_counter_dict = {}  # maps hypercube to num times played
        avg_reward_dict = {}  # maps hypercube to avg reward
        num_cubes = self.num_cubes_per_dim ** self.context_dim
        time_taken_arr = np.zeros(self.num_rounds)

        for t in tqdm(range(1, self.num_rounds + 1)):
            starting_time = time.time()
            available_workers = self.problem_model.get_available_arms(t)

            # THE LINE BELOW ASSUMES ALL WORKERS HAVE FIXED COST, which is the case in our simulations
            budget = int(self.problem_model.get_task_budget(t) / available_workers[0].cost)

            # Hypercubes that the arrived arms belong to
            worker_cube_dict = {worker: self.get_hypercube_of_context(worker.context) for worker in available_workers}

            # if uninitialzied cubes, play arms randomly
            # if len(initialized_cube_set) < num_cubes:
            if t == 1:
                index_arr = np.random.random(len(available_workers))
            else:
                # determine index of each worker: index of worker = index of hypercube that it is inside of
                index_arr = [self.get_index_of_cube(played_counter_dict.get(worker_cube_dict[worker], 0),
                                                   avg_reward_dict.get(worker_cube_dict[worker], 0), t) for worker in available_workers]
                index_arr = np.array(index_arr)

            arm_indices = self.problem_model.oracle(t, budget, index_arr, available_workers)
            superarm = [available_workers[idx] for idx in arm_indices]

            rewards = self.problem_model.play_arms(t, superarm)  # Returns a list of Reward objects

            # Store reward obtained
            total_reward_arr[t - 1] = self.problem_model.get_total_reward(rewards, t)
            regret_arr[t - 1] = self.problem_model.get_regret(t, budget, superarm)

            # Update the counters for each cube
            for reward in rewards:
                cube = worker_cube_dict[reward.worker]
                new_counter = played_counter_dict[cube] = played_counter_dict.get(cube, 0) + 1
                avg_reward_dict[cube] = (avg_reward_dict.get(cube, 0) * (new_counter - 1) + reward.performance) / new_counter
            time_taken_arr[t - 1] = time.time() - starting_time

        if self.save_cubes:
            with open('CUCB-hcubes', 'wb') as output:
                pickle.dump((played_counter_dict, avg_reward_dict), output, pickle.HIGHEST_PROTOCOL)
        return {
            'time_taken_arr': time_taken_arr,
            'total_reward_arr': total_reward_arr,
            'regret_arr': regret_arr,
            'played_counter_dict': played_counter_dict,
            'avg_reward_dict': avg_reward_dict,
        }

    def get_index_of_cube(self, num_times_played, avg_reward, t):
        if num_times_played == 0:
            return float("inf")
        return avg_reward + self.exploration_factor * np.sqrt(3 * np.log(t) / (2 * num_times_played))
