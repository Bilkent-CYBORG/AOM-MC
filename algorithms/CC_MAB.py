import random
import time

import numpy as np
import pickle

from tqdm import tqdm

from problem_models import ProblemModel

"""
This class represents the CC-MAB algorithm.
"""


class CCMAB:
    problem_model: ProblemModel

    def __init__(self, problem_model: ProblemModel, context_dim,
                 unexplored_factor=1, alpha=1-1/np.e):  # Assumes a 1 x 1 x ... x 1 context space
        self.unexplored_factor = unexplored_factor
        self.context_dim = context_dim
        self.num_rounds = problem_model.num_rounds
        self.hT = np.ceil(self.num_rounds ** (1 / (3*alpha + context_dim)))
        # self.hT = 2
        self.cube_length = 1 / self.hT
        self.problem_model = problem_model

    def get_hypercube_of_context(self, context):
        return tuple((context / self.cube_length).astype(int))

    def run_algorithm(self):
        hypercubes_played_list = []
        total_reward_arr = np.zeros(self.num_rounds)
        regret_arr = np.zeros(self.num_rounds)
        hypercube_played_counter_dict = {}
        avg_reward_dict = {}  # maps hypercube to avg reward
        time_taken_arr = np.zeros(self.num_rounds)

        for t in tqdm(range(1, self.num_rounds + 1)):
            starting_time = time.time()
            worker_cube_dict = {}
            arrived_cube_arms_dict = {}
            available_workers = self.problem_model.get_available_arms(t)

            # THE LINE BELOW ASSUMES ALL WORKERS HAVE FIXED COST, which is the case in our simulations
            budget = int(self.problem_model.get_task_budget(t) / available_workers[0].cost)

            # Hypercubes that the arrived arms belong to
            arrived_cube_set = set()
            for available_worker in available_workers:
                hypercube = self.get_hypercube_of_context(available_worker.context)
                worker_cube_dict[available_worker] = hypercube
                if hypercube not in arrived_cube_arms_dict:
                    arrived_cube_arms_dict[hypercube] = list()
                arrived_cube_arms_dict[hypercube].append(available_worker)
                arrived_cube_set.add(hypercube)

            # Identify underexplored hypercubes
            underexplored_arm_set = set()
            for cube in arrived_cube_set:
                if hypercube_played_counter_dict.get(cube, 0) <= self.unexplored_factor * t ** (
                        2 / (3 + self.context_dim)) * np.log(t):
                    underexplored_arm_set.update(arrived_cube_arms_dict[cube])

            # Play arms
            if len(underexplored_arm_set) >= budget:
                slate = random.sample(underexplored_arm_set, budget)
            else:
                slate = []
                slate.extend(underexplored_arm_set)
                not_chosen_arms = list(set(available_workers) - underexplored_arm_set)
                i = 0
                conf_list = np.empty(len(not_chosen_arms))
                for arm in not_chosen_arms:
                    conf_list[i] = avg_reward_dict.get(worker_cube_dict[arm], 0)
                    i += 1
                arm_indices = self.problem_model.oracle(t, budget - len(slate), conf_list, not_chosen_arms)
                for index in arm_indices:
                    selected_arm = not_chosen_arms[index]
                    slate.append(selected_arm)

            rewards = self.problem_model.play_arms(t, slate)  # Returns a list of Reward objects

            # Store reward obtained
            total_reward_arr[t - 1] = self.problem_model.get_total_reward(rewards, t)
            regret_arr[t - 1] = self.problem_model.get_regret(t, budget, slate)

            # Update the counters
            played_cubes = []
            for reward in rewards:
                cube_with_context = worker_cube_dict[reward.worker]
                played_cubes.append(cube_with_context)
                new_counter = hypercube_played_counter_dict[cube_with_context] = hypercube_played_counter_dict.get(
                    cube_with_context, 0) + 1
                avg_reward_dict[cube_with_context] = (avg_reward_dict.get(cube_with_context, 0) * (
                        new_counter - 1) + reward.performance) / new_counter
            hypercubes_played_list.append(played_cubes)
            time_taken_arr[t - 1] = time.time() - starting_time

        with open('CC-MAB-hcubes', 'wb') as output:
            pickle.dump((hypercube_played_counter_dict, avg_reward_dict), output, pickle.HIGHEST_PROTOCOL)
        return {
            'time_taken_arr': time_taken_arr,
            'hypercubes_played_list': hypercubes_played_list,
            'total_reward_arr': total_reward_arr,
            'regret_arr': regret_arr,
            'hypercube_played_counter_dict': hypercube_played_counter_dict,
            'avg_reward_dict': avg_reward_dict,
        }
