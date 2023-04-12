import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from model_classes.Arm import Arm
from model_classes.Reward import Reward
from problem_models.ProblemModel import ProblemModel
from unused_old.tree_plotter import true_mean_hist, plot_2d_hist

"""
This is Simulation I from the paper.
"""

SAVED_FILE_NAME = "simple_simulation"  # file where the saved simulation-ready hdf5 dataset will be written to


# 2 dim mean fun
def context_to_mean_fun(context):
    x = context[0]
    y = context[1]

    outcome = np.square((1 + np.sin(5 * x) * np.sin(7 * y)) / 2)
    return outcome


# can only be used if context is 1D
def plot_1_dim_context():
    x_arr = np.linspace(0, 1, 1000)
    mean_arr = context_to_mean_fun(x_arr)

    plt.figure()
    plt.plot(x_arr, mean_arr)
    plt.tight_layout()
    plt.savefig("mean_1d_plot.pdf", bbox_inches='tight', pad_inches=0.02)
    plt.show()
    print(f"Average mean is {mean_arr.mean()}")


class SimpleProblemModel(ProblemModel):
    df: pd.DataFrame

    def __init__(self, num_rounds, use_saved, exp_num_arms, context_dim, min_budget, max_budget, non_uniform=True,
                 seed=43213, saved_file_name=SAVED_FILE_NAME, plot_mean_hist=True):
        super().__init__(num_rounds)
        self.non_uniform = non_uniform
        self.exp_num_arms = exp_num_arms
        self.plot_mean_hist = plot_mean_hist
        self.max_budget = max_budget
        self.min_budget = min_budget
        self.context_dim = context_dim
        uni_str = "nuni" if non_uniform else "uni"
        self.saved_file_name = f"{saved_file_name}_{uni_str}.hdf5"
        self.rng = np.random.default_rng(seed)
        if not use_saved:
            self.initialize_dataset()
        self.h5_file = None

    def get_number_of_workers(self):  # return total number of workers/arms/edges
        if self.h5_file is None:
            self.h5_file = h5py.File(self.saved_file_name, "r")
        return self.h5_file.attrs["num_edges"]

    def get_max_budget(self) -> int:
        return self.max_budget

    def get_available_arms(self, t):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.saved_file_name, "r")

        t = t - 1
        # Construct a list of Arm objects (i.e., available edges)
        context_dataset = self.h5_file[f"{t}"]["context_dataset"]
        exp_outcome_dataset = self.h5_file[f"{t}"]["mean_dataset"]

        arm_list = np.empty(len(context_dataset), dtype=object)
        for i, (context, exp_outcome) in enumerate(zip(context_dataset, exp_outcome_dataset)):
            arm_list[i] = Arm(i, np.array(context), exp_outcome)
        return arm_list

    def get_regret(self, t, budget, slate):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.saved_file_name, "r")

        # This function assumes that g_list is in the order of available arms list. i.e., the ith element of g_list
        # is the estimate prob (i.e., index/gain) of the ith edge of the edge dataset of t

        t = t - 1  # because time starts from 1 but index starts from 0

        true_means = self.h5_file[f"{t}"]["mean_dataset"][:]

        highest_means = np.sort(true_means)[-budget:]  # greedy oracle
        algo_reward_sum = bench_reward_sum = 0
        for i, worker in enumerate(slate):
            bench_reward_sum += 1 * highest_means[i]
            algo_reward_sum += 1 * worker.true_mean
        return bench_reward_sum - algo_reward_sum

    def get_total_reward(self, rewards, t=None):
        total_reward = np.sum([reward.performance for reward in rewards])
        return total_reward

    def play_arms(self, t, slate):
        reward_list = [Reward(arm, 1.0 * np.random.binomial(1, arm.true_mean)) for arm in slate]
        return reward_list

    def get_random_arm_indices(self, t, budget):
        # this function samples budget many random left nodes and returns the edges connected to them so that they
        # can be passed to the play_arms function
        if self.h5_file is None:
            self.h5_file = h5py.File(self.saved_file_name, "r")

        # This function assumes that g_list is in the order of available arms list. i.e., the ith element of g_list
        # is the estimate prob (i.e., index/gain) of the ith edge of the edge dataset of t

        t = t - 1  # because time starts from 1 but index starts from 0

        num_arms = len(self.h5_file[f"{t}"]["mean_dataset"][:])
        rand_arms_indices = np.random.choice(range(num_arms), budget, replace=False)

        return rand_arms_indices

    def oracle(self, t, budget, est_outcomes, workers=None):
        return np.argsort(est_outcomes)[-budget:]

    def get_task_budget(self, t):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.saved_file_name, "r")
        t = t - 1  # because time starts from 1 but index starts from 0
        return self.h5_file[f"{t}"].attrs["budget"]

    def initialize_dataset(self):
        print("Generating simple dataset for tree visualization...")

        # create h5 dataset
        h5_file = h5py.File(self.saved_file_name, "w")

        total_num_arms = 0
        for time in tqdm(range(self.num_rounds)):
            curr_time_group = h5_file.create_group(f"{time}")

            num_base_arms = self.rng.poisson(self.exp_num_arms)
            curr_time_group.attrs['budget'] = np.random.randint(self.min_budget, self.max_budget + 1)

            if self.non_uniform:
                # mixture of two Gaussian distributions
                mean1 = (3 * np.pi / 10, 3 * np.pi / 14)
                mean0 = (0.314159, 0.224399)
                sigma = np.array([[0.02, 0],
                                  [0, 0.01]])
                dist1_prob = 0.4
                selection_arr = self.rng.binomial(1, dist1_prob, num_base_arms)
                indices_of_1s = np.flatnonzero(selection_arr)
                indices_of_0s = np.flatnonzero(selection_arr == 0)

                context_arr = np.empty((num_base_arms, self.context_dim))
                context_arr[indices_of_1s] = self.rng.multivariate_normal(mean1, sigma, len(indices_of_1s))
                context_arr[indices_of_0s] = self.rng.multivariate_normal(mean0, sigma, len(indices_of_0s))
            else:
                context_arr = self.rng.random((num_base_arms, self.context_dim))

            # context_arr = self.rng.normal(0.7, 0.2, size=(num_base_arms, self.context_dim))
            # context_arr[context_arr <= 0] = self.rng.uniform(0, 0.5, np.count_nonzero(context_arr <= 0))
            # context_arr[context_arr >= 1] = self.rng.uniform(0, 1, np.count_nonzero(context_arr >= 1))
            clip_indices = np.logical_or(context_arr <= 0, context_arr >= 1)
            context_arr[clip_indices] = self.rng.uniform(0, 1, np.count_nonzero(clip_indices))

            context_dataset = curr_time_group.create_dataset("context_dataset", data=context_arr)

            mean_arr = context_to_mean_fun(context_arr.T)  # transpose because each col should be a context
            mean_dataset = curr_time_group.create_dataset("mean_dataset", data=mean_arr)
            total_num_arms += num_base_arms

        h5_file.attrs["num_edges"] = total_num_arms
        print(f"Average number of edges: {total_num_arms / self.num_rounds}")

        if self.plot_mean_hist:
            uni_str = "nuni" if self.non_uniform else "uni"
            mean_arr = np.concatenate([h5_file[f"{t}"]["mean_dataset"][:] for t in range(self.num_rounds)])
            true_mean_hist(mean_arr, f"simple_{uni_str}_mean_hist")

            if self.context_dim == 2:
                # plot 2D histogram
                context_arr = np.vstack([h5_file[f"{t}"]["context_dataset"][:] for t in range(self.num_rounds)])
                plot_2d_hist(context_arr, f"simple_{uni_str}_context_2d_hist")
            else:
                context_arr = np.vstack([h5_file[f"{t}"]["context_dataset"][:] for t in range(self.num_rounds)]).flatten()
                true_mean_hist(context_arr, f"simple_{uni_str}_context_hist")
