import matplotlib.pyplot as plt
import os

import h5py
import numpy as np
import pandas as pd
from pytim import PyTimGraph
from scipy.stats import norm
from tqdm import tqdm

from model_classes.Arm import Arm
from model_classes.Reward import Reward
from problem_models.ProblemModel import ProblemModel
from unused_old.tree_plotter import true_mean_hist, plot_2d_hist
import seaborn as sns

sns.set(style='whitegrid')

"""
Dynamic probabilistic maximum coverage problem model. This is Simulation II in the paper
"""

SAVED_FILE_NAME = "dpmc_simulation.hdf5"  # file where the saved simulation-ready hdf5 dataset will be written to
TEMP_TIM_PATH = "temp_graphs/"
TIM_EPSILON = 0.1
CLIP_MIN = 1e-6


def sigmoid(x):
    return 2 / (1 + np.exp(-4 * x)) - 1


# bottom left corner all 1
# def context_to_mean_fun(a_vect, context):
#     outcome = np.copy(context.mean(axis=0)) / 2
#     outcome[np.logical_and(context[0] < 1/128, context[1] < 1/128)] = 1
#     return outcome


# 1 dim mean fun
# def context_to_mean_fun(a_vect, context):
#     x = context
#     # return ((1 + np.sin(25 * context) * np.cos(37 * context)) / 2).flatten()
#     return ((1 + np.sin(400*x) * np.sin(350*x)) / 2).flatten()

# 3 dim mean fun
def context_to_mean_fun(context):
    x1 = context[0]
    x2 = context[1]
    x3 = context[2]
    # x4 = context[3]
    # x5 = context[4]
    # x6 = context[5]

    # return norm.pdf((1 + np.cos(8*x1) * np.sin(9*x2) * np.cos(10*x3)) / 4 +
    #                 (1 + np.cos(15*x4) * np.sin(17*x5) * np.sin(12*x6)) / 4, loc=0, scale=0.2) / norm.pdf(0, loc=0, scale=0.2)
    return norm.pdf((1 + np.sin(7*x1) * np.cos(8*x2) * np.sin(9*x3)) / 2, loc=0, scale=0.15) / norm.pdf(0, loc=0, scale=0.15)

# 2D mean fun from simple simulations
# def context_to_mean_fun(a_vect, context):
#     x = context[0]
#     y = context[1]
#
#     outcome = np.square((1 + np.sin(5 * x) * np.sin(7 * y)) / 2)
#     return outcome


# def context_to_mean_fun(a_vect, context):
#     # context is a (d, n) matrix where d is the dimension of context (i.e., each column is a separate context)
#     # temp = context.T @ a_vect / a_vect.sum()
#     return norm.pdf(np.cos(4 * np.pi * np.square(context[0] - context[1])), loc=0, scale=0.3) / norm.pdf(0, loc=0,
#                                                                                                          scale=0.3)
#     # return norm.pdf(np.cos(4 * np.pi * np.abs(context[0] - context[1])), loc=0, scale=0.3) / norm.pdf(0, loc=0, scale=0.3)
#     # return (2 + np.sin(50*np.pi*context[0]) + np.cos(50*np.pi*context[1]))/4
#     # return norm.pdf(np.cos(50 * np.pi * np.abs(context[0] - context[1])), loc=0, scale=0.5) / norm.pdf(0, loc=0, scale=0.5)


# def context_to_mean_fun(a_mat, context):
#     # context is a (d, n) matrix where d is the dimension of context (i.e., each column is a separate context)
#     temp = context.T @ a_mat.T @ a_mat @ context
#     diag = np.diag(temp) if isinstance(temp, np.ndarray) else temp
#     diag = np.square(diag)
#     return 0.5 * np.cos(diag) + 0.5
#     # return np.clip(diag, 0, 1)


# def context_to_mean_fun(a_mat, context):
#     return norm.pdf(np.cos(2*np.pi*(context[0] - context[1])), loc=0, scale=0.3) / norm.pdf(0, loc=0, scale=0.3)

# can only be used if context is 2D
def plot_mean_surface(seed=3422):
    rng = np.random.default_rng(seed)
    x_arr = np.linspace(0, 1, 5000)
    xx, yy = np.meshgrid(x_arr, x_arr)
    mean_arr = context_to_mean_fun(np.vstack([xx[np.newaxis], yy[np.newaxis]]))

    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # plt.plot(xx, yy, marker='.', color='k', linestyle='none')
    plt.contour(xx, yy, mean_arr)
    plt.colorbar()
    # ax.plot_surface(xx, xx, mean_arr, color='b')
    plt.tight_layout()
    plt.savefig("mean_contour.pdf", bbox_inches='tight', pad_inches=0.02)
    plt.show()


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


class DPMCProblemModel(ProblemModel):
    df: pd.DataFrame

    def __init__(self, num_rounds, exp_left_nodes, exp_right_nodes, exp_edges_min, exp_edges_max, use_saved, context_dim, min_budget,
                 max_budget, seed=1103938, tim_graph_name="", uniform_context=True, saved_file_name=SAVED_FILE_NAME,
                 plot_mean_hist=True):
        super().__init__(num_rounds)
        self.uniform_context = uniform_context
        self.plot_mean_hist = plot_mean_hist
        self.max_budget = max_budget
        self.min_budget = min_budget
        self.context_dim = context_dim
        self.saved_file_name = saved_file_name
        self.exp_edges_min = exp_edges_min
        self.exp_edges_max = exp_edges_max
        self.tim_graph_name = tim_graph_name
        self.exp_left_nodes = exp_left_nodes
        self.exp_right_nodes = exp_right_nodes
        self.rng = np.random.default_rng(seed)
        if not use_saved:
            self.initialize_dataset()
        self.h5_file = None
        self.benchmark_superarm_list = None

    def set_benchmark_superarm_list(self, superarm_list):
        self.benchmark_superarm_list = superarm_list

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

        t = t - 1
        # keep track of the product of 1-p for each right node, where p is edge probability
        opt_right_id_prob_dict = {}
        algo_right_id_prob_dict = {}
        edge_dataset = self.h5_file[f"{t}"]["edge_dataset"]

        # compute the algorithm's expected reward
        for arm in slate:
            edge_idx = arm.unique_id
            expected_outcome = arm.true_mean
            right_node_id = edge_dataset[edge_idx, 1]
            algo_right_id_prob_dict[right_node_id] = algo_right_id_prob_dict.get(right_node_id, 1) * (
                    1 - expected_outcome)

        algo_exp_number_nodes = np.sum(1 - np.array(list(algo_right_id_prob_dict.values())))

        # compute benchmark expected reward
        if self.benchmark_superarm_list is not None:
            opt_slate = self.benchmark_superarm_list[t]
        else:
            # determine best super arm for this round using TIM+
            available_arms = self.get_available_arms(t + 1)
            true_means = [arm.true_mean for arm in available_arms]
            slate_indices = self.oracle(self.get_task_budget(t + 1), true_means, t + 1)
            opt_slate = [available_arms[idx] for idx in slate_indices]

        for arm in opt_slate:
            edge_idx = arm.unique_id
            expected_outcome = arm.true_mean
            right_node_id = edge_dataset[edge_idx, 1]
            opt_right_id_prob_dict[right_node_id] = opt_right_id_prob_dict.get(right_node_id, 1) * (
                    1 - expected_outcome)

        opt_exp_number_nodes = np.sum(1 - np.array(list(opt_right_id_prob_dict.values())))

        exp_regret = opt_exp_number_nodes - algo_exp_number_nodes
        return exp_regret

    def get_total_reward(self, rewards, t=None):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.saved_file_name, "r")

        t = t - 1  # because time starts from 1 but index starts from 0
        # go through edges and trigger
        activated_users = set()  # activated right nodes
        for reward in rewards:
            edge_idx = reward.worker.unique_id

            # get user id corresponding to this edge
            user_id = self.h5_file[f"{t}"]['edge_dataset'][edge_idx][1]
            is_triggered = reward.performance
            if is_triggered == 1:
                activated_users.add(user_id)
        return len(activated_users)

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

        edge_arr = self.h5_file[f"{t}"]["edge_dataset"][:]
        rand_left_nodes = np.random.choice(np.unique(edge_arr[:, 0]), budget, replace=False)

        edge_indices = np.argwhere(np.in1d(edge_arr[:, 0], rand_left_nodes)).flatten()
        return edge_indices

    def oracle(self, t, budget, est_outcomes, workers=None):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.saved_file_name, "r")

        # This function assumes that g_list is in the order of available arms list. i.e., the ith element of g_list
        # is the estimate prob (i.e., index/gain) of the ith edge of the edge dataset of t

        t = t - 1  # because time starts from 1 but index starts from 0

        edge_arr = self.h5_file[f"{t}"]["edge_dataset"][:]
        num_edges = len(edge_arr)
        num_nodes = len(np.unique(edge_arr.flatten()))
        if not isinstance(est_outcomes, np.ndarray):
            est_outcomes = np.array(est_outcomes)
        edge_probs = np.minimum(1, est_outcomes)
        # edge_probs = self.sigmoid(g_list)

        graph_arr = np.hstack([edge_arr, edge_probs.reshape(-1, 1)])
        temp_graph_path = os.path.join(TEMP_TIM_PATH, f"graph_{self.tim_graph_name}_round_{t}")
        np.savetxt(temp_graph_path, graph_arr,
                   delimiter='\t', fmt=['%d', '%d', '%f'])
        graph = PyTimGraph(bytes(temp_graph_path, 'ascii'),
                           num_nodes, num_edges, budget, bytes('IC', 'ascii'))
        solution_node_ids = graph.get_seed_set(TIM_EPSILON)

        # determine edges connected to chosen left nodes by TIM
        edge_indices = np.argwhere(np.in1d(edge_arr[:, 0], list(solution_node_ids))).flatten()
        return edge_indices

    def get_task_budget(self, t):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.saved_file_name, "r")
        t = t - 1  # because time starts from 1 but index starts from 0
        return self.h5_file[f"{t}"].attrs["budget"]

    def initialize_dataset(self):
        print("Generating synthetic dynamic probabilistic maximum coverage (DPMC) dataset...")

        # sample number of left and right nodes for all rounds. Needed to allocate h5 dataset
        left_node_count_arr = self.rng.poisson(self.exp_left_nodes, self.num_rounds)
        right_node_count_arr = self.rng.poisson(self.exp_right_nodes, self.num_rounds)

        # create h5 dataset
        h5_file = h5py.File(self.saved_file_name, "w")

        # determine random A matrix for computing expected outcome from context
        h5_file.attrs["a_mat"] = a_mat = self.rng.uniform(0, 1, (self.context_dim, 1))

        # TODO ONLY WORKS IF CONTEXT is 3-DIM
        context_norm_mean = np.array([0.5, 0.4, 0.6])
        context_norm_std = np.array([[0.15, 0.1, 0.05],
                                     [0.1, 0.15, 0.07],
                                     [0.05, 0.07, 0.15]])

        total_num_edges = 0
        for time in tqdm(range(self.num_rounds)):
            curr_time_group = h5_file.create_group(f"{time}")

            left_node_ids = np.arange(left_node_count_arr[time])
            # start right node id after largest left node id
            right_node_ids = np.arange(left_node_count_arr[time],
                                       left_node_count_arr[time] + right_node_count_arr[time])

            # determine num edges for each left node
            # num_edges_arr = self.rng.poisson(self.exp_num_edges, left_node_count_arr[time])
            num_edges_arr = self.rng.integers(self.exp_edges_min, self.exp_edges_max + 1, left_node_count_arr[time])
            num_edges_arr = np.minimum(num_edges_arr, right_node_count_arr[time])
            num_edges_arr = np.maximum(num_edges_arr, 1)

            curr_time_group.attrs['budget'] = np.random.randint(self.min_budget, self.max_budget + 1)

            # for each left node randomly pick edges and also context and means
            node_id_set = set()
            edges_list = []
            for left_node_id in left_node_ids:
                node_id_set.add(left_node_id)
                right_node_id_connections = self.rng.choice(right_node_ids, num_edges_arr[left_node_id], replace=False)
                node_id_set.update(list(right_node_id_connections))

                repeated_left_id = np.repeat(left_node_id, num_edges_arr[left_node_id])
                curr_edge_pairs = np.vstack([repeated_left_id, right_node_id_connections]).T

                edges_list.extend(list(curr_edge_pairs))

            # ensure that ever right node is connected to at least one left node
            unconnected_right_nodes = set(range(left_node_count_arr[time] + right_node_count_arr[time])) - node_id_set
            if len(unconnected_right_nodes) > 0:
                for right_node_id in unconnected_right_nodes:
                    # randomly pick a left node and connect it to the unconnected right node
                    rand_left_node_id = self.rng.choice(left_node_ids)
                    edges_list.append(np.array([rand_left_node_id, right_node_id], dtype=np.int32))
            edge_dataset = curr_time_group.create_dataset("edge_dataset", data=np.array(edges_list, dtype=np.int32))

            # generate contexts and expected outcomes (means)
            num_edges = len(edge_dataset)
            if self.uniform_context:
                context_arr = self.rng.random((num_edges, self.context_dim))
            elif self.context_dim == 3:
                context_arr = self.rng.multivariate_normal(context_norm_mean, context_norm_std, size=num_edges)
            elif self.context_dim == 2:
                # mixture of two Gaussian distributions from simple simulations
                mean1 = (0.3, 0.3)
                mean0 = (0.8, 0.8)
                sigma = np.array([[0.05, 0],
                                  [0, 0.05]])
                dist1_prob = 0.5
                selection_arr = self.rng.binomial(1, dist1_prob, num_edges)
                indices_of_1s = np.flatnonzero(selection_arr)
                indices_of_0s = np.flatnonzero(selection_arr == 0)

                context_arr = np.empty((num_edges, self.context_dim))
                context_arr[indices_of_1s] = self.rng.multivariate_normal(mean1, sigma, len(indices_of_1s))
                context_arr[indices_of_0s] = self.rng.multivariate_normal(mean0, sigma, len(indices_of_0s))
            elif self.context_dim == 1:
                context_arr = self.rng.normal(0.5, 0.2, size=(num_edges, 1))
            else:
                raise RuntimeError

            clip_indices = np.logical_or(context_arr <= 0, context_arr >= 1)
            context_arr[clip_indices] = self.rng.random(np.count_nonzero(clip_indices))
            context_dataset = curr_time_group.create_dataset("context_dataset", data=context_arr)

            mean_arr = context_to_mean_fun(context_arr.T)  # transpose because each col should be a context
            mean_dataset = curr_time_group.create_dataset("mean_dataset", data=mean_arr)
            total_num_edges += num_edges

        h5_file.attrs["num_edges"] = total_num_edges
        print(f"Average number of edges: {total_num_edges / self.num_rounds}")

        if self.plot_mean_hist:
            mean_arr = np.concatenate([h5_file[f"{t}"]["mean_dataset"][:] for t in range(self.num_rounds)])
            true_mean_hist(mean_arr, "dpmc_mean_hist")
            if self.context_dim == 2:
                # plot 2D histogram
                context_arr = np.vstack([h5_file[f"{t}"]["context_dataset"][:] for t in range(self.num_rounds)])
                plot_2d_hist(context_arr, "dpmc_context_hist")
            else:
                context_arr = np.vstack(
                    [h5_file[f"{t}"]["context_dataset"][:] for t in range(self.num_rounds)]).flatten()
                true_mean_hist(context_arr, "dpmc_context_hist")


if __name__ == '__main__':
    plot_mean_surface()
