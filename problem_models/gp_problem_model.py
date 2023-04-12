from wolframclient.evaluation import WolframLanguageSession
import warnings

from unused_old.tree_plotter import true_mean_hist

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', "RuntimeWarning: overflow encountered in det")

# import gmaps
import pickle

import gpflow
import numpy as np
import pandas as pd
from tqdm import tqdm

from fs_loader import city_name
from problem_models.ProblemModel import ProblemModel
from model_classes.Reward import Reward
from model_classes.Arm import Arm
import seaborn as sns

sns.set_style("white")

"""
This file contains code for a semi-synthetic problem model that uses the location data from the four square dataset
for worker and task locations and generates the rest of the contexts randomly. This is Simulation III in the paper.
"""

city_gen_df_name_dict = {'tky': 'semi_synth_tky_simulation_df',
                         'nyc': 'semi_synth_nyc_simulation_df'}

saved_df_name = city_gen_df_name_dict[city_name]  # file where the saved simulation-ready dataframe will be saved
# DISTANCE_THRESHOLD = 1 if city_name == 'tky' else np.sqrt(1/2)
DISTANCE_THRESHOLD = np.sqrt(2) / 4
AVAILABILITY_PROB = 1
MIN_NUM_PTS_IN_GRID = 10
quadrant_factor_dict = {0: 1,
                        1: 0.8,
                        2: 0.5,
                        3: 0.3}


# def context_to_mean_fun(context):
#     """
#     context[0] = task location
#     context[1] = worker location
#     context[2] = worker context
#     """
#     return np.sqrt(norm.pdf(np.linalg.norm(context[0] - context[1]), loc=0, scale=0.1) / norm.pdf(0, loc=0, scale=0.1) * context[2])


# def context_to_mean_fun(context):
#     """
#     context[0] = task location
#     GONE context[1] = task context
#     context[1] = worker location
#     context[2] = worker context
#     """
#     return norm.pdf(np.linalg.norm(context[0] - context[1]), loc=0, scale=0.4) * \
#            np.sqrt(context[2]) / norm.pdf(0, loc=0, scale=0.4)

def get_log_of_det(matrix):  # given matrix X, returns log|X|
    det = np.linalg.det(matrix)
    if det != 0 and det != np.inf:
        return np.log(det)

    # because of numerical precision, numpy found the determinant as 0 or inf.
    # to find its log, we us a power scalar s.t. det(2**c * matrix) won't be zero. We do so using binary search
    power_arr = np.arange(0, 50.0, 0.1)
    starting_ind = 0
    ending_ind = len(power_arr)
    found = False
    while not found:
        mid_index = (starting_ind + ending_ind) // 2
        if mid_index == len(power_arr):
            return 0
        power_scalar = power_arr[mid_index]
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            mat_scalar_det = np.linalg.det(2 ** power_scalar * matrix)
        if mat_scalar_det == 0:  # search second half
            starting_ind = mid_index + 1
        elif mat_scalar_det == np.inf:  # search first half
            ending_ind = mid_index - 1
        else:
            found = True

    log_det = np.log(mat_scalar_det) - matrix.shape[0] * power_scalar * np.log(2)
    return log_det


def filter_workers(worker_id_loc_pair, task_loc, distance_thresh):
    """
    Filters the workers based on the distance condition given in the paper
    Parameters
    ----------
    worker_id_loc_pair
    task_loc
    distance_thresh

    Returns
    -------

    """
    return list(
        filter(lambda id_loc_pair: np.linalg.norm(id_loc_pair[1] - task_loc) < distance_thresh, worker_id_loc_pair))


def context_to_mean_fun(context):
    """
    context[0] = task location
    context[1] = worker location
    context[2] = worker pref
    context[3] = task pref
    """
    cos_sim = np.dot(context[2], context[3]) / (np.linalg.norm(context[2]) * np.linalg.norm(context[3]))
    cos_sim_scaled = (cos_sim + 1) / 2
    return cos_sim_scaled, cos_sim_scaled


def get_avail_prob(time_sec):
    # convert to int then back to float, because seconds in day are discrete
    time_sec = time_sec / (24.0 * 3600)
    return (1 + np.sin(2 * np.pi * time_sec - np.pi / 2)) / 2


class GPProblemModel(ProblemModel):
    df: pd.DataFrame

    def __init__(self, num_rounds, exp_avai_workers, use_saved, context_dim, min_budget, max_budget, kernel,
                 task_loc_sq_length=np.sqrt(2) / 8, loc_space_discret_size=100, percent_loc_indices=0.2,
                 time_dep_avai=True, plot_mean_hist=True, num_concurrent_tasks=3):
        super().__init__(num_rounds)
        self.num_concurrent_tasks = num_concurrent_tasks
        self.plot_mean_hist = plot_mean_hist
        self.percent_loc_indices = percent_loc_indices
        self.task_loc_sq_length = task_loc_sq_length
        self.kernel = gpflow.utilities.freeze(kernel)
        self.loc_space_discret_size = loc_space_discret_size

        # minus 6 b/c worker and loc each have 2D location context and 1D battery/difficulty, div 2 b/c task and worker
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.time_dep_avai = time_dep_avai
        if not time_dep_avai:
            assert (context_dim - 4) // 2 == (context_dim - 4) / 2
        if time_dep_avai:
            self.num_prefs = (context_dim - 5) // 2  # 5 b/c 2 for task location, 2 for worker location, 1 for time
        else:
            self.num_prefs = (context_dim - 4) // 2  # 5 b/c 2 for task location, 2 for worker location, 1 for time
        if not use_saved:
            df, cov_mat, loc_discr = self.initialize_df(exp_avai_workers)
            df.set_index('time', inplace=True)
            with open(saved_df_name, 'wb') as output:
                pickle.dump((df, cov_mat, loc_discr), output, pickle.HIGHEST_PROTOCOL)
            self.df = df
            self.cov_mat = cov_mat
            self.loc_discr = loc_discr
        else:
            with open(saved_df_name, 'rb') as input_file:
                self.df, self.cov_mat, self.loc_discr = pickle.load(input_file)

        self.num_workers = len(self.df.loc[1:self.num_rounds].index)

    def get_number_of_workers(self):
        return self.num_workers

    def get_max_budget(self) -> int:
        return self.max_budget

    def get_random_arm_indices(self, t, budget):
        return np.random.choice(np.arange(len(self.df.loc[t])), budget, replace=False)

    def get_cov(self, loc_discr):
        x = loc_discr.reshape(-1, 2)
        return self.kernel(x)

    def sample_gp(self, loc_disc, num_samples):
        x = loc_disc.reshape(-1, 2)
        var = self.kernel(x)
        L = np.linalg.cholesky(var + 1e-14 * np.eye(x.shape[0]))
        z_mat = np.random.normal(size=(x.shape[0], num_samples))
        samples = L @ z_mat

        return samples.T  # each ROW is a sample

    def get_available_arms(self, t):
        # Construct a list of Arm objects
        arm_list = []
        for _, row in self.df.loc[t].iterrows():
            arm_list.append(Arm(len(arm_list), row['context'], row['true_mean'], row['worker_cost']))
        return np.array(arm_list)

    # log reward
    def get_regret(self, t, budget, slate):
        return 0  # TODO: IMPLEMENT
        # df = self.df.loc[t]
        # highest_means = df['true_mean'].nlargest(budget)  # greedy oracle
        # algo_reward_sum = bench_reward_sum = 0
        # for i, worker in enumerate(slate):
        #     bench_reward_sum += AVAILABILITY_PROB * highest_means.iloc[i]
        #     algo_reward_sum += AVAILABILITY_PROB * df.iloc[worker.unique_id]['true_mean']
        # return np.log(1 + bench_reward_sum) - np.log(1 + algo_reward_sum)

    def reward_fun(self, t, worker_loc_indices):
        # get task grid covariances
        loc_indices_in_square = self.df.loc[t].iloc[0]['loc_indices_in_square']
        task_loc_discr_log_det = self.df.loc[t].iloc[0]['task_loc_discr_log_det']

        worker_cov = self.cov_mat[worker_loc_indices][:, worker_loc_indices]
        worker_log_det = get_log_of_det(worker_cov)

        # check if there are non-unique indices (i.e., some worker indices are exatly the same as some task location ones)
        combined_indices = np.unique(np.hstack([worker_loc_indices, loc_indices_in_square]))
        overall_cov = self.cov_mat[combined_indices][:, combined_indices]

        # df = pd.DataFrame(combined_indices)
        # df = df[df.duplicated(keep=False)]
        # df = df.groupby(list(df)).apply(lambda x: tuple(x.index))
        # if len(df) > 0:
        #     for n_tuple in df.tolist():
        #         overall_cov[n_tuple[1:]] += eps
        #         overall_cov[:, n_tuple[1:]] += eps
        #         overall_cov[n_tuple[1:], n_tuple[1:]] -= 2 * eps
        # # combined_indices = np.unique(np.hstack([worker_loc_indices, loc_indices_in_square]))
        # # overall_cov = self.cov_mat[combined_indices][:, combined_indices]

        overall_log_det = get_log_of_det(overall_cov)

        reward = 0.5 * (worker_log_det + task_loc_discr_log_det - overall_log_det)
        return reward

    def get_total_reward(self, rewards, t=None):
        df = self.df.loc[rewards[0].t]

        # get successful worker covariances
        worker_loc_indices = [df.iloc[reward.worker.unique_id]['worker_loc_index'] for reward in rewards if
                              reward.temp_perf == 1]
        if len(worker_loc_indices) == 0:
            return 0

        reward = self.reward_fun(rewards[0].t, worker_loc_indices)

        return reward

    def play_arms(self, t, slate):
        reward_list = []
        df = self.df.loc[t]
        for worker in slate:
            # first check if available
            if self.time_dep_avai:
                task_time_secs = df["task_time_seconds"].iloc[0]
                avail_prob = get_avail_prob(task_time_secs)
            else:
                avail_prob = AVAILABILITY_PROB
            available = np.random.binomial(1, avail_prob)
            # performance = np.random.binomial(1, df.iloc[worker.unique_id]['true_mean']) * available
            temp_perf = np.random.binomial(1, df.iloc[worker.unique_id]['mean_without_loc']) * available
            performance = temp_perf
            # performance = (df.iloc[worker.unique_id]['true_mean'] + np.random.normal(0, 0.05)) * available
            reward_list.append(Reward(worker, performance, temp_perf, t))
        return reward_list

    def oracle(self, t, budget, est_outcomes, workers=None):
        df = self.df.loc[t]

        # est_outcomes represents the worker indices in our algorithm (i.e., UCB worker outcomes)
        # perform greedy marginal reward maximization
        picked_workers = set()  # set of (array) indexes of picked workers
        curr_worker_loc_indices = []
        available_indexes = set(range(len(est_outcomes)))
        curr_reward = 0
        reward_arr = np.zeros(len(workers))
        while len(picked_workers) < budget:
            marginal_reward_arr = np.zeros(len(workers))
            marginal_reward_arr[list(picked_workers)] = -np.inf
            for i in available_indexes:
                est_outcome = est_outcomes[i]
                temp = curr_worker_loc_indices + [df.iloc[workers[i].unique_id]['worker_loc_index']]
                reward_arr[i] = self.reward_fun(t, temp)
                marginal_reward_arr[i] = est_outcome * (reward_arr[i] - curr_reward)
            best_worker_i = marginal_reward_arr.argmax()
            curr_reward = reward_arr[best_worker_i]
            curr_worker_loc_indices.append(df.iloc[workers[best_worker_i].unique_id]['worker_loc_index'])
            picked_workers.add(best_worker_i)
            available_indexes -= {best_worker_i}
        return list(picked_workers)

    def get_task_budget(self, t):
        return self.df.loc[t].iloc[0]['task_budget']

    def initialize_df(self, exp_avai_workers):
        print("Generating workers...")
        # with open('gmm_{}'.format(city_name), 'rb') as input_file:
        #     gmm = pickle.load(input_file)
        session = WolframLanguageSession("/home/sepehr/bin/WolframKernel")
        session.evaluate('ld=Import["/home/sepehr/CC-UCB_IEEE/problem_models/tkyKDE.wmlf"];')
        row_list = []

        # discretize location space
        loc_discr = np.random.random((self.loc_space_discret_size, self.loc_space_discret_size)).reshape(-1, 2)

        # choose task location grid specific indices
        task_specific_loc_indices = np.random.choice(range(len(loc_discr)),
                                                     int(self.percent_loc_indices * len(loc_discr)), replace=False)
        temp_and_bool = np.zeros(len(loc_discr))
        temp_and_bool[task_specific_loc_indices] = 1

        # convert to set
        worker_specific_loc_indices = np.array([x for x in range(len(loc_discr)) if x not in task_specific_loc_indices])

        # plot all points and task location specific points TODO: REMOVE
        # plt.scatter(loc_discr[task_specific_loc_indices, 0], loc_discr[task_specific_loc_indices, 1], s=2, color='r')
        # plt.scatter(loc_discr[worker_specific_loc_indices, 0], loc_discr[worker_specific_loc_indices, 1], s=2, color='g')
        # plt.show()

        cov_mat = self.get_cov(loc_discr).numpy()
        # cov_mat = np.random.random((len(loc_discr), len(loc_discr)))

        for time in tqdm(range(1, self.num_rounds + 1)):
            # given that tasks t, t+1, ..., t+n-1 tasks arrive, where n is num_concurrent_tasks, then each of those
            # tasks arrive in the hour interval [t_0, t_0+1] where t_0= (t+n-1/n) % 24
            # more generally, for any task t, t0 = ((t-1)//n) % 24
            t0 = ((time - 1) // self.num_concurrent_tasks) % 24
            task_time_seconds = np.random.randint(t0 * 3600, (t0 + 1) * 3600)  # the time of arriving task in seconds

            # non uniform task context
            task_pref = np.random.random(self.num_prefs)
            task_budget = np.random.randint(self.min_budget, self.max_budget + 1)
            task_budget = 1 if task_budget < 1 else task_budget
            worker_cost = 1
            budget = int(task_budget / worker_cost)

            # sample num available workers from Po distribution
            num_avai_workers = 0
            while num_avai_workers <= budget:
                num_avai_workers = np.random.poisson(exp_avai_workers)

            avail_worker_locs = []
            # while len(avail_worker_locs) < num_avai_workers:
            while len(avail_worker_locs) < num_avai_workers:
                # filter discretized points s.t. they're inside the task square
                loc_indices_in_square = []
                while len(loc_indices_in_square) < MIN_NUM_PTS_IN_GRID:
                    task_location = np.array(session.evaluate("RandomVariate[ld]"))
                    np.clip(task_location, 0, 1, out=task_location)
                    loc_indices_in_square = np.logical_and(
                        task_location - self.task_loc_sq_length / 2 <= loc_discr,
                        loc_discr <= task_location + self.task_loc_sq_length / 2)
                    loc_indices_in_square = np.vstack(
                        [loc_indices_in_square[:, 0], loc_indices_in_square[:, 1], temp_and_bool]).all(axis=0)
                    loc_indices_in_square = np.argwhere(loc_indices_in_square).flatten()
                task_loc_discr_cov = cov_mat[loc_indices_in_square][:, loc_indices_in_square]
                task_loc_discr_log_det = get_log_of_det(task_loc_discr_cov)

                # plot the task and worker discreitizations
                # task_loc_discr = loc_discr[loc_indices_in_square]
                # print(f"Number of points inside square: {task_loc_discr.shape[0]}")
                #
                # rect = patches.Rectangle(task_location - self.task_loc_sq_length / 2, self.task_loc_sq_length,
                #                          self.task_loc_sq_length, linewidth=1, edgecolor='b', facecolor='none')
                #
                # plt.figure(figsize=(4, 4))
                # plt.gca().add_patch(rect)
                # plt.scatter(loc_discr[worker_specific_loc_indices, 0], loc_discr[worker_specific_loc_indices, 1], s=2, color='g')
                # plt.scatter(loc_discr[task_specific_loc_indices, 0], loc_discr[task_specific_loc_indices, 1], s=2, color='r')
                # plt.scatter(task_loc_discr[:, 0], task_loc_discr[:, 1], s=2, color='b')
                # plt.ylabel("Scaled latitude")
                # plt.xlabel("Scaled longitude")
                # plt.tight_layout()
                # plt.savefig("loc_disc_ex.pdf", bbox_inches='tight', pad_inches=0.01)
                # plt.show()

                # worker_locs = gmm.sample(5 * num_avai_workers)[0]
                # avail_worker_locs = list(
                #     filter(lambda loc: np.linalg.norm(loc - task_location) < DISTANCE_THRESHOLD,
                #            worker_locs))

                # sample workers from discretiziation
                worker_discr_indices = np.random.choice(worker_specific_loc_indices, 5 * num_avai_workers,
                                                        replace=False)
                worker_locs = loc_discr[worker_discr_indices, :]
                condition = np.linalg.norm(worker_locs - task_location, axis=1) < DISTANCE_THRESHOLD
                avail_worker_locs = worker_locs[condition]
                avail_worker_discr_indices = worker_discr_indices[condition]

            # plot worker locations alongside task location discretizations
            # task_loc_discr = loc_discr[loc_indices_in_square]
            # print(f"Number of discretized locations in task square: {task_loc_discr.shape[0]}")
            # print(f"Number of discretized worker locations: {avail_worker_locs[:num_avai_workers].shape[0]}")
            #
            # plt.figure(figsize=(4, 4))
            # plt.scatter(worker_locs[:, 0], worker_locs[:, 1], s=2, color='gray', label="Worker location")
            # plt.scatter(avail_worker_locs[:, 0], avail_worker_locs[:, 1], s=2, color='g', label="Available worker location")
            # plt.scatter(task_loc_discr[:, 0], task_loc_discr[:, 1], s=2, color='b', label="Task square discretization")
            # plt.scatter(task_location[0], task_location[1], s=4, color='r', label="Task location")
            #
            #
            # rect = patches.Rectangle(task_location - self.task_loc_sq_length / 2, self.task_loc_sq_length,
            #                          self.task_loc_sq_length, linewidth=1, edgecolor='b', facecolor='none',
            #                          )
            # circle = patches.Circle(task_location, DISTANCE_THRESHOLD, linewidth=1, edgecolor='r', facecolor='none',
            #                         )
            # plt.gca().add_patch(rect)
            # plt.gca().add_patch(circle)
            #
            # plt.ylabel("Scaled latitude")
            # plt.xlabel("Scaled longitude")
            # plt.xlim([0, 1])
            # plt.ylim([0, 1])
            # plt.legend()
            # plt.tight_layout()
            # plt.savefig("loc_disc_ex.pdf", bbox_inches='tight', pad_inches=0.01)
            # exit(0)

            for worker_loc_index, worker_location in zip(avail_worker_discr_indices[:num_avai_workers],
                                                         avail_worker_locs[:num_avai_workers]):
                np.clip(worker_location, 0, 1, out=worker_location)
                worker_pref = np.random.random(self.num_prefs)

                # compact context is a list of each context, whereas context is the concatenation of each context
                # context is what will be used by AOM-MC to perform the discretization
                # compact_context = [task_location, worker_location, task_context]
                compact_context = [task_location, worker_location, worker_pref, task_pref]

                mean_without_loc, mean_with_loc = context_to_mean_fun(compact_context)
                if self.time_dep_avai:
                    mean_with_loc *= get_avail_prob(task_time_seconds)
                    # add time as a context
                    compact_context.insert(0, task_time_seconds / (24 * 3600))

                context = np.hstack(compact_context)
                row_list.append(
                    (time, task_budget, worker_cost, worker_loc_index, context, mean_without_loc, mean_with_loc,
                     loc_indices_in_square, task_loc_discr_log_det, task_time_seconds))

        df = pd.DataFrame(row_list,
                          columns=['time', 'task_budget', 'worker_cost', 'worker_loc_index', 'context',
                                   'mean_without_loc', 'true_mean', 'loc_indices_in_square',
                                   'task_loc_discr_log_det', "task_time_seconds"])
        if self.plot_mean_hist:
            true_mean_hist(df["true_mean"], "gp_mean_hist")
        session.stop()
        return df, cov_mat, loc_discr
