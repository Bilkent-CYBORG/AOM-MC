import os
import time

from algorithms.AOM_MC import AOMMC
from algorithms.CC_MAB import CCMAB
from algorithms.benchmark_algo import Benchmark
from algorithms.cucb import CUCB
from algorithms.eps_greedy import DiscretizedEpsGreedy
from algorithms.random_algo import RandomAlgo
from plot_init import init_plot_params
from problem_models.dpmc_problem_model import DPMCProblemModel
from fractions import Fraction

from problem_models.simple_problem_model import SimpleProblemModel

import multiprocessing
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # disable tensorflow printing logging messages

import tensorflow as tf

# warnings.filterwarnings('error', "invalid value encountered")
# set GPU VRAM to grow as needed
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
import gpflow
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from model_classes.Hypercube import Hypercube
# from fs_problem_model import FoursquareProblemModel
from problem_models.gp_problem_model import GPProblemModel
import argparse

# import matplotlib.font_manager as font_manager
#
# font_dirs = [r'C:\Program Files (x86)\Python37-32\Lib\site-packages\matplotlib\mpl-data\fonts\NimbusRomNo9L-Reg.otf']
# font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
# font_list = font_manager.createFontList(font_files)
# font_manager.fontManager.ttflist.extend(font_list)

# path = r'C:\Program Files (x86)\Python37-32\Lib\site-packages\matplotlib\mpl-data\fonts\NimbusRomNo9L-Reg.otf'
# prop = font_manager.FontProperties(fname=path)
# plt.rcParams['font.family'] = 'Nimbus Roman No9 L'
# mpl.rcParams['font.sans-serif'] = 'Nimbus Roman No9 L'

# rc('text', usetex=True)
# sns.set(style='whitegrid', font='Nimbus Roman No9 L', rc={'text.usetex': True})
# plt.style.use('seaborn')


"""
!!! Note that when the parameter below is set to True, the script will use the arm-pairs that were used to produce the 
figures in the paper. When set to False, the script will generate arm-pairs (i.e., number of arms, worker batteries, etc)
and run the simulations on them. Therefore, to reproduce the results in the paper, it must be set to True.!!!
"""

# run types
MULTIPLE_ROUNDS = 'multiple_round'
SINGLE_ROUND = 'single_round'

# problem model types
GP_MODEL = "gp"  # Simulation III
DPMC_MODEL = "dpmc"  # Simulation II
SIMPLE_MODEL_UNI = "simple_uni"  # Simulation I uniform context arrivals
SIMPLE_MODEL_NUNI = "simple_nuni"  # Simulation II non-uniform context arrivals

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--model_type", default=GP_MODEL, help="Which model type to run. You can choose from "
                                                           "simple_uni, simple_nuni, dpmc, or gp. simple_uni and "
                                                           "simple_nuni simulate Simulation I with uniform and "
                                                           "non-uniform arrivals, respectively. dpmc simulates "
                                                           "Simulation II and gp simulates Simulation III.")

parser.add_argument("--running_mode", default=SINGLE_ROUND, nargs="?",
                    help="If set to single_round, the script will run a single round of simulations. If set to "
                         "multiple_rounds, the script will run multiple rounds of simulations with different T values.")
parser.add_argument("--num_threads", default=1, nargs="?", type=int,
                    help="Number of threads to use for running the simulations. When set to -1, will use all available")
parser.add_argument("--num_reps", default=1, nargs="?", type=int,
                    help="How many times each run should be repeated.")
parser.add_argument("--use_saved_sim", default=False, nargs="?", type=bool,
                    help="Whether to use the saved simulation problem models. When set to True, the script will use "
                         "the already-generated problem models and will not regenerate them as new.")
parser.add_argument("--plot_refine", default=False, nargs="?", type=bool,
                    help="Whether to plot in which round AOM-MC made refinements to nodes.")

args = parser.parse_args()

running_mode = args.running_mode  # 'single_round', 'multiple_rounds'
model_type = args.model_type

use_saved_data = True  # when True, the script simply plots the data of the most recently ran simulation, if available
# this means that no simulations are run when True.
use_generated_workers_in_paper = args.use_saved_sim

rolling = False
REWARD_MEAN_WINDOW_SIZE = 500
plot_refine = args.plot_refine

num_threads_to_use = args.num_threads  # number of threads to run the simulation on. When set to -1, will run on all available threads
if num_threads_to_use == -1:
    num_threads_to_use = int(multiprocessing.cpu_count())

num_times_to_run = args.num_reps

# if MULTIPLE_ROUNDS, then each algorithm is run for each round num in num_rounds_arr to plot regret
# if SINGLE_ROUND, then each algorithm is run using the largest round num (i.e., second arg of linspace). Used for
# plotting reward and time taken
if model_type == GP_MODEL:
    num_rounds_arr = np.linspace(100, 3000, 15).astype(int)
elif model_type == DPMC_MODEL:
    num_rounds_arr = np.linspace(100, 40000, 15).astype(int)
elif model_type == SIMPLE_MODEL_UNI or model_type == SIMPLE_MODEL_NUNI:
    num_rounds_arr = np.linspace(100, 300000, 15).astype(int)

# GP params
if model_type == GP_MODEL:
    kernel = gpflow.kernels.Matern52(0.05, 0.05)
    loc_space_discret_size = 170
    max_budget = 8
    min_budget = 3
    exp_num_workers = 50
    context_dim = 9
    aommc_factor_arr = [1]
    cucb_num_cubes_per_dim_arr = [2, 3, 4]
    cucb_factor_arr = [1]
    eps_num_cubes_per_dim_arr = [2, 3, 4]
    scale_aommc_indices = [False]  # no need to scale indices to [0, 1]

elif model_type == DPMC_MODEL:
    # synthetic DPMC params
    exp_left_nodes = 50
    exp_right_nodes = 300
    max_budget = 10
    min_budget = 4
    context_dim = 3
    exp_edges_min = 1
    exp_edges_max = 4

    cucb_num_cubes_per_dim_arr = [4, 8, 16, 32, 64, 128]
    cucb_factor_arr = [1]
    aommc_factor_arr = [1]
    scale_aommc_indices = [True]  # must scaled indices to [0, 1] because they represent edge probabilities

elif model_type == SIMPLE_MODEL_UNI or model_type == SIMPLE_MODEL_NUNI:
    # simple model params
    min_budget = max_budget = 100
    context_dim = 2
    exp_base_arms = 350
    aommc_factor_arr = [1]
    cucb_factor_arr = cucb_num_cubes_per_dim_arr = []
    scale_aommc_indices = [False]
    is_non_uni = model_type == SIMPLE_MODEL_NUNI

# ACC-UCB and AOM-MC params
v1 = np.sqrt(context_dim)
v2 = 1
rho = 0.5
N = 2 ** context_dim
root_hypercube = Hypercube(1, np.full(context_dim, 0.5))  # this is called x_{0,1} in the paper

reference_algo = bench_name = "Benchmark"
aommc_name = "AOM-MC"
ccmab_name = "CC-MAB"
cucb_name = "CUCB"
eps_name = r"$\epsilon_n$-greedy"
rand_name = "Random"

# names of algorithms to plot. If empty plot all
plotting_names_arr = []


def run_one_try(problem_model, run_num):
    algo_result_dict = {}

    if model_type == DPMC_MODEL:
        problem_model.tim_graph_name = f"run_num_{run_num}"

    # run random algorithm
    print("Running random algorithm")
    rand_algo = RandomAlgo(problem_model)
    algo_result_dict[rand_name] = rand_algo.run_algorithm()

    for num_cubes_per_dim in eps_num_cubes_per_dim_arr:
        eps_algo = DiscretizedEpsGreedy(problem_model, context_dim, num_cubes_per_dim, c=0.1, d=0.2)
        print(f"Running eps-greedy with {num_cubes_per_dim} cubes per dim...")
        algo_result_dict[
            f"{eps_name} with {num_cubes_per_dim} disc./dim"] = eps_algo.run_algorithm()

    for num_cubes_per_dim in cucb_num_cubes_per_dim_arr:
        for factor in cucb_factor_arr:
            cucb_algo = CUCB(problem_model, context_dim, num_cubes_per_dim, exploration_factor=factor)
            str_name = f"{cucb_name} with {num_cubes_per_dim} disc./dim and factor {Fraction(factor)}" if factor > 1 else \
                f"{cucb_name} with {num_cubes_per_dim} disc./dim"
            print(f"Running CUCB with {num_cubes_per_dim} cubes per dim and {Fraction(factor)} exp. factor...")
            algo_result_dict[str_name] = cucb_algo.run_algorithm()

    if model_type != SIMPLE_MODEL_UNI and model_type != SIMPLE_MODEL_NUNI and model_type != GP_MODEL:
        print('Running Benchmark...')
        bench_algo = Benchmark(problem_model)
        algo_result_dict[bench_name] = bench_algo.run_algorithm()
    if model_type == DPMC_MODEL:
        # save benchmark choices to be used later when computing regret
        problem_model.set_benchmark_superarm_list(algo_result_dict[bench_name]["bench_slate_list"])

    for scale, factor in zip(scale_aommc_indices, aommc_factor_arr):
        aom_mc_algo = AOMMC(problem_model, v1, v2, N, rho, root_hypercube, exploration_factor=factor, scale=scale)
        print(f"Running AOM-MC with exploration factor {Fraction(factor)}...")
        aom_result = aom_mc_algo.run_algorithm()
        if factor == 1:
            algo_result_dict[aommc_name] = aom_result
        else:
            algo_result_dict[f"{aommc_name} with factor {Fraction(factor)}"] = aom_result

    if model_type == GP_MODEL:
        cc_mab_algo = CCMAB(problem_model, root_hypercube.get_dimension())
        print("Running CC-MAB...")
        mab_result = cc_mab_algo.run_algorithm()
        algo_result_dict[ccmab_name] = mab_result
    return algo_result_dict


def run_once_num_round(num_rounds):
    if model_type == GP_MODEL:
        problem_model = GPProblemModel(num_rounds, exp_num_workers, use_generated_workers_in_paper, context_dim,
                                       min_budget, max_budget, kernel, loc_space_discret_size=loc_space_discret_size)
    elif model_type == DPMC_MODEL:
        problem_model = DPMCProblemModel(num_rounds, exp_left_nodes, exp_right_nodes, exp_edges_min, exp_edges_max,
                                         use_generated_workers_in_paper, context_dim, min_budget, max_budget)
    elif model_type == SIMPLE_MODEL_UNI or model_type == SIMPLE_MODEL_NUNI:
        problem_model = SimpleProblemModel(num_rounds, use_generated_workers_in_paper, exp_base_arms, context_dim,
                                           min_budget, max_budget, is_non_uni)

    else:
        raise RuntimeError("No such model type!")

    print("Running simulations on {thread_count} threads".format(thread_count=num_threads_to_use))
    parallel_results = Parallel(n_jobs=num_threads_to_use)(
        delayed(run_one_try)(problem_model, i) for i in range(num_times_to_run))

    with open("{}_parallel_results_rounds_{}".format(model_type, num_rounds), 'wb') as output:
        pickle.dump(parallel_results, output, pickle.HIGHEST_PROTOCOL)
    return parallel_results


def run_for_diff_num_rounds():
    if not use_generated_workers_in_paper:  # preload problem model with max num rounds
        if model_type == GP_MODEL:
            problem_model = GPProblemModel(max(num_rounds_arr), exp_num_workers, False, context_dim,
                                           min_budget, max_budget, kernel,
                                           loc_space_discret_size=loc_space_discret_size)
        elif model_type == DPMC_MODEL:
            problem_model = DPMCProblemModel(max(num_rounds_arr), exp_left_nodes, exp_right_nodes, exp_edges_min,
                                             exp_edges_max,
                                             use_generated_workers_in_paper, context_dim, min_budget, max_budget)
        else:
            raise RuntimeError("No such model type!")

    parallel_results_list = []
    print("Running simulations on {thread_count} threads".format(thread_count=num_threads_to_use))

    for num_rounds in tqdm(num_rounds_arr):
        if model_type == GP_MODEL:
            problem_model = GPProblemModel(num_rounds, exp_num_workers, True, context_dim,
                                           min_budget, max_budget, kernel,
                                           loc_space_discret_size=loc_space_discret_size)

        elif model_type == DPMC_MODEL:
            problem_model = DPMCProblemModel(num_rounds, exp_left_nodes, exp_right_nodes, exp_edges_min, exp_edges_max,
                                             True, context_dim, min_budget, max_budget)
        else:
            raise RuntimeError("No such model type!")

        print("Doing {} many rounds...".format(num_rounds))
        parallel_results = Parallel(n_jobs=num_threads_to_use)(
            delayed(run_one_try)(problem_model, i) for i in range(num_times_to_run))
        parallel_results_list.append(parallel_results)

        with open("{}_parallel_results_rounds_{}".format(model_type, num_rounds), 'wb') as output:
            pickle.dump(parallel_results, output, pickle.HIGHEST_PROTOCOL)
    return parallel_results_list


def plot_cum_regret(results_list):
    init_plot_params()
    algo_names = list(results_list[0][0].keys())
    num_Ts = len(results_list)
    cum_regret_arr = np.zeros((len(algo_names), len(results_list[0]), num_Ts))  # algo, repeat, T

    for i, results in enumerate(results_list):
        for j, result in enumerate(results):
            for k, algo_name in enumerate(algo_names):
                algo_dict = result[algo_name]
                final_regret = np.cumsum(algo_dict['regret_arr'])[-1]
                cum_regret_arr[k, j, i] = final_regret

    cum_regret_avg = cum_regret_arr.mean(axis=1)
    cum_regret_std = cum_regret_arr.std(axis=1)

    plt.figure(figsize=(6.4, 4))
    for i, algo_name in enumerate(algo_names):
        if algo_name != reference_algo:
            color = next(plt.gca()._get_lines.prop_cycler)['color']
            mean, std = cum_regret_avg[i], cum_regret_std[i]
            plt.plot(num_rounds_arr, mean, label=algo_name.replace("CCGP-UCB", "O'CLOK-UCB"), color=color)
            plt.fill_between(num_rounds_arr, mean - std, mean + std, alpha=0.3, color=color)

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.legend()
    plt.xlabel("Number of rounds ($T$)")
    plt.ylabel("Cumulative regret")
    plt.tight_layout()
    plt.savefig("cum_regret.pdf", bbox_inches='tight', pad_inches=0.03)


def get_reward_reg_time_from_result(parallel_results, algo_names):
    algo_reward_dict = {}
    algo_cum_reward_dict = {}
    algo_regret_dict = {}
    algo_time_dict = {}
    aom_round_refine_arr = []  # array of array. Inner ith array contains round in which a node was refined in ith run.
    num_times_to_run = len(parallel_results)
    for i, entry in enumerate(parallel_results):
        aom_round_refine_arr.append([])
        for algo_name in algo_names:
            result = entry[algo_name]

            if algo_name == aommc_name:
                prev_round_no = -1
                for round_no, _ in result['node_split_list']:
                    if round_no > prev_round_no * 1.0:
                        aom_round_refine_arr[-1].append(round_no)
                    prev_round_no = round_no

            if algo_name not in algo_reward_dict:
                num_rounds = len(result['total_reward_arr'])
                algo_reward_dict[algo_name] = np.zeros((num_times_to_run, num_rounds))
                algo_cum_reward_dict[algo_name] = np.zeros((num_times_to_run, num_rounds))
                algo_regret_dict[algo_name] = np.zeros((num_times_to_run, num_rounds))
                algo_time_dict[algo_name] = np.zeros((num_times_to_run, num_rounds))

            if rolling:
                algo_reward_dict[algo_name][i] = pd.Series(result['total_reward_arr']).rolling(
                    REWARD_MEAN_WINDOW_SIZE).mean().values
            else:
                algo_cum_reward_dict[algo_name][i] = result['total_reward_arr'].cumsum()
                algo_reward_dict[algo_name][i] = pd.Series(result['total_reward_arr']).expanding().mean().values
                # algo_reward_dict[algo_name][i] = np.cumsum(result['total_reward_arr'])
            algo_regret_dict[algo_name][i] = np.cumsum(result['regret_arr'])
            if algo_name != reference_algo:
                algo_time_dict[algo_name][i] = np.cumsum(result['time_taken_arr'])
        for algo_name in algo_names:
            if algo_name != reference_algo and reference_algo in algo_reward_dict:
                algo_reward_dict[algo_name][i] /= algo_reward_dict[reference_algo][i]

    # make sure arrays have same number of elements
    max_arr = max(aom_round_refine_arr, key=len)
    max_refine_arr_len = len(max_arr)
    for arr in aom_round_refine_arr:
        while len(arr) < max_refine_arr_len:
            arr.append(max_arr[len(arr) - max_refine_arr_len])

    aom_avg_round_refine = np.mean(aom_round_refine_arr, axis=0)
    return algo_reward_dict, algo_cum_reward_dict, algo_regret_dict, algo_time_dict, aom_avg_round_refine


def plot_reward_and_time(parallel_results):
    init_plot_params()
    algo_names = list(parallel_results[0].keys())
    if len(plotting_names_arr) > 0:
        algo_names = [name for name in algo_names if name in plotting_names_arr or name == bench_name]
    num_rounds = len(parallel_results[0][algo_names[0]]['total_reward_arr'])

    algo_reward_dict, algo_cum_reward_dict, algo_regret_dict, algo_time_dict, aom_avg_round_refine = \
        get_reward_reg_time_from_result(parallel_results, algo_names)

    algo_reward_avg_dict = {}
    algo_reward_std_dict = {}
    algo_cum_reward_avg_dict = {}
    algo_cum_reward_std_dict = {}
    algo_regret_avg_dict = {}
    algo_regret_std_dict = {}
    algo_time_avg_dict = {}
    algo_time_std_dict = {}
    for algo_name in algo_names:
        algo_reward_avg_dict[algo_name] = algo_reward_dict[algo_name].mean(axis=0)
        algo_reward_std_dict[algo_name] = algo_reward_dict[algo_name].std(axis=0)
        algo_cum_reward_avg_dict[algo_name] = algo_cum_reward_dict[algo_name].mean(axis=0)
        algo_cum_reward_std_dict[algo_name] = algo_cum_reward_dict[algo_name].std(axis=0)
        algo_regret_avg_dict[algo_name] = algo_regret_dict[algo_name].mean(axis=0)
        algo_regret_std_dict[algo_name] = algo_regret_dict[algo_name].std(axis=0)
        algo_time_avg_dict[algo_name] = algo_time_dict[algo_name].mean(axis=0)
        algo_time_std_dict[algo_name] = algo_time_dict[algo_name].std(axis=0)

        xnew = np.arange(1, num_rounds + 1)
        # smooth
        # xnew = np.linspace(1, num_rounds, 100)
        # spl = make_interp_spline(range(1, num_rounds + 1), algo_reward_avg_dict[algo_name], k=3)  # type: BSpline
        # algo_reward_avg_dict[algo_name] = spl(xnew)
        #
        # spl = make_interp_spline(range(1, num_rounds + 1), algo_reward_std_dict[algo_name], k=3)  # type: BSpline
        # algo_reward_std_dict[algo_name] = spl(xnew)
        #
        # algo_reward_avg_dict[algo_name][0] = algo_reward_std_dict[algo_name][0] = 0

    # PLOT CUMULATIVE REGRET
    # for algo_name in algo_reward_dict.keys():
    #     algo_regret_std_dict[algo_name][indices] = None
    #
    # plt.figure(1)
    # for algo_name in algo_reward_dict.keys():
    #     if algo_name != reference_algo:
    #         plt.errorbar(range(1, num_rounds + 1), algo_regret_avg_dict[algo_name],
    #                      yerr=algo_regret_std_dict[algo_name],
    #                      label=algo_name, capsize=2, linewidth=2)
    #
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # plt.legend()
    # plt.xlabel("Round number $(t)$")
    # plt.ylabel("Cumulative regret up to $t$")
    # plt.tight_layout()
    # plt.savefig("cum_regret.pdf", bbox_inches='tight', pad_inches=0.03)

    # PLOT AVERAGE REWARD

    fig = plt.figure(figsize=(6.4, 4))
    ax = fig.add_subplot(111)

    for algo_name in algo_names:
        if algo_name != reference_algo:
            color = next(plt.gca()._get_lines.prop_cycler)['color']
            mean, std = algo_reward_avg_dict[algo_name], algo_reward_std_dict[algo_name]
            ax.plot(xnew, mean, label=algo_name, color=color)
            ax.fill_between(xnew, mean - std, mean + std, alpha=0.3, color=color)

    if plot_refine:
        round_refine_y = np.interp(aom_avg_round_refine, xnew, algo_reward_avg_dict[aommc_name])
        ax.scatter(aom_avg_round_refine, round_refine_y, marker="x", color='black', zorder=3, label="Node refined")

    plt.legend()
    # plt.xlim(0, 200)
    # plt.ylim(0.8, 1)  # We need to do this b/c otherwise the legend was not seen
    plt.xlabel("Round number $(t)$")
    plt.ylabel("Average task reward divided by\nbenchmark reward up to round $t$")
    plt.tight_layout()
    plt.savefig("avg_reward.pdf", bbox_inches='tight', pad_inches=0.03)

    # PLOT CUMULATIVE REWARD
    fig = plt.figure(figsize=(6.4, 4))
    ax = fig.add_subplot(111)
    for algo_name in algo_names:
        if algo_name != reference_algo:
            color = next(plt.gca()._get_lines.prop_cycler)['color']
            mean, std = algo_cum_reward_avg_dict[algo_name], algo_cum_reward_std_dict[algo_name]
            ax.plot(xnew, mean, label=algo_name, color=color)
            ax.fill_between(xnew, mean - std, mean + std, alpha=0.3, color=color)

    plt.legend()
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.xlabel("Round number $(t)$")
    plt.ylabel("Cumulative reward up to round $t$")
    plt.tight_layout()
    plt.savefig("avg_cum_reward.pdf", bbox_inches='tight', pad_inches=0.03)

    # PLOT TIME TAKEN
    plt.figure(figsize=(6.4, 4))
    for algo_name in algo_names:
        if algo_name != reference_algo:
            color = next(plt.gca()._get_lines.prop_cycler)['color']
            mean, std = algo_time_avg_dict[algo_name], algo_time_std_dict[algo_name]
            plt.plot(range(1, 1 + num_rounds), mean, label=algo_name.replace("CCGP-UCB", "O'CLOK-UCB"), color=color)
            plt.fill_between(range(1, 1 + num_rounds), mean - std, mean + std, alpha=0.3, color=color)

    plt.legend()
    # plt.xlim(0, 200)
    # plt.ylim(0.95, 1)  # We need to do this b/c otherwise the legend was not seen
    plt.xlabel("Arriving task $(t)$")
    plt.ylabel("Time taken (s)")
    plt.tight_layout()
    plt.savefig("time_taken.pdf", bbox_inches='tight', pad_inches=0.03)


def get_reg_from_multi_round(results_list, gp_alg_names, algo_names, num_Ts):
    non_gp_alg_names = [x for x in algo_names if x not in gp_alg_names]
    cum_regret_arr = np.zeros((len(algo_names), len(results_list[0]), num_Ts))  # algo, repeat, T

    for i, results in enumerate(results_list):
        for j, result in enumerate(results):
            if i == len(results_list) - 1:  # last result is one with most num rounds so GP algs will be included
                for k, algo_name in enumerate(gp_alg_names):
                    algo_dict = result[algo_name]
                    for m, final_T in enumerate(num_rounds_arr):
                        cum_regret_arr[algo_names.index(algo_name), j, m] = np.cumsum(algo_dict['regret_arr'])[
                            final_T - 1]
            for k, algo_name in enumerate(non_gp_alg_names):
                algo_dict = result[algo_name]
                final_regret = np.cumsum(algo_dict['regret_arr'])[-1]
                cum_regret_arr[algo_names.index(algo_name), j, i] = final_regret

    return cum_regret_arr


if __name__ == '__main__':
    starting_time = time.time()
    if not use_saved_data:
        if running_mode == MULTIPLE_ROUNDS:
            print(f"Running with different round numbers (i.e., different T) using {model_type} problem setting")
            parallel_results_list = run_for_diff_num_rounds()
            plot_cum_regret(parallel_results_list)
            plot_reward_and_time(parallel_results_list[-1])
            # plot_mut_information(parallel_results_list[-1], kernel)
        elif running_mode == SINGLE_ROUND:
            print(f"Running with one round number (T={max(num_rounds_arr)}) using {model_type} problem setting")
            parallel_results = run_once_num_round(max(num_rounds_arr))
            if model_type != SIMPLE_MODEL_UNI and model_type != SIMPLE_MODEL_NUNI:  # to plot discretizations see tree_plotter.py
                plot_reward_and_time(parallel_results)
            # plot_mut_information(parallel_results, kernel)

    else:
        if running_mode == MULTIPLE_ROUNDS:
            parallel_results_list = []
            for num_rounds in num_rounds_arr:
                with open('{}_parallel_results_rounds_{}'.format(model_type, num_rounds), 'rb') as input_file:
                    parallel_results = pickle.load(input_file)
                parallel_results_list.append(parallel_results)
            plot_cum_regret(parallel_results_list)
            plot_reward_and_time(parallel_results_list[-1])
            # plot_mut_information(parallel_results_list[-1], kernel)

        elif running_mode == SINGLE_ROUND:
            with open('{}_parallel_results_rounds_{}'.format(model_type, max(num_rounds_arr)), 'rb') as input_file:
                parallel_results = pickle.load(input_file)
            # with open('{}_parallel_results_rounds_{}_all'.format(model_type, max(num_rounds_arr)), 'rb') as input_file:
            #     parallel_results_aom = pickle.load(input_file)
            # with open("dpmc_parallel_results_rounds_40000_cucb_4_8", 'rb') as input_file:
            #     parallel_results_cucb = pickle.load(input_file)
            # parallel_results = []
            # temp_keys = ["Benchmark", "AOM-MC"] + [f"CUCB with {x} cubes and factor 1" for x in [4, 8, 16, 32, 64, 128]]
            # temp_good_keys = ["Benchmark", "AOM-MC"] + [f"CUCB with {x} disc./dim" for x in [4, 8, 16, 32, 64, 128]]
            # for a, b in zip(parallel_results_cucb, parallel_results_aom):
            #     a.update(b)
            #     temp_dict = {}
            #     for key, good_key in zip(temp_keys, temp_good_keys):
            #         temp_dict[good_key] = a[key]
            #     parallel_results.append(temp_dict)
            # with open("dpmc_parallel_results_rounds_40000", 'wb') as output:
            #     pickle.dump(parallel_results, output, pickle.HIGHEST_PROTOCOL)

            if model_type != SIMPLE_MODEL_UNI and model_type != SIMPLE_MODEL_NUNI:  # to plot discretizations see tree_plotter.py
                plot_reward_and_time(parallel_results)
            # plot_mut_information(parallel_results, kernel)

    print(f"Total running time: {time.time() - starting_time: .2f}s")
    plt.show()
