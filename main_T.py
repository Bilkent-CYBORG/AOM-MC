import multiprocessing
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from tqdm import tqdm

from CC_MAB import CCMAB
from AOM_MC import AOMMC
from Hypercube import Hypercube
from benchmark_algo import Benchmark
from problem_models.gowalla_problem_model import GowallaProblemModel

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
sns.set(style='whitegrid', font='Nimbus Roman No9 L',)

"""
This python script is responsible for running ACC-UCB, CC-MAB, Random, and benchmark on the Gowalla dataset
for a given number of times.
!!! Note that when the parameter below is set to True, the script will use the arm-pairs that were used to produce the 
figures in the paper. When set to False, the script will generate arm-pairs (i.e., number of arms, worker batteries, etc)
and run the simulations on them. Therefore, to reproduce the results in the paper, it must be set to True.!!!
"""
use_generated_workers_in_paper = True

rolling = False
is_uniform = False
REWARD_MEAN_WINDOW_SIZE = 500

num_threads_to_use = 4  # number of threads to run the simulation on. When set to -1, will run on all available threads
use_saved_data = False  # when True, the script simply plots the data of the most recently ran simulation, if available
# this means that no simulations are run when True.

num_times_to_run = 4
num_rounds_arr = np.linspace(100, 5000, 5, dtype=np.int)
num_std_to_show = 5
exp_num_workers = 100
line_style_dict = '-'

v1 = np.sqrt(13)
v2 = 1
rho = 0.5
N = 8
root_hypercube = Hypercube(1, np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))  # this is called x_{0,1} in the paper


def run_one_try(problem_model):
    # random_algo = Random(problem_model)

    # acc_ucb_algo = ACCUCB(problem_model, v1, v2, N, rho, root_hypercube)
    bench_algo = Benchmark(problem_model)
    cc_mab_algo = CCMAB(problem_model, root_hypercube.get_dimension())
    aom_mc_algo = AOMMC(problem_model, v1, v2, N, rho, root_hypercube)

    print('Running AOM-MC...')
    aom_result = aom_mc_algo.run_algorithm()
    # acc_result = aom_result
    print('Running CC-MAB...')
    mab_result = cc_mab_algo.run_algorithm()
    print('Running benchmark...')
    bench_result = bench_algo.run_algorithm()
    acc_result = bench_result

    return {'aom_result': aom_result,
            'acc_result': acc_result,
            'mab_result': mab_result,
            'bench_result': bench_result}


if __name__ == '__main__':
    aom_round_refine_arr = []  # array of array. Inner ith array contains round in which a node was refined in ith run.
    aom_reward_runs_arr = np.zeros((num_times_to_run, len(num_rounds_arr)))
    mab_reward_runs_arr = np.zeros((num_times_to_run, len(num_rounds_arr)))
    acc_reward_runs_arr = np.zeros((num_times_to_run, len(num_rounds_arr)))
    bench_reward_runs_arr = np.zeros((num_times_to_run, len(num_rounds_arr)))

    aom_regret_runs_arr = np.zeros((num_times_to_run, len(num_rounds_arr)))
    mab_regret_runs_arr = np.zeros((num_times_to_run, len(num_rounds_arr)))
    acc_regret_runs_arr = np.zeros((num_times_to_run, len(num_rounds_arr)))
    if not use_saved_data:
        # problem_model = TestProblemModel(num_rounds, max(exp_num_workers), use_generated_workers_in_paper)
        if not use_generated_workers_in_paper:
            problem_model = GowallaProblemModel(max(num_rounds_arr), exp_num_workers, False)
        if num_threads_to_use == -1:
            num_threads_to_use = int(multiprocessing.cpu_count())
        print("Running on {thread_count} threads".format(thread_count=num_threads_to_use))
        for num_rounds in tqdm(num_rounds_arr):
            problem_model = GowallaProblemModel(num_rounds, exp_num_workers, True)
            print("Doing {uniform} T={num_rounds}...".format(uniform=is_uniform, num_rounds=num_rounds))
            parallel_results = Parallel(n_jobs=num_threads_to_use)(
                delayed(run_one_try)(problem_model) for _ in tqdm(range(num_times_to_run)))

            with open('parallel_results_T_{num_rounds}'.format(num_rounds=num_rounds), 'wb') as output:
                pickle.dump(parallel_results, output, pickle.HIGHEST_PROTOCOL)

    for j, num_rounds in enumerate(num_rounds_arr):
        with open('parallel_results_T_{num_rounds}'.format(num_rounds=num_rounds), 'rb') as input_file:
            parallel_results = pickle.load(input_file)

        # Load the ucb, mab, acc, and bench rewards and regrets
        for i, entry in enumerate(parallel_results):
            aom_result = entry['aom_result']
            acc_result = entry['acc_result']
            mab_result = entry['mab_result']
            bench_result = entry['bench_result']


            bench_reward_runs_arr[i, j] = pd.Series(bench_result['total_reward_arr']).expanding().mean().values[-1]
            aom_reward_runs_arr[i, j] = pd.Series(aom_result['total_reward_arr']).expanding().mean().values[-1] / \
                                     bench_reward_runs_arr[i, j]
            mab_reward_runs_arr[i, j] = pd.Series(mab_result['total_reward_arr']).expanding().mean().values[-1] / \
                                     bench_reward_runs_arr[i, j]
            acc_reward_runs_arr[i, j] = pd.Series(acc_result['total_reward_arr']).expanding().mean().values[-1] / \
                                     bench_reward_runs_arr[i, j]

            aom_regret_runs_arr[i, j] = np.cumsum(aom_result['regret_arr'])[-1]
            mab_regret_runs_arr[i, j] = np.cumsum(mab_result['regret_arr'])[-1]
            acc_regret_runs_arr[i, j] = np.cumsum(acc_result['regret_arr'])[-1]

    # Find the mean and std of the regrets and rewards
    bench_avg_reward = np.mean(bench_reward_runs_arr, axis=0)
    aom_avg_reward = np.mean(aom_reward_runs_arr, axis=0)
    mab_avg_reward = np.mean(mab_reward_runs_arr, axis=0)
    acc_avg_reward = np.mean(acc_reward_runs_arr, axis=0)

    aom_std_reward = np.std(aom_reward_runs_arr, axis=0)
    mab_std_reward = np.std(mab_reward_runs_arr, axis=0)
    acc_std_reward = np.std(acc_reward_runs_arr, axis=0)
    bench_std_reward = np.std(bench_reward_runs_arr, axis=0)

    aom_avg_regret = np.mean(aom_regret_runs_arr, axis=0)
    mab_avg_regret = np.mean(mab_regret_runs_arr, axis=0)
    acc_avg_regret = np.mean(acc_regret_runs_arr, axis=0)

    aom_std_regret = np.std(aom_regret_runs_arr, axis=0)
    mab_std_regret = np.std(mab_regret_runs_arr, axis=0)
    acc_std_regret = np.std(acc_regret_runs_arr, axis=0)

    # PLOT CUMULATIVE REGRET
    # Only show a few error bars
    for i in range(len(mab_std_regret)):
        if i == 0 or i % int(len(num_rounds_arr) / num_std_to_show) != 0 and i != len(mab_std_regret) - 1:
            aom_std_regret[i] = mab_std_regret[i] = acc_std_regret[i] = None

    plt.figure(1)
    plt.errorbar(num_rounds_arr, aom_avg_regret, yerr=aom_std_regret,
                 label="AOM-MC", capsize=2, color='r',
                 linestyle=line_style_dict, linewidth=2)
    plt.errorbar(num_rounds_arr, mab_avg_regret, yerr=mab_std_regret,
                 label="CC-MAB", capsize=2, color='g',
                 linestyle=line_style_dict, linewidth=2)
    plt.errorbar(num_rounds_arr, acc_avg_regret, yerr=acc_std_regret,
                 label="ACC-UCB", capsize=2, color='b',
                 linestyle=line_style_dict, linewidth=2)

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.legend()
    plt.xlabel("Arriving task $(t)$")
    plt.ylabel("Cumulative regret up to $t$")
    plt.tight_layout()
    plt.savefig("cum_regret.pdf", bbox_inches='tight', pad_inches=0.01)

    # PLOT AVERAGE REWARD
    # Only show a few error bars
    for i in range(len(mab_std_regret)):
        if i == 0 or i % int(len(num_rounds_arr) / num_std_to_show) != 0 and i != len(mab_std_regret) - 1:
            aom_std_reward[i] = mab_std_reward[i] = acc_std_reward[i] = bench_std_reward[i] = None
    plt.figure(2)
    plt.errorbar(num_rounds_arr, aom_avg_reward, yerr=aom_std_reward,
                 label="AOM-MC", capsize=2, color='r',
                 linestyle=line_style_dict, linewidth=2)
    # plt.errorbar(range(1, num_rounds + 1), bench_avg_reward, yerr=bench_std_reward,
    #              label="Bench", capsize=2, color='purple',
    #              linestyle=line_style_dict[budget], linewidth=2)
    plt.errorbar(num_rounds_arr, mab_avg_reward, yerr=mab_std_reward,
                 label="CC-MAB", capsize=2, color='g',
                 linestyle=line_style_dict, linewidth=2)
    plt.errorbar(num_rounds_arr, acc_avg_reward, yerr=acc_std_reward,
                 label="ACC-UCB", capsize=2, color='b',
                 linestyle=line_style_dict, linewidth=2)

    # plt.errorbar(range(1, num_rounds + 1), bench_avg_reward, yerr=bench_std_reward,
    #              label="Oracle", capsize=2, color='brown',
    #              linestyle=line_style_dict[budget], linewidth=2)

    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.legend()
    # plt.xlim(1000)
    # plt.ylim(0.60, 0.70)  # We need to do this b/c otherwise the legend was not seen
    plt.xlabel("Arriving task $(t)$")
    plt.ylabel("Average task reward divided\nby benchmark reward up to $t$")
    plt.tight_layout()
    plt.savefig("avg_reward.pdf", bbox_inches='tight', pad_inches=0.01)
    plt.show()
