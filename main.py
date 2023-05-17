# Author: Xu Ye <kan.ye@tum.de>

"Main entry point for configuring experiments."

import os
import random
import traceback

import vrp.third_party.cvrplib as cvrplib

from vrp.decomp.decomposers import KMedoidsDecomposer
from vrp.decomp.solvers import GortoolsSolverWrapper
import vrp.decomp.helpers as helpers
from vrp.decomp.logger import logger
from vrp.decomp.constants import *
import vrp.decomp.distance_matrices as DM
from experiments import ExperimentRunner


if __name__ == "__main__":

    def sample_benchmarks(sample_size, instance_sizes):
        sample_benchmarks = []
        for size in instance_sizes:
            benchmark = cvrplib.list_names(low=size, high=size, vrp_type='vrptw')
            sample = random.sample(benchmark, sample_size)
            benchmark_dir_name = SOLOMON if size == 100 else HG
            sample_benchmarks.append((sample, benchmark_dir_name))

        return sample_benchmarks


    def get_experiments(experiments_only):
        def k_medoids():
            # for each instance, run a set of experiments
            # each experiment is a diff way to decompose the instance
            experiments = []

            '''Euclidean'''
            if not experiments_only:
                experiments.append(KMedoidsDecomposer(DM.euclidean_vectorized, name='euclidean'))

            '''Edit 1/2'''
            '''MODIFY: add experiments to run'''

            '''Qi et al. 2012'''
            dist_matrix_func = DM.get_qi_2012_vectorized()
            experiments.append(KMedoidsDecomposer(dist_matrix_func, name=f'qi_2012'))

            '''TSD: Temporal Spatial Distance'''
            dist_matrix_func = DM.get_dist_matrix_func_v2_2()
            ext = dist_matrix_func.__name__.removesuffix('_vectorized')
            experiments.append(KMedoidsDecomposer(dist_matrix_func, name=f'Both_{ext}', use_overlap=True, use_gap=True, normalize=True))

            '''END MODIFY: add experiments to run'''

            return experiments

        return k_medoids


    '''Edit 2/2'''
    '''MODIFY: parameters for experiments'''
    min_total = True # Default: True. (True if OF = min total time, else OF = min driving time)
    num_clusters_range = (10, 10) # inclusive
    repeat_n_times = 1 # number of replications for an experiment
    time_limit = 10 # solver time limt; use >=20 if using HGS bc <20 even decomp can't find feasible solution for R1_10_1
    output_dir_name = 'E_Test_Main'
    sleep_time = 15 # time (in seconds) to sleep b/t experiments
    experiments_only = False # Default: False. (True if only run experiments and not Basis/No decomp/Euclidean)

    input = {
        # 'Test': ['C1_10_1'],

        # Homberger and Gehring 1999 (HG) all 1k-node instances
        'C1': C1_10,
        'C2': C2_10,
        'R1': R1_10,
        'R2': R2_10,
        'RC1': RC1_10,
        'RC2': RC2_10,
    }

    benchmark_dir_name = HG
    experiments = get_experiments(experiments_only)

    # sample_size = 10
    # instance_sizes = [100, 200, 400, 600, 800, 1000]
    # benchmarks = sample_benchmarks(sample_size, instance_sizes)
    '''END MODIFY: parameters for experiments'''


    solver = GortoolsSolverWrapper(time_limit, min_total=min_total)

    for name, benchmark_class in input.items():
        helpers.make_dirs(output_dir_name)
        file_name = f'{output_dir_name}_' + experiments.__name__ + f'_{name}'
        file_name = os.path.join(output_dir_name, file_name)

        # e.g. [(['C101', 'C102', 'C103'], SOLOMON)]
        benchmarks = [(benchmark_class, benchmark_dir_name)]

        runner = ExperimentRunner(solver, benchmarks, num_clusters_range, repeat_n_times, file_name, sleep_time=sleep_time)
        runner.add_experiements(experiments())

        try:
            runner.run(experiments_only)
        except Exception as err:
            tb_msg = traceback.format_exc()
            logger.error(tb_msg)
            raise

