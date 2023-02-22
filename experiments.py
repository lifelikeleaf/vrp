# Author: Xu Ye <kan.ye@tum.de>

import os
import random
import traceback

import cvrplib
import pandas as pd

from vrp.decomp.decomposition import DecompositionRunner
from vrp.decomp.decomposers import (
    KMeansDecomposer,
    KMedoidsDecomposer,
    APDecomposer
)
from vrp.decomp.solvers import HgsSolverWrapper
import vrp.decomp.helpers as helpers
from vrp.decomp.logger import logger
from vrp.decomp.constants import *
import vrp.decomp.distance_matrices as DM

logger = logger.getChild(__name__)


class ExperimentRunner:
    def __init__(self, solver, benchmarks, num_clusters_range, repeat_n_times, output_file_name, allow_sleep=False) -> None:
        self.benchmarks = benchmarks
        self.num_clusters_range = num_clusters_range
        self.repeat_n_times = repeat_n_times
        self.output_file_name = output_file_name
        self.experiments = []
        self.solver = solver
        self.decomp_runner = None
        self.allow_sleep = allow_sleep


    def add_experiement(self, experiment):
        self.experiments.append(experiment)


    def add_experiements(self, experiments):
        self.experiments.extend(experiments)


    def get_all_decomp(self, instance_name, experiment_name):
        '''write/log all results, not just best found'''

        logger.info('')
        exp_header = f'Running experiment: {experiment_name} on instance {instance_name}'
        logger.info(f"--------------- {exp_header} ---------------")

        # try clustering with diff number of clusters
        min_clusters, max_clusters = self.num_clusters_range
        for num_clusters in range(min_clusters, max_clusters + 1):
            # optionally repeat n times bc clustering algorithm may find diff clusters on each run
            # may no longer be needed after increasing clustering iterations
            for i in range(self.repeat_n_times):
                self.decomp_runner.decomposer.num_clusters = num_clusters
                solution = self.decomp_runner.run(in_parallel=True, num_workers=num_clusters)
                cost = solution.metrics[METRIC_COST]
                cost_wait = cost + solution.metrics[METRIC_WAIT_TIME]
                routes = solution.routes

                sol_header = f'Solution for experiment: {experiment_name} on instance {instance_name}'
                logger.info(f"------ {sol_header} ------")
                logger.info(f"Decomp cost: {cost} (include wait time cost: {cost_wait}) with "
                            f"{num_clusters} clusters and {len(routes)} routes; iteration {i}")
                logger.info('')

                # write KPIs to excel
                excel_data = {
                    KEY_INSTANCE_NAME: [instance_name],
                    KEY_ITERATION: [i],
                    KEY_NUM_SUBPROBS: [num_clusters],
                    KEY_NUM_ROUTES: [len(routes)],
                    KEY_COST: [cost],
                    KEY_COST_WAIT: [cost_wait],
                }
                df = pd.DataFrame(excel_data)
                helpers.write_to_excel(df, self.output_file_name, sheet_name=experiment_name)

                # write detailed routes to json
                ## routes_key = f'{instance_name}_{experiment_name}_{num_clusters}_{i}'
                json_data = {
                    KEY_INSTANCE_NAME: instance_name,
                    KEY_EXPERIMENT_NAME: experiment_name,
                    KEY_NUM_SUBPROBS: num_clusters,
                    KEY_ITERATION: i,
                    KEY_ROUTES: routes,
                }
                helpers.write_to_json(json_data, self.output_file_name)

                if self.allow_sleep:
                    # let the CPU take a break after each iteration (per repetition per cluster)
                    helpers.sleep(10, __name__)

        if self.allow_sleep:
            # break after each experiment
            helpers.sleep(60, __name__)


    def run_experiments(self, inst):
        for decomposer in self.experiments:
            self.decomp_runner = DecompositionRunner(inst, decomposer, self.solver)
            self.get_all_decomp(inst.extra['name'], decomposer.name)


    def read_instance(self, dir_name, instance_name):
        logger.info('')
        logger.info(f'Benchmark instance name: {instance_name}')

        file_name = os.path.join(CVRPLIB, dir_name, instance_name)
        inst, bk_sol = cvrplib.read(
            instance_path=f'{file_name}.txt',
            solution_path=f'{file_name}.sol'
        )
        min_tours = helpers.get_min_tours(inst.demands, inst.capacity)

        logger.info(f'Min num tours: {min_tours}')
        logger.info(f'Best known cost: {bk_sol.cost} with {len(bk_sol.routes)} routes')

        return inst, bk_sol


    def get_no_decomp_solution(self, inst):
        # call solver directly without decomposition
        logger.info('')
        solution = self.solver.solve(inst)
        cost = solution.metrics[METRIC_COST]
        cost_wait = cost + solution.metrics[METRIC_WAIT_TIME]
        logger.info(f'No decomp cost: {cost} (include wait time cost: {cost_wait}) with {len(solution.routes)} routes')
        json_data = {
            KEY_INSTANCE_NAME: inst.extra['name'],
            KEY_EXPERIMENT_NAME: 'No decomp',
            KEY_ROUTES: solution.routes,
        }
        helpers.write_to_json(json_data, self.output_file_name)
        return cost, cost_wait, solution.routes


    def run(self, experiments_only=False):
        for benchmark, benchmark_dir_name in self.benchmarks:
            for instance_name in benchmark:
                inst, bk_sol = self.read_instance(benchmark_dir_name, instance_name)
                converted_inst = helpers.convert_cvrplib_to_vrp_instance(inst)

                if not experiments_only:
                    no_decomp_cost, no_decomp_cost_wait, no_decomp_routes = self.get_no_decomp_solution(converted_inst)

                    # prepare data to be written to excel
                    excel_data = {
                        KEY_INSTANCE_NAME: [instance_name],
                        f'{KEY_NUM_ROUTES}_BK': [len(bk_sol.routes)],
                        f'{KEY_NUM_ROUTES}_NO_decomp': [len(no_decomp_routes)],
                        f'{KEY_COST}_BK': [bk_sol.cost],
                        f'{KEY_COST}_NO_decomp': [no_decomp_cost],
                        f'{KEY_COST_WAIT}_NO_decomp': [no_decomp_cost_wait],
                    }

                    # write base reference data to excel in its own tab
                    # subsequently each experiment will also write its output
                    # in its own tab - one tab per experiment, one row per instance
                    df = pd.DataFrame(excel_data)
                    ## df = df.reindex(sorted(df.columns), axis=1)
                    helpers.write_to_excel(df, self.output_file_name, sheet_name='Basis')

                # run all the decomposition experiments on current VRP instance
                self.run_experiments(converted_inst)


if __name__ == "__main__":
    # args = helpers.get_args_parser(os.path.basename(__file__))


    def sample_benchmarks(sample_size, instance_sizes):
        sample_benchmarks = []
        for size in instance_sizes:
            benchmark = cvrplib.list_names(low=size, high=size, vrp_type='vrptw')
            sample = random.sample(benchmark, sample_size)
            benchmark_dir_name = SOLOMON if size == 100 else HG
            sample_benchmarks.append((sample, benchmark_dir_name))

        return sample_benchmarks


    def k_medoids():
        # for each instance, run a set of experiments
        # each experiment is a diff way to decompose the instance
        # the best found solution is over a range of num_clusters and repeated n times
        experiments = []

        '''Euclidean'''
        experiments.append(KMedoidsDecomposer(DM.euclidean_vectorized, name='euclidean'))

        # '''temporal dist based'''
        # dist_matrix_func = DM.v1_vectorized
        # ext = dist_matrix_func.__name__.removesuffix('_vectorized')
        # experiments.append(KMedoidsDecomposer(dist_matrix_func, name=f'OL_{ext}', use_overlap=True, normalize=True))
        # experiments.append(KMedoidsDecomposer(dist_matrix_func, name=f'Gap_{ext}', use_gap=True, normalize=True))
        # experiments.append(KMedoidsDecomposer(dist_matrix_func, name=f'Both_{ext}', use_overlap=True, use_gap=True, normalize=True))

        '''temporal weight based'''
        # '''absolute overlap'''
        # '''v2.1 50/100%'''
        # dist_matrix_func = DM.v2_1_vectorized
        # ext = dist_matrix_func.__name__.removesuffix('_vectorized')
        # experiments.append(KMedoidsDecomposer(dist_matrix_func, name=f'OL_{ext}', use_overlap=True, normalize=True))
        # experiments.append(KMedoidsDecomposer(dist_matrix_func, name=f'Gap_{ext}', use_gap=True, normalize=True))
        # experiments.append(KMedoidsDecomposer(dist_matrix_func, name=f'Both_{ext}', use_overlap=True, use_gap=True, normalize=True))

        '''relative overlap'''
        # '''v2.2 100/100%'''
        # dist_matrix_func = DM.v2_2_vectorized
        # ext = dist_matrix_func.__name__.removesuffix('_vectorized')
        # experiments.append(KMedoidsDecomposer(dist_matrix_func, name=f'OL_{ext}', use_overlap=True, normalize=True))
        # experiments.append(KMedoidsDecomposer(dist_matrix_func, name=f'Gap_{ext}', use_gap=True, normalize=True))
        # experiments.append(KMedoidsDecomposer(dist_matrix_func, name=f'Both_{ext}', use_overlap=True, use_gap=True, normalize=True))

        # '''v2.3 50/50%'''
        # dist_matrix_func = DM.v2_3_vectorized
        # ext = dist_matrix_func.__name__.removesuffix('_vectorized')
        # experiments.append(KMedoidsDecomposer(dist_matrix_func, name=f'OL_{ext}', use_overlap=True, normalize=True))
        # experiments.append(KMedoidsDecomposer(dist_matrix_func, name=f'Gap_{ext}', use_gap=True, normalize=True))
        # experiments.append(KMedoidsDecomposer(dist_matrix_func, name=f'Both_{ext}', use_overlap=True, use_gap=True, normalize=True))

        # '''v2.4 30/30%'''
        # dist_matrix_func = DM.v2_4_vectorized
        # ext = dist_matrix_func.__name__.removesuffix('_vectorized')
        # experiments.append(KMedoidsDecomposer(dist_matrix_func, name=f'OL_{ext}', use_overlap=True, normalize=True))
        # experiments.append(KMedoidsDecomposer(dist_matrix_func, name=f'Gap_{ext}', use_gap=True, normalize=True))
        # experiments.append(KMedoidsDecomposer(dist_matrix_func, name=f'Both_{ext}', use_overlap=True, use_gap=True, normalize=True))

        '''v2.5 15/15%'''
        dist_matrix_func = DM.v2_5_vectorized
        ext = dist_matrix_func.__name__.removesuffix('_vectorized')
        experiments.append(KMedoidsDecomposer(dist_matrix_func, name=f'OL_{ext}', use_overlap=True, normalize=True))
        experiments.append(KMedoidsDecomposer(dist_matrix_func, name=f'Gap_{ext}', use_gap=True, normalize=True))

        # '''v2.6 50/30%'''
        # dist_matrix_func = DM.v2_6_vectorized
        # ext = dist_matrix_func.__name__.removesuffix('_vectorized')
        # experiments.append(KMedoidsDecomposer(dist_matrix_func, name=f'OL_{ext}', use_overlap=True, normalize=True))
        # experiments.append(KMedoidsDecomposer(dist_matrix_func, name=f'Gap_{ext}', use_gap=True, normalize=True))
        # experiments.append(KMedoidsDecomposer(dist_matrix_func, name=f'Both_{ext}', use_overlap=True, use_gap=True, normalize=True))

        # '''v2.7 40/40%'''
        # dist_matrix_func = DM.v2_7_vectorized
        # ext = dist_matrix_func.__name__.removesuffix('_vectorized')
        # experiments.append(KMedoidsDecomposer(dist_matrix_func, name=f'OL_{ext}', use_overlap=True, normalize=True))
        # experiments.append(KMedoidsDecomposer(dist_matrix_func, name=f'Gap_{ext}', use_gap=True, normalize=True))

        '''v2.8 20/20%'''
        dist_matrix_func = DM.v2_8_vectorized
        ext = dist_matrix_func.__name__.removesuffix('_vectorized')
        experiments.append(KMedoidsDecomposer(dist_matrix_func, name=f'OL_{ext}', use_overlap=True, normalize=True))
        experiments.append(KMedoidsDecomposer(dist_matrix_func, name=f'Gap_{ext}', use_gap=True, normalize=True))

        '''v2.9 10/10%'''
        dist_matrix_func = DM.v2_9_vectorized
        ext = dist_matrix_func.__name__.removesuffix('_vectorized')
        experiments.append(KMedoidsDecomposer(dist_matrix_func, name=f'OL_{ext}', use_overlap=True, normalize=True))
        experiments.append(KMedoidsDecomposer(dist_matrix_func, name=f'Gap_{ext}', use_gap=True, normalize=True))

        # '''v2.10 60/60%'''
        # dist_matrix_func = DM.v2_10_vectorized
        # ext = dist_matrix_func.__name__.removesuffix('_vectorized')
        # experiments.append(KMedoidsDecomposer(dist_matrix_func, name=f'OL_{ext}', use_overlap=True, normalize=True))
        # experiments.append(KMedoidsDecomposer(dist_matrix_func, name=f'Gap_{ext}', use_gap=True, normalize=True))

        return experiments


    '''parameters for experiments'''

    '''Basis'''
    experiments_only = False # True if only run experiments, don't run Basis

    allow_sleep = True
    num_clusters_range = (2, 6) # inclusive
    repeat_n_times = 1
    time_limit = 20 # <20 (5, 10, 15) even decomp can't find feasible solution for R1_10_1
    output_dir_name = 'E13'
    experiments = k_medoids
    benchmark_dir_name = HG
    input = {
        # SOLOMON
        # 'C1': C1,
        # 'C2': C2,
        # 'R1': R1,
        # 'R2': R2,
        # 'RC1': RC1,
        # 'RC2': RC2,

        # HG focus group
        # 'focus_C1': FOCUS_GROUP_C1,
        # 'focus_C2': FOCUS_GROUP_C2,
        # 'focus_R1': FOCUS_GROUP_R1,
        # 'focus_R2': FOCUS_GROUP_R2,
        # 'focus_RC1': FOCUS_GROUP_RC1,
        # 'focus_RC2': FOCUS_GROUP_RC2,

        # HG all
        # TODO: run one class at a time
        # 'C1': C1_10,
        # 'C2': C2_10,
        # 'R1': R1_10,
        # 'R2': R2_10,
        # 'RC1': RC1_10,
        # 'RC2': RC2_10,
    }

    # file_name = experiments.__name__ + '_test'
    # # Example instance returning no feasible solution: 'R1_6_1'

    # sample_size = 10
    # instance_sizes = [100, 200, 400, 600, 800, 1000]
    # benchmarks = sample_benchmarks(sample_size, instance_sizes)
    '''parameters for experiments'''


    solver = HgsSolverWrapper(time_limit, trivial_init_sol=False)
    for name, benchmark_class in input.items():
        helpers.make_dirs(output_dir_name)
        file_name = f'{output_dir_name}_' + experiments.__name__ + f'_{name}'
        file_name = os.path.join(output_dir_name, file_name)

        # e.g. [(['C101', 'C102', 'C103'], SOLOMON)]
        benchmarks = [(benchmark_class, benchmark_dir_name)]

        runner = ExperimentRunner(solver, benchmarks, num_clusters_range, repeat_n_times, file_name, allow_sleep=allow_sleep)
        runner.add_experiements(experiments())

        try:
            runner.run(experiments_only)
        except Exception as err:
            tb_msg = traceback.format_exc()
            logger.error(tb_msg)
            raise

