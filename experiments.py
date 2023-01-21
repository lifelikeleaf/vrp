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

logger = logger.getChild(__name__)


class Experiment:
    def __init__(self, name, decomposer) -> None:
        self.name = name
        self.decomposer = decomposer


class ExperimentRunner:
    def __init__(self, solver, benchmarks, num_clusters_range, repeat_n_times, output_file_name) -> None:
        self.benchmarks = benchmarks
        self.num_clusters_range = num_clusters_range
        self.repeat_n_times = repeat_n_times
        self.output_file_name = output_file_name
        self.experiments: list[Experiment] = []
        self.solver = solver
        self.decomp_runner = None


    def add_experiement(self, experiment: Experiment):
        self.experiments.append(experiment)


    def add_experiements(self, experiments: list[Experiment]):
        self.experiments.extend(experiments)


    def get_all_decomp(self, instance_name, experiment_name):
        '''write/log all results, not just best found'''

        logger.info('')
        exp_header = f'Running experiment: {experiment_name} on instance {instance_name}'
        logger.info(f"--------------- {exp_header} ---------------")

        # try clustering with diff number of clusters
        min_clusters, max_clusters = self.num_clusters_range
        for num_clusters in range(min_clusters, max_clusters + 1):
            # repeat n times bc clustering algorithm may find diff clusters on each run
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

                # let the CPU take a break
                helpers.sleep(3, __name__)


    def run_experiments(self, inst):
        for experiment in self.experiments:
            if self.decomp_runner is None:
                self.decomp_runner = DecompositionRunner(inst, experiment.decomposer, self.solver)
            else:
                # self.decomp_runner already exists, update its inst and
                # decomposer attributes, as they may have changed
                self.decomp_runner.inst = inst
                self.decomp_runner.decomposer = experiment.decomposer

            self.get_all_decomp(inst.extra['name'], experiment.name)


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
        experiments.append(Experiment('euclidean', KMedoidsDecomposer()))
        experiments.append(Experiment('TW', KMedoidsDecomposer(use_tw=True)))
        # gap by default is negative (gap < 0)
        experiments.append(Experiment('TW_Gap', KMedoidsDecomposer(use_tw=True, use_gap=True)))
        # # currently no wait time in OF value, only wait time added post-routing
        # # so this essentially makes all gap a penalty (gap > 0)
        # experiments.append(Experiment('TW_Pos_Gap', KMedoidsDecomposer(use_tw=True, use_gap=True, minimize_wait_time=True)))

        # # normalized version of above
        experiments.append(Experiment('euclidean_norm', KMedoidsDecomposer(normalize=True)))
        experiments.append(Experiment('TW_norm', KMedoidsDecomposer(use_tw=True, normalize=True)))
        experiments.append(Experiment('TW_Gap_norm', KMedoidsDecomposer(use_tw=True, use_gap=True, normalize=True)))
        # experiments.append(Experiment('TW_Pos_Gap_norm', KMedoidsDecomposer(use_tw=True, use_gap=True, minimize_wait_time=True, normalize=True)))
        return experiments


    '''parameters for experiments'''
    num_clusters_range = (2, 5) # inclusive
    repeat_n_times = 3
    time_limit = 10
    experiments = k_medoids
    experiments_only = False # only run experiments, don't run Basis
    # input = {'C1': C1, 'C2': C2, 'R1': R1, 'R2': R2, 'RC1': RC1, 'RC2': RC2}
    # input = {'focus_C1': FOCUS_GROUP_C1, 'focus_RC2': FOCUS_GROUP_RC2}
    input = {'RC2': RC2}
    benchmark_dir_name = SOLOMON

    # file_name = experiments.__name__ + '_test'
    # # Example instance returning no feasible solution: 'R1_6_1'

    # sample_size = 10
    # instance_sizes = [100, 200, 400, 600, 800, 1000]
    # benchmarks = sample_benchmarks(sample_size, instance_sizes)
    '''parameters for experiments'''


    solver = HgsSolverWrapper(time_limit)
    for name, benchmark in input.items():
        file_name = experiments.__name__ + f'_{name}'
        # e.g. [(['C101', 'C102', 'C103'], SOLOMON)]
        benchmarks = [(benchmark, benchmark_dir_name)]

        runner = ExperimentRunner(solver, benchmarks, num_clusters_range, repeat_n_times, file_name)
        runner.add_experiements(experiments())

        try:
            runner.run(experiments_only)
        except Exception as err:
            tb_msg = traceback.format_exc()
            logger.error(tb_msg)
            raise

