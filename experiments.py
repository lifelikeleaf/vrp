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
                            f"{num_clusters} clusters and {len(routes)} routes")
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


    def run(self):
        for benchmark, size in self.benchmarks:
            dir_name = SOLOMON if size == 100 else HG

            for instance_name in benchmark:
                inst, bk_sol = self.read_instance(dir_name, instance_name)
                converted_inst = helpers.convert_cvrplib_to_vrp_instance(inst)
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
                ## sheet_name = f'{dir_name}-{size}'
                helpers.write_to_excel(df, self.output_file_name, sheet_name='Basis')

                # run all the decomposition experiments on current VRP instance
                self.run_experiments(converted_inst)


if __name__ == "__main__":
    # args = helpers.get_args_parser(os.path.basename(__file__))


    def sample_benchmarks(sample_size, instance_sizes):
        # benchmarks = []
        sample_benchmarks = []
        for size in instance_sizes:
            benchmark = cvrplib.list_names(low=size, high=size, vrp_type='vrptw')
            # benchmarks.append((benchmark, size))
            sample = random.sample(benchmark, sample_size)
            sample_benchmarks.append((sample, size))

        return sample_benchmarks


    # run on a deterministic set of instances rather than a random sample
    # so that new experiments can be compared to old ones w/o rerunning old ones
    # focus on the 1k nodes benchmark where decomp is important

    # But use Solomon 100-node benchmark first for faster experiments
    '''Solomon 100-node benchmark first'''
    C1 = ['C101', 'C102', 'C103', 'C104', 'C105', 'C106', 'C107', 'C108', 'C109']
    C2 = ['C201', 'C202', 'C203', 'C204', 'C205', 'C206', 'C207', 'C208']
    R1 = ['R101', 'R102', 'R103', 'R104', 'R105', 'R106', 'R107', 'R108', 'R109', 'R110', 'R111', 'R112']
    R2 = ['R201', 'R202', 'R203', 'R204', 'R205', 'R206', 'R207', 'R208', 'R209', 'R210', 'R211']
    RC1 = ['RC101', 'RC102', 'RC103', 'RC104', 'RC105', 'RC106', 'RC107', 'RC108']
    RC2 = ['RC201', 'RC202', 'RC203', 'RC204', 'RC205', 'RC206', 'RC207', 'RC208']

    '''HG 1k-node benchmark'''
    ## geographically clustered
    ## narrow TWs
    C1_10 = ['C1_10_1', 'C1_10_2', 'C1_10_3', 'C1_10_4', 'C1_10_5', 'C1_10_6', 'C1_10_7', 'C1_10_8', 'C1_10_9', 'C1_10_10']
    ## wide TWs
    C2_10 = ['C2_10_1', 'C2_10_2', 'C2_10_3', 'C2_10_4', 'C2_10_5', 'C2_10_6', 'C2_10_7', 'C2_10_8', 'C2_10_9', 'C2_10_10']

    ## randomly distributed
    R1_10 = ['R1_10_1', 'R1_10_2', 'R1_10_3', 'R1_10_4', 'R1_10_5', 'R1_10_6', 'R1_10_7', 'R1_10_8', 'R1_10_9', 'R1_10_10']
    R2_10 = ['R2_10_1', 'R2_10_2', 'R2_10_3', 'R2_10_4', 'R2_10_5', 'R2_10_6', 'R2_10_7', 'R2_10_8', 'R2_10_9', 'R2_10_10']

    ## mixed
    RC1_10 = ['RC1_10_1', 'RC1_10_2', 'RC1_10_3', 'RC1_10_4', 'RC1_10_5', 'RC1_10_6', 'RC1_10_7', 'RC1_10_8', 'RC1_10_9', 'RC1_10_10']
    RC2_10 = ['RC2_10_1', 'RC2_10_2', 'RC2_10_3', 'RC2_10_4', 'RC2_10_5', 'RC2_10_6', 'RC2_10_7', 'RC2_10_8', 'RC2_10_9', 'RC2_10_10']


    def k_medoids():
        # for each instance, run a set of experiments
        # each experiment is a diff way to decompose the instance
        # the best found solution is over a range of num_clusters and repeated n times
        experiments = []
        experiments.append(Experiment('euclidean', KMedoidsDecomposer()))
        experiments.append(Experiment('TW', KMedoidsDecomposer(use_tw=True)))
        # # gap by default is negative (gap < 0)
        # experiments.append(Experiment('TW_Gap', KMedoidsDecomposer(use_tw=True, use_gap=True)))
        # # currently no wait time in OF value, only wait time added post-routing
        # # so this essentially makes all gap a penalty (gap > 0)
        # experiments.append(Experiment('TW_Pos_Gap', KMedoidsDecomposer(use_tw=True, use_gap=True, minimize_wait_time=True)))

        # # normalized version of above
        # experiments.append(Experiment('euclidean_norm', KMedoidsDecomposer(normalize=True)))
        # experiments.append(Experiment('TW_norm', KMedoidsDecomposer(use_tw=True, normalize=True)))
        # experiments.append(Experiment('TW_Gap_norm', KMedoidsDecomposer(use_tw=True, use_gap=True, normalize=True)))
        # experiments.append(Experiment('TW_Pos_Gap_norm', KMedoidsDecomposer(use_tw=True, use_gap=True, minimize_wait_time=True, normalize=True)))
        return experiments


    '''parameters for experiments'''
    num_clusters_range = (3, 3) # inclusive
    repeat_n_times = 1
    time_limit = 5
    experiments = k_medoids
    file_name = experiments.__name__ + '_test'
    # benchmarks = [(RC2, 100)]

    # sample_size = 10
    # instance_sizes = [100, 200, 400, 600, 800, 1000]

    benchmarks = [(['C101'], 100)]
    # Example instance returning no feasible solution:
    # benchmarks = [(['R1_6_1'], 600)]
    # benchmarks = sample_benchmarks(sample_size, instance_sizes)

    '''parameters for experiments'''


    solver = HgsSolverWrapper(time_limit)
    runner = ExperimentRunner(solver, benchmarks, num_clusters_range, repeat_n_times, file_name)
    runner.add_experiements(experiments())

    try:
        runner.run()
    except Exception as err:
        tb_msg = traceback.format_exc()
        logger.error(tb_msg)
        raise

