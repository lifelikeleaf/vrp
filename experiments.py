import os
import random
import traceback
import time

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

logger = logger.getChild(__name__)

SOLOMON = 'Vrp-Set-Solomon' # n=100; 56 instances
HG = 'Vrp-Set-HG' # n=[200, 400, 600, 800, 1000]; 60 instances each


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


    def repeat(self, num_clusters):
        best_found_local = {}
        best_found_local['cost'] = float('inf')

        # repeat n times bc clustering algorithm may find diff clusters on each run
        for _ in range(0, self.repeat_n_times):
            self.decomp_runner.decomposer.num_clusters = num_clusters
            cost, routes = self.decomp_runner.run(in_parallel=True, num_workers=num_clusters)

            if cost < best_found_local['cost']:
                best_found_local['cost'] = cost
                best_found_local['routes'] = routes
                best_found_local['num_clusters'] = num_clusters

        return best_found_local


    def run_clusters_range(self):
        best_found = {}
        best_found['cost'] = float('inf')

        # try clustering with diff number of clusters
        min_clusters, max_clusters = self.num_clusters_range
        for num_clusters in range(min_clusters, max_clusters + 1):
            best_found_local = self.repeat(num_clusters)

            if best_found_local['cost'] < best_found['cost']:
                best_found['cost'] = best_found_local['cost']
                best_found['routes'] = best_found_local['routes']
                best_found['num_clusters'] = best_found_local['num_clusters']

        return best_found


    def get_decomp_best_found(self, experiment_name):
        cost_key = f'1_{experiment_name}_cost'
        num_routes_key = f'2_{experiment_name}_num_routes'
        num_clusters_key = f'3_{experiment_name}_num_clusters'

        best_found = self.run_clusters_range()

        sol_header = 'Solution - ' + experiment_name
        logger.info(f"--------------- {sol_header} ---------------")
        logger.info(f"Best decomp cost: {best_found['cost']} with "
                    f"{best_found['num_clusters']} clusters and {len(best_found['routes'])} routes")
        logger.info('')
        for i, route in enumerate(best_found['routes']):
            logger.debug(f"Route {i}: \n{route}")
        logger.info('')

        return {
            num_clusters_key: best_found['num_clusters'],
            num_routes_key: len(best_found['routes']),
            cost_key: best_found['cost'],
        }


    def run_experiments(self, inst):
        experiment_data = []
        for experiment in self.experiments:
            if self.decomp_runner is None:
                self.decomp_runner = DecompositionRunner(inst, experiment.decomposer, self.solver)
            else:
                # self.decomp_runner already exists, update its inst and
                # decomposer attributes, as they may have changed
                self.decomp_runner.inst = inst
                self.decomp_runner.decomposer = experiment.decomposer

            experiment_data.append(self.get_decomp_best_found(experiment.name))

            # let the CPU take a break after each experiment
            sec = 3
            logger.info(f'Sleeping for {sec} sec after each experiment')
            time.sleep(sec)
        
        return experiment_data


    def read_instance(self, dir_name, instance_name):
        logger.info('')
        logger.info(f'Benchmark instance name: {instance_name}')

        file_name = os.path.join('CVRPLIB', dir_name, instance_name)
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
        no_decomp_cost, no_decomp_routes = self.solver.solve(inst)
        logger.info(f'No decomp cost: {no_decomp_cost} with {len(no_decomp_routes)} routes')
        return no_decomp_cost, no_decomp_routes


    def run(self):
        for benchmark, size in self.benchmarks:
            dir_name = SOLOMON if size == 100 else HG

            for instance_name in benchmark:
                inst, bk_sol = self.read_instance(dir_name, instance_name)

                converted_inst = helpers.convert_cvrplib_to_vrp_instance(inst)

                no_decomp_cost, no_decomp_routes = self.get_no_decomp_solution(converted_inst)

                # run all the decomposition experiments on current VRP instance
                decomp_data = self.run_experiments(converted_inst)

                # prepare data to be written to excel
                excel_data = {
                    '0_instance_name': [instance_name],
                    '2_best_known_num_routes': [len(bk_sol.routes)],
                    '1_best_known_cost': [bk_sol.cost],
                    '2_no_decomp_num_routes': [len(no_decomp_routes)],
                    '1_no_decomp_cost': [no_decomp_cost],
                }

                for data in decomp_data:
                    excel_data.update(data)

                df = pd.DataFrame(excel_data)
                df = df.reindex(sorted(df.columns), axis=1)
                sheet_name = f'{dir_name}-{size}'
                helpers.write_to_excel(df, self.output_file_name, sheet_name)


if __name__ == "__main__":
    # args = helpers.get_args_parser(os.path.basename(__file__))


    def k_medoids():
        experiments = []
        prefix = '' #file_name
        experiments.append(Experiment(f'{prefix}_euclidean', KMedoidsDecomposer()))
        experiments.append(Experiment(f'{prefix}_TW', KMedoidsDecomposer(use_tw=True)))
        experiments.append(Experiment(f'{prefix}_TW_Neg', KMedoidsDecomposer(use_tw=True, allow_neg_dist=True)))
        experiments.append(Experiment(f'{prefix}_TW_Gap', KMedoidsDecomposer(use_tw=True, use_gap=True)))
        return experiments


    ### parameters for experiments
    sample_size = 2
    num_clusters_range = (6, 6) # inclusive
    repeat_n_times = 1
    instance_sizes = [600] # [100, 200, 400, 600, 800, 1000]
    time_limit = 5
    experiments = k_medoids
    file_name = experiments.__name__
    ### parameters for experiments


    benchmarks = []
    sample_benchmarks = []
    for size in instance_sizes:
        benchmark = cvrplib.list_names(low=size, high=size, vrp_type='vrptw')
        benchmarks.append((benchmark, size))
        sample = random.sample(benchmark, sample_size)
        sample_benchmarks.append((sample, size))

    # sample_benchmarks = [(['R1_6_1'], 600)]
    # sample_benchmarks = [(['C101'], 100)]

    solver = HgsSolverWrapper(time_limit)
    runner = ExperimentRunner(solver, sample_benchmarks, num_clusters_range, repeat_n_times, file_name)
    runner.add_experiements(experiments())

    try:
        runner.run()
    except Exception as err:
        tb_msg = traceback.format_exc()
        logger.error(tb_msg)
        raise

