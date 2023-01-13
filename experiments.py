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

            # let the CPU take a break
            helpers.sleep(3, __name__)

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


    def get_decomp_best_found(self, instance_name, experiment_name):
        cost_key = f'cost_{experiment_name}'
        num_routes_key = f'num_routes_{experiment_name}'
        num_subprobs_key = f'num_subprobs_{experiment_name}'

        best_found = self.run_clusters_range()

        sol_header = 'Solution - ' + experiment_name
        logger.info(f"--------------- {sol_header} ---------------")
        logger.info(f"Best decomp cost: {best_found['cost']} with "
                    f"{best_found['num_clusters']} clusters and {len(best_found['routes'])} routes")
        logger.info('')
        for i, route in enumerate(best_found['routes']):
            logger.debug(f"Route {i}: \n{route}")
        logger.info('')

        excel_data = {
            'instance_name': [instance_name],
            num_subprobs_key: best_found['num_clusters'],
            num_routes_key: len(best_found['routes']),
            cost_key: best_found['cost'],
        }
        df = pd.DataFrame(excel_data)
        helpers.write_to_excel(df, self.output_file_name, experiment_name)

        return excel_data


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

            experiment_data.append(self.get_decomp_best_found(inst.extra['name'], experiment.name))

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

                # prepare data to be written to excel
                excel_data = {
                    'instance_name': [instance_name],
                    'num_routes_BK': [len(bk_sol.routes)],
                    'num_routes_NO_decomp': [len(no_decomp_routes)],
                    'cost_BK': [bk_sol.cost],
                    'cost_NO_decomp': [no_decomp_cost],
                }

                # for data in decomp_data:
                #     excel_data.update(data)

                # write base reference data to excel in its own tab
                # subsequently each experiment will also write its output
                # in its own tab - one tab per experiment, one row per instance
                df = pd.DataFrame(excel_data)
                # df = df.reindex(sorted(df.columns), axis=1)
                # sheet_name = f'{dir_name}-{size}'
                helpers.write_to_excel(df, self.output_file_name, 'Basis')

                # run all the decomposition experiments on current VRP instance
                decomp_data = self.run_experiments(converted_inst)


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
        # gap by default is negative
        experiments.append(Experiment('TW_Gap', KMedoidsDecomposer(use_tw=True, use_gap=True)))
        # what if gap is always positive by turning on min wait time, even though wait time is not yet included in OF
        experiments.append(Experiment('TW_Pos_Gap', KMedoidsDecomposer(use_tw=True, use_gap=True, minimize_wait_time=True)))

        # normalized version of above
        experiments.append(Experiment('euclidean_norm', KMedoidsDecomposer(normalize=True)))
        experiments.append(Experiment('TW_norm', KMedoidsDecomposer(use_tw=True, normalize=True)))
        experiments.append(Experiment('TW_Gap_norm', KMedoidsDecomposer(use_tw=True, use_gap=True, normalize=True)))
        experiments.append(Experiment('TW_Pos_Gap_norm', KMedoidsDecomposer(use_tw=True, use_gap=True, minimize_wait_time=True, normalize=True)))
        return experiments


    '''parameters for experiments'''
    num_clusters_range = (2, 5) # inclusive
    repeat_n_times = 1
    time_limit = 10
    experiments = k_medoids
    file_name = experiments.__name__

    # sample_size = 10
    # instance_sizes = [100, 200, 400, 600, 800, 1000]

    # benchmarks = [(['C101'], 100)]
    # benchmarks = sample_benchmarks(sample_size, instance_sizes)
    benchmarks = [(C1_10, 1000)]
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

