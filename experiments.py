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


def run_experiment(decomp_func, benchmarks, num_clusters_range, repeat_n_times, output_file_name):
    for benchmark, size in benchmarks:
        dir_name = SOLOMON if size == 100 else HG

        for instance_name in benchmark:
            logger.info(f'\nBenchmark instance name: {instance_name}')

            file_name = os.path.join('CVRPLIB', dir_name, instance_name)
            inst, bk_sol = cvrplib.read(
                instance_path=f'{file_name}.txt',
                solution_path=f'{file_name}.sol'
            )
            min_tours = helpers.get_min_tours(inst.demands, inst.capacity)
            logger.info(f'Min num tours: {min_tours}')
            logger.info(f'Best known cost: {bk_sol.cost}')
            logger.info(f'Best known num routes: {len(bk_sol.routes)}')

            inst = helpers.convert_cvrplib_to_vrp_instance(inst)
            no_decomp_cost, no_decomp_routes = no_decomp(inst)
            logger.info(f'No decomp cost: {no_decomp_cost}')
            logger.info(f'No decomp num routes: {len(no_decomp_routes)}')

            decomp = get_decomp_best_found(decomp_func, inst, num_clusters_range, repeat_n_times, include_tw=False)
            decomp_tw = get_decomp_best_found(decomp_func, inst, num_clusters_range, repeat_n_times, include_tw=True)

            # prepare data to be written to excel
            df = pd.DataFrame({
                'Instance name': [instance_name],
                'Best known num routes': [len(bk_sol.routes)],
                'No decomp num routes': [len(no_decomp_routes)],
                'Decomp num routes': [len(decomp['routes'])],
                'Decomp TW num routes': [len(decomp_tw['routes'])],
                'Best known cost': [bk_sol.cost],
                'No decomp cost': [no_decomp_cost],
                'Decomp cost': [decomp['cost']],
                'Decomp TW cost': [decomp_tw['cost']],
                'Decomp num clusters': [decomp['num_clusters']],
                'Decomp TW num clusters': [decomp_tw['num_clusters']],
            })
            sheet_name = f'{dir_name}-{size}'
            helpers.write_to_excel(df, output_file_name, sheet_name)


def no_decomp(inst):
    solver = HgsSolverWrapper()
    return solver.solve(inst)


def get_decomp_best_found(decomp_func, inst, num_clusters_range, repeat_n_times, include_tw):
    best_found = {}
    best_found['cost'] = float('inf')

    # try clustering with diff number of clusters
    min_clusters, max_clusters = num_clusters_range
    for num_clusters in range(min_clusters, max_clusters + 1):
        best_found_local = {}
        best_found_local['cost'] = float('inf')

        for _ in range(0, repeat_n_times):
            # repeat n times bc clustering algorithm could find diff clusters on each run
            cost, routes = decomp_func(inst, num_clusters, include_tw=include_tw)
            if cost < best_found_local['cost']:
                best_found_local['cost'] = cost
                best_found_local['routes'] = routes
                best_found_local['num_clusters'] = num_clusters

        if best_found_local['cost'] < best_found['cost']:
            best_found['cost'] = best_found_local['cost']
            best_found['routes'] = best_found_local['routes']
            best_found['num_clusters'] = best_found_local['num_clusters']

    sol_header = 'Solution'
    if include_tw:
        sol_header += ' TW'
    logger.info(f"----- {sol_header} -----")
    logger.info(f"Best decomp cost: {best_found['cost']} with "
                f"{best_found['num_clusters']} clusters and {len(best_found['routes'])} routes")
    for i, route in enumerate(best_found['routes']):
        logger.debug(f"Route {i}: \n{route}")

    return best_found


def k_medoids(inst, num_clusters, include_tw):
    # cluster sizes are better balanced than k-means;
    # better result with 3 clusters
    decomposer = KMedoidsDecomposer(
        inst,
        num_clusters,
        include_tw,
        # use_gap=True,
        # minimize_wait_time=True,
    )

    solver = HgsSolverWrapper()
    runner = DecompositionRunner(decomposer, solver)
    return runner.run(True, num_clusters)


def k_means(inst, num_clusters):
    # same result with 2 or 3 clusters
    decomposer = KMeansDecomposer(
        inst,
        num_clusters,
    )


def ap(inst, num_clusters):
    # had 9 clusters; TODO: control num of clusters?
    # set preference based on k-means++? merge clusters?
    decomposer = APDecomposer(
        inst,
        num_clusters,
        include_tw=True,
        # use_gap=True,
        # minimize_wait_time=True,
    )


if __name__ == "__main__":
    # args = helpers.get_args_parser(os.path.basename(__file__))

    # parameters for experiment
    sample_size = 5
    num_clusters_range = (2, 5)
    repeat_n_times = 2
    func = k_medoids
    file_name = func.__name__ # 'k_medoids'

    benchmarks = []
    instance_sizes = [600, 800, 1000] # [100, 200, 400, 600, 800, 1000]
    sample_benchmarks = []
    for size in instance_sizes:
        benchmark = cvrplib.list_names(low=size, high=size, vrp_type='vrptw')
        benchmarks.append((benchmark, size))
        sample = random.sample(benchmark, sample_size)
        sample_benchmarks.append((sample, size))

    # sample_benchmarks = [(['R1_6_1'], 600)]
    # sample_benchmarks = [(['C101'], 100)]

    try:
        run_experiment(func, sample_benchmarks, num_clusters_range, repeat_n_times, file_name)
    except Exception as err:
        tb_msg = traceback.format_exc()
        logger.error(tb_msg)
        raise

