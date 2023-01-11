import os
import argparse
import random

import cvrplib
import pandas as pd

from vrp.decomp.decomposition import DecompositionRunner
from vrp.decomp.decomposers import KMeansDecomposer, KMedoidsDecomposer, \
    APDecomposer
from vrp.decomp.solvers import HgsSolverWrapper
import vrp.decomp.helpers as helpers
from vrp.decomp.logger import logger

logger = logger.getChild(__name__)

SOLOMON = 'Vrp-Set-Solomon' # n=100; 56 instances
HG = 'Vrp-Set-HG' # n=[200, 400, 600, 800, 1000]; 60 instances each


def write_to_excel(df: pd.DataFrame, sheet_name, replace=True):
    file_name = 'output.xlsx'
    if_sheet_exists = 'overlay' if not replace else 'replace'
    try:
        with pd.ExcelWriter(file_name, mode='a', if_sheet_exists=if_sheet_exists) as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    except FileNotFoundError:
        with pd.ExcelWriter(file_name, mode='x') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)


def run_experiment(benchmarks, num_clusters_range, repeat_n_times):
    for benchmark, size in benchmarks:
        # prepare data to be written to excel
        instance_names = benchmark

        bk_costs = []
        bk_num_routes = []
        
        no_decomp_costs = []
        no_decomp_num_routes = []
        
        best_found_costs = []
        best_found_num_routes = []
        best_found_num_clusters = []

        dir_name = SOLOMON if size == 100 else HG

        for instance_name in benchmark:
            logger.info(f'Benchmark instance name: {instance_name}')

            file_name = os.path.join('CVRPLIB', dir_name, instance_name)
            inst, bk_sol = cvrplib.read(
                instance_path=f'{file_name}.txt',
                solution_path=f'{file_name}.sol'
            )
            inst = helpers.convert_cvrplib_to_vrp_instance(inst)

            logger.info(f'Best known cost: {bk_sol.cost}')
            logger.info(f'Best known num routes: {len(bk_sol.routes)}')
            bk_costs.append(bk_sol.cost)
            bk_num_routes.append(len(bk_sol.routes))

            no_decomp_cost, no_decomp_routes = no_decomp(inst)
            logger.info(f'No decomp cost: {no_decomp_cost}')
            logger.info(f'No decomp num routes: {len(no_decomp_routes)}')
            no_decomp_costs.append(no_decomp_cost)
            no_decomp_num_routes.append(len(no_decomp_routes))

            best_found = {}
            best_found['cost'] = float('inf')

            # try clustering with diff number of clusters
            min_clusters, max_clusters = num_clusters_range
            for num_clusters in range(min_clusters, max_clusters + 1):
                best_found_local = {}
                best_found_local['cost'] = float('inf')

                for _ in range(0, repeat_n_times):
                    # repeat n times bc k-medoids could find diff clusters on each run
                    cost, routes = k_medoids(inst, num_clusters, include_tw=True)
                    if cost < best_found_local['cost']:
                        best_found_local['cost'] = cost
                        best_found_local['routes'] = routes
                        best_found_local['num_clusters'] = num_clusters

                if best_found_local['cost'] < best_found['cost']:
                    best_found['cost'] = best_found_local['cost']
                    best_found['routes'] = best_found_local['routes']
                    best_found['num_clusters'] = best_found_local['num_clusters']

            best_found_costs.append(best_found['cost'])
            best_found_num_routes.append(len(best_found['routes']))
            best_found_num_clusters.append(best_found['num_clusters'])
            logger.info("----- Solution -----")
            logger.info(f"Best decomp cost: {best_found['cost']} with "
                        f"{best_found['num_clusters']} clusters and {len(best_found['routes'])} routes")
            for i, route in enumerate(best_found['routes']):
                logger.info(f"Route {i}: \n{route}")

        df = pd.DataFrame({
            'Instance name': instance_names,
            'Best known cost': bk_costs,
            'No decomp cost': no_decomp_costs,
            'Best decomp cost': best_found_costs,
            'Number of clusters': best_found_num_clusters,
            'Best known num routes': bk_num_routes,
            'No decomp num routes': no_decomp_num_routes,
            'Best decomp num routes': best_found_num_routes,
        })
        sheet_name = f'{dir_name}-{size}'
        write_to_excel(df, sheet_name)


def no_decomp(inst):
    solver = HgsSolverWrapper()
    return solver.solve(inst)


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


def k_means():
    # same result with 2 or 3 clusters
    decomposer = KMeansDecomposer(
        inst,
        args.num_clusters,
        args.include_time_windows,
    )


def ap():
    # had 9 clusters; TODO: control num of clusters?
    # set preference based on k-means++? merge clusters?
    decomposer = APDecomposer(
        inst,
        args.num_clusters,
        args.include_time_windows,
        # use_gap=True,
        # minimize_wait_time=True,
    )


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="Example usages: "
            f"python {os.path.basename(__file__)} -b=1 -n='C206' -k=3 -t | "
            f"python {os.path.basename(__file__)} -b=2 -n='C1_2_1' -k=3 -t"
    )

    parser.add_argument(
        '-n', '--instance_name', required=True,
        help='benchmark instance name without file extension, e.g. "C206"'
    )
    parser.add_argument(
        '-b', '--benchmark', default=1, choices=[1, 2], type=int,
        help='benchmark dataset to use: 1=Solomon (1987), '
            '2=Homberger and Gehring (1999); Default=1'
    )
    parser.add_argument('-k', '--num_clusters', required=True, type=int,
                        help='number of clusters')
    parser.add_argument('-t', '--include_time_windows', action='store_true',
                        help='use time windows for clustering or not')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # args = get_args_parser()

    # parameters for experiment
    sample_size = 5
    num_clusters_range = (2, 5)
    repeat_n_times = 3

    benchmarks = [(cvrplib.list_names(low=100, high=100, vrp_type='vrptw'), 100)]
    for instance_size in range(200, 1001, 200):
        benchmarks.append((cvrplib.list_names(low=instance_size, high=instance_size, vrp_type='vrptw'), instance_size))

    sample_benchmarks = []
    for benchmark, size in benchmarks:
        sample = random.sample(benchmark, sample_size)
        sample_benchmarks.append((sample, size))

    run_experiment(sample_benchmarks, num_clusters_range, repeat_n_times)
