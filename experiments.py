import os
import argparse
import time

import cvrplib

from vrp.decomp.decomposition import DecompositionRunner
from vrp.decomp.decomposers import KMeansDecomposer, KMedoidsDecomposer, \
    APDecomposer
from vrp.decomp.solvers import HgsSolverWrapper
import vrp.decomp.helpers as helpers

HG = 'Vrp-Set-HG' # n=[200, 1000]
SOLOMON = 'Vrp-Set-Solomon' # n=100


if __name__ == "__main__":
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


    benchmark = SOLOMON if args.benchmark == 1 else HG
    path = os.path.join('CVRPLIB', benchmark, args.instance_name)
    inst, sol = cvrplib.read(
        instance_path=f'{path}.txt',
        solution_path=f'{path}.sol'
    )

    inst = helpers.convert_cvrplib_to_vrp_instance(inst)

    # same result with 2 or 3 clusters
    # decomposer = KMeansDecomposer(
    #     inst,
    #     args.num_clusters,
    #     args.include_time_windows,
    # )

    # cluster sizes are better balanced than k-means;
    # better result with 3 clusters
    decomposer = KMedoidsDecomposer(
        inst,
        args.num_clusters,
        args.include_time_windows,
        # use_gap=True,
        # minimize_wait_time=True,
    )

    # had 9 clusters; TODO: control num of clusters?
    # set preference based on k-means++? merge clusters?
    # decomposer = APDecomposer(
    #     inst,
    #     args.num_clusters,
    #     args.include_time_windows,
    #     use_gap=True,
    #     # minimize_wait_time=True,
    # )

    solver = HgsSolverWrapper()
    runner = DecompositionRunner(decomposer, solver)

    start = time.time()
    total_cost, total_routes = runner.run(True, args.num_clusters)
    end = time.time()
    print('Run time:', end-start)

    print("\n----- Solution -----")
    print("Total cost: ", total_cost)
    for i, route in enumerate(total_routes):
        print(f"Route {i}:", route)
