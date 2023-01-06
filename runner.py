import os
import argparse
import time

from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation as AP
from sklearn.metrics import pairwise_distances
from sklearn_extra.cluster import KMedoids

from hgs.baselines.hgs_vrptw import hgspy
from wurlitzer import pipes
import cvrplib
import numpy as np
from scipy.spatial.distance import euclidean

from decomposition import AbstractSolverWrapper, AbstractDecomposer, \
    DecompositionRunner, Node, VRPInstance
import helpers

HG = 'Vrp-Set-HG' # n=[200, 1000]
SOLOMON = 'Vrp-Set-Solomon' # n=100


class BaseDecomposer(AbstractDecomposer):
    """Abstract class that implements some common methods used by
    all decomposers implemented in this module.
    """
    def __init__(self, inst, num_clusters, include_tw=False) -> None:
        """

        Parameters
        ----------
        inst: `VRPInstance`
            A VRP problem instance.
        num_clusters: int
            Number of clusters to decompose the VRP problem instance into.
        include_tw: bool
            True if time windows should be included in features, else False.
            Default is False.

        """
        super().__init__(inst)
        self.num_clusters = num_clusters
        self.include_tw = include_tw
        # a list of feature vectors representing the customers
        # to be clustered, excluding the depot
        self.fv = self.build_feature_vectors()


    def build_feature_vectors(self):
        """Build feature vectors for clustering from VRP problem instance."""
        fv = []
        nodes = self.inst.nodes
        for i in range(len(nodes)):
            row = []
            # x, y coords for customer i
            row.append(nodes[i].x_coord)
            row.append(nodes[i].y_coord)
            if self.include_tw:
                # earliest service start time for customer i
                row.append(nodes[i].start_time)
                # lastest service start time for customer i
                row.append(nodes[i].end_time)
            fv.append(row)

        # By CVRPLIB convention, index 0 is always depot;
        # depot should not be clustered
        return np.array(fv[1:])


    def get_clusters(self, labels):
        # array index in labels are customer IDs,
        # value at a given index is the cluster ID.
        clusters = [[] for i in range(self.num_clusters)]
        for i in range(len(labels)):
            # customer id is shifted by 1 bc index 0 is depot;
            # and depot is not clustered
            clusters[labels[i]].append(i + 1)

        return clusters


class KMeansDecomposer(BaseDecomposer):
    def decompose(self):
        # Run the k-means algorithm.
        # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

        print('Running k-means...')
        kmeans = KMeans(n_clusters=self.num_clusters, n_init=10).fit(self.fv)
        labels = kmeans.labels_
        return self.get_clusters(labels)


class HgsSolverWrapper(AbstractSolverWrapper):
    def __init__(self, time_limit=10, cpp_output=False) -> None:
        self.time_limit = time_limit
        self.cpp_output = cpp_output


    def build_instance_for_hgs(self, inst: VRPInstance):
        """Converts a `VRPInstance` to argument types
        accepted by `hgspy.Params`:

        hgspy.Params(
            config: hgspy.Config,
            coords: List[Tuple[int, int]],
            demands: List[int],
            vehicle_cap: int,
            time_windows: List[Tuple[int, int]],
            service_durations: List[int],
            duration_matrix: List[List[int]],
            release_times: List[int]
        )
        """
        coords = []
        demands = []
        time_windows = []
        service_durations = []
        duration_matrix = []
        for i in range(len(inst.nodes)):
            node = inst.nodes[i]
            coords.append((node.x_coord, node.y_coord))
            demands.append(node.demand)
            time_windows.append((node.start_time, node.end_time))
            service_durations.append(node.service_time)
            duration_matrix.append(node.distances)

        return dict(
            coords = coords,
            demands = demands,
            vehicle_cap = inst.vehicle_capacity,
            time_windows = time_windows,
            service_durations = service_durations,
            duration_matrix = duration_matrix,
            # not used but required by hgspy.Params
            release_times=[0] * len(inst.nodes),
        )


    def solve(self, inst: VRPInstance):
        # Calls the HGS solver with default config and passing in a
        # VRP problem instance.

        instance = self.build_instance_for_hgs(inst)

        # Capture C-level stdout/stderr
        with pipes() as (out, err):
            config = hgspy.Config(
                nbVeh=-1,
                timeLimit=self.time_limit,
                useWallClockTime=True
            )

            params = hgspy.Params(config, **instance)
            split = hgspy.Split(params)
            ls = hgspy.LocalSearch(params)
            pop = hgspy.Population(params, split, ls)
            algo = hgspy.Genetic(params, split, pop, ls)
            algo.run()
            # get the best found solution (type Individual) from
            # the population pool
            solution = pop.getBestFound()

        if self.cpp_output:
            print(f'Output from C++: \n {out.read()}')

        # return solution
        # for some reason returning solution alone makes solution.routes = []
        return solution.cost, solution.routes


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
    path = f'CVRPLIB/{benchmark}/{args.instance_name}'
    inst, sol = cvrplib.read(
        instance_path=f'{path}.txt',
        solution_path=f'{path}.sol'
    )

    inst = helpers.convert_cvrplib_to_vrp_instance(inst)
    decomposer = KMeansDecomposer(inst, args.num_clusters, args.include_time_windows)
    solver = HgsSolverWrapper()
    runner = DecompositionRunner(decomposer, solver, parallel_run_solver=True)

    start = time.time()
    total_cost, total_routes = runner.run()
    end = time.time()
    print('Run time:', end-start)

    print("\n----- Solution -----")
    print("Total cost: ", total_cost)
    for i, route in enumerate(total_routes):
        print(f"Route {i}:", route)
