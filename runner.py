import os
import argparse
import time
from multiprocessing import Pool
import pprint as p

from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation as AP
from sklearn.metrics import pairwise_distances
from sklearn_extra.cluster import KMedoids
import kmedoids as fast_kmedoids

import cvrplib
import numpy as np
from scipy.spatial.distance import euclidean

from decomposition import AbstractSolverWrapper, AbstractDecomposer, \
    DecompositionRunner, Node, VRPInstance


HG = 'Vrp-Set-HG' # n=[200, 1000]
SOLOMON = 'Vrp-Set-Solomon' # n=100


def convert_to_vrp_instance(benchmark) -> VRPInstance:
    node_list = []
    for customer_id in range(len(benchmark.coordinates)):
        params = dict(
            x_coord = benchmark.coordinates[customer_id][0],
            y_coord = benchmark.coordinates[customer_id][1],
            demand = benchmark.demands[customer_id],
            distances = benchmark.distances[customer_id],
            start_time = benchmark.earliest[customer_id],
            end_time = benchmark.latest[customer_id],
            service_time = benchmark.service_times[customer_id],
        )
        node = Node(**params)
        node_list.append(node)

    return VRPInstance(node_list, benchmark.capacity)


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
        customers = self.inst.customers
        for i in range(len(customers)):
            row = []
            # x, y coords for customer i
            row.append(customers[i].x_coord)
            row.append(customers[i].y_coord)
            if self.include_tw:
                # earliest service start time for customer i
                row.append(customers[i].start_time)
                # lastest service start time for customer i
                row.append(customers[i].end_time)
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
            clusters[labels[i]].append(i+1)

        return clusters


class KMeansDecomposer(BaseDecomposer):
    def decompose(self):
        """Run the k-means algorithm.
        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

        """
        print('Running k-means...')
        kmeans = KMeans(n_clusters=self.num_clusters, n_init=10).fit(self.fv)
        labels = kmeans.labels_
        return self.get_clusters(labels)



class SolverWrapper(AbstractSolverWrapper):
    def solve(self, inst):
        pass


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

    inst = convert_to_vrp_instance(inst)

    decomposer = KMeansDecomposer(inst, args.num_clusters)
    clusters = decomposer.decompose()
    print(f'{len(clusters)} clusters:')
    for i in range(len(clusters)):
        print(f'cluster {i} size: {len(clusters[i])}')
    print(clusters)
