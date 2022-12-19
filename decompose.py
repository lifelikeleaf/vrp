import os
import argparse
import time
from multiprocessing import Pool
import math

from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation as AP
from sklearn.metrics import pairwise_distances
from sklearn_extra.cluster import KMedoids
import kmedoids as fast_kmedoids
import cvrplib
import hgs.tools as tools
import numpy as np
from scipy.spatial.distance import euclidean

import solver_hgs as hgs

HG = 'Vrp-Set-HG' # n=[200, 1000]
SOLOMON = 'Vrp-Set-Solomon' # n=100


def get_min_tours(inst):
    """Returns the minimum number of tours (i.e. vehicles required) for routing the given instance.

    Params:
    - inst: benchmark instance data
    """
    # total demand of all customers / vehicle capacity
    return math.ceil(sum(inst.demands) / inst.capacity)


def compute_pairwise_spatial_temportal_distance(node_1, node_2):
    # The callable should take two arrays from X as input and return a value indicating the distance between them.
    return euclidean(node_1, node_2) # PLACEHOLDER TODO: implement TWOL


def compute_spatial_temportal_distance_matrix(fv):
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html
    metric = compute_pairwise_spatial_temportal_distance
    return pairwise_distances(fv, metric=metric)


def build_feature_vectors(inst: cvrplib.Instance.VRPTW):
    """Build feature vectors for clustering from instance data.
    
    Params:
    - inst: benchmark instance data

    Returns:
    - fv: a list of feature vectors representing the customers to be clustered, excluding the depot
    """
    fv = []
    for i in range(len(inst.coordinates)):
        row = []
        row.extend(inst.coordinates[i]) # only includes x, y coords for now
        fv.append(row)

    # By CVRPLIB convention, index 0 is always depot; depot should not be clustered
    return fv[1:]


def get_clusters(labels, n_clusters):
    # a dict of clustered customer IDs (array index in labels)
    clusters = {f'cluster{i}': [] for i in range(n_clusters)}
    for i in range(len(labels)):
        # customer id is shifted by 1 bc index 0 is depot
        clusters[f'cluster{labels[i]}'].append(i+1)

    return clusters


def run_k_means(fv, n_clusters):
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    """Run the k-means algorithm with the given feature vectors and number of clusters.
    
    Params:
    - fv: a list of feature vectors representing the customers to be clustered
    - n_cluster: number of clusters

    Returns:
    - labels: a list of labels indicating which cluster a customer belongs to
    - clusters: a list of clusters of customer IDs
    """
    print('Running k-means...')
    kmeans = KMeans(n_clusters=n_clusters, n_init=10).fit(fv)
    labels = kmeans.labels_
    return get_clusters(labels, n_clusters)


def run_k_medoids(fv, n_clusters):
    # https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.KMedoids.html
    print('Running k-medoids...')
    # metric = 'euclidean' #  or a callable
    metric = 'precomputed' # for 'precomputed' must pass the fit() method a distance matrix instead of a fv
    dist_matrix = compute_spatial_temportal_distance_matrix(fv)
    method = 'pam'
    init = 'k-medoids++' # {‘random’, ‘heuristic’, ‘k-medoids++’, ‘build’}, default='build'
    kmedoids = KMedoids(n_clusters=n_clusters, metric=metric, method=method, init=init).fit(dist_matrix)
    labels = kmedoids.labels_
    return get_clusters(labels, n_clusters)


def run_k_medoids_fasterpam(fv, n_clusters):
    print('Running k-medoids FasterPAM...')
    dist_matrix = compute_spatial_temportal_distance_matrix(fv)
    init = 'first' # {"random", "first", "build"}
    kmedoids = fast_kmedoids.fasterpam(dist_matrix, n_clusters, init=init)
    labels = kmedoids.labels
    # kmedoids.loss: Loss of this clustering (sum of deviations), can be used to pick best out of 10 inits
    return get_clusters(labels, n_clusters)


def run_ap(fv):
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html
    print('Running Affinity Propogation...')
    affinity = 'euclidean' # {'precomputed', 'euclidean'}
    ap = AP(affinity=affinity).fit(fv)
    labels = ap.labels_
    n_clusters = len(ap.cluster_centers_indices_)
    return get_clusters(labels, n_clusters)


def build_decomposed_instance(inst, cluster):
    """Build a decomposed problem instance (i.e. a subproblem) by selecting the depot and customers in the given cluster only.
    
    Params:
    - inst: original problem instance
    - cluster: a list clustered customer IDs

    Returns:
    - decomposed_instance: a decomposed problem instance suitable for HGS consumption
    """
    # init with attributes of the depot
    depot = 0
    coords = [tuple(inst.coordinates[depot])]
    demands = [inst.demands[depot]]
    time_windows = [(inst.earliest[depot], inst.latest[depot])]
    service_durations = [inst.service_times[depot]]
    duration_matrix = []
    depot_duration_row = [inst.distances[depot][depot]]
    for customer_id in cluster:
        # select distances from the depot to customers in the given cluster only
        depot_duration_row.append(inst.distances[depot][customer_id])
    duration_matrix.append(depot_duration_row)

    for customer_id in cluster:
        coords.append(tuple(inst.coordinates[customer_id]))
        demands.append(inst.demands[customer_id])
        time_windows.append((inst.earliest[customer_id], inst.latest[customer_id]))
        service_durations.append(inst.service_times[customer_id])
        # distance to the depot is always included
        customer_duration_row = [inst.distances[customer_id][depot]]
        for customer_id_again in cluster:
            # select distances from this customer to customers in the given cluster only
            customer_duration_row.append(inst.distances[customer_id][customer_id_again])
        duration_matrix.append(customer_duration_row)

    decomp_inst = dict(
        coords=coords,
        demands=demands,
        vehicle_cap=inst.capacity,
        time_windows=time_windows,
        service_durations=service_durations,
        duration_matrix=duration_matrix,
        release_times=[0] * len(coords), # not used but required by hgspy.Params
    )

    print('cluster size: ', len(cluster))
    print('coords: ', np.array(decomp_inst['coords']).shape)
    print('demands: ', np.array(decomp_inst['demands']).shape)
    print('vehicle_cap: ', decomp_inst['vehicle_cap'])
    print('time_windows: ', np.array(decomp_inst['time_windows']).shape)
    print('service_durations: ', np.array(decomp_inst['service_durations']).shape)
    print('duration_matrix: ', np.array(decomp_inst['duration_matrix']).shape)
    print()

    return decomp_inst


def map_decomposed_to_original_customer_ids(decomp_routes, cluster):
    '''Map subproblem customer IDs back to the original customer IDs.'''
    original_routes = []
    for route in decomp_routes:
        # shift cluster index by 1 bc customer IDs start at 1 (0 is the depot)
        # customer with id=1 in the subproblem is the customer with id=cluster[0] in the original problem
        route_with_original_customer_ids = [cluster[customer_id-1] for customer_id in route]
        original_routes.append(route_with_original_customer_ids)

    return original_routes


def sequential_run_hgs(inst, clusters):
    total_cost = 0
    total_routes = []
    for _, cluster in clusters.items():
        decomp_inst = build_decomposed_instance(inst, cluster)
        cost, routes = run_hgs_on_decomposed_instance(decomp_inst, cluster)

        total_cost += cost
        total_routes.extend(routes)

    return total_cost, total_routes


def run_hgs_on_decomposed_instance(decomp_inst, cluster):
    print(f"Process ID: {os.getpid()}")
    cost, decomp_routes = hgs.call_hgs(decomp_inst)
    original_routes = map_decomposed_to_original_customer_ids(decomp_routes, cluster)
    return cost, original_routes


def parallel_run_hgs(inst, clusters):
    decomp_inst_list = []
    cluster_list = []
    for _, cluster in clusters.items():
        decomp_inst_list.append(build_decomposed_instance(inst, cluster))
        cluster_list.append(cluster)

    # start worker processes
    # num_workers = min(os.cpu_count(), len(clusters))
    num_workers = len(clusters)
    with Pool(processes=num_workers) as pool:
        results = pool.starmap(run_hgs_on_decomposed_instance, list(zip(decomp_inst_list, cluster_list)))

    total_cost = 0
    total_routes = []
    for cost, routes in results:
        total_cost += cost
        total_routes.extend(routes)

    return total_cost, total_routes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example usages: "
                                    f"python {os.path.basename(__file__)} -b=1 -n='C206' | "
                                    f"python {os.path.basename(__file__)} -b=2 -n='C1_2_1' | "
                                    f"python {os.path.basename(__file__)} -b=3 -n='ORTEC-VRPTW-ASYM-0bdff870-d1-n458-k35'")
    parser.add_argument('-n', '--instance_name', required=True,
                        help='benchmark instance name without file extension, e.g. "C206"')
    parser.add_argument('-b', '--benchmark', default=1, choices=[1, 2, 3], type=int,
                        help='benchmark dataset to use: 1=Solomon (1987), 2=Homberger and Gehring (1999), '
                             '3=ORTEC (benchmarks from EURO Meets NeurIPS 2022 Vehicle Routing Competition). Default=1')
    args = parser.parse_args()

    if args.benchmark == 1 or args.benchmark == 2: # Solomon or HG
        benchmark = SOLOMON if args.benchmark == 1 else HG
        path = f'CVRPLIB/{benchmark}/{args.instance_name}'
        inst, sol = cvrplib.read(instance_path=f'{path}.txt', solution_path=f'{path}.sol')
        # print(np.array(inst.distances).shape)

        fv = build_feature_vectors(inst)
        # print(fv[0:4])

        start = time.time()
        num_clusters = 3
        # clusters = run_k_means(fv, num_clusters) # same result with 2 or 3 clusters
        clusters = run_k_medoids(fv, num_clusters) # cluster sizes are better balanced than k-means; better result with 3 clusters
        # clusters = run_k_medoids_fasterpam(fv, num_clusters) # actually runs slower; same result as standard medoids as expected
        # clusters = run_ap(fv) # had 9 clusters; TODO: control num of clusters?

        end = time.time()
        print('Cluster run time:', end-start)
        print(f'{len(clusters)} clusters:\n', clusters)

        # asymmetric diff
        # set(route1) - set(cluster0)
        # symmetric diff
        # set(cluster0).symmetric_difference(set(route1))

        start = time.time()
        # total_cost, total_routes = sequential_run_hgs(inst, clusters)
        total_cost, total_routes = parallel_run_hgs(inst, clusters)
        end = time.time()
        print('Solver run time:', end-start)

        print("\n----- Solution -----")
        print("Total cost: ", total_cost)
        for i, route in enumerate(total_routes):
            print(f"Route {i}:", route)

    else: # ORTEC
        inst = tools.read_vrplib(os.path.join('hgs/instances', f'{args.instance_name}.txt'))
