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
OBJ_MIN_WAIT_TIME = False # is waiting time included in travel time in the objective function?

def get_min_tours(inst):
    """Returns the minimum number of tours (i.e. vehicles required) for routing the given instance.

    Params:
    - inst: benchmark instance data
    """
    # total demand of all customers / vehicle capacity
    return math.ceil(sum(inst.demands) / inst.capacity)


def get_time_window_overlap_or_gap(tw_1, tw_2):
    tw_start_1, tw_end_1 = tw_1
    tw_start_2, tw_end_2 = tw_2
    # overlap if overlap_or_gap > 0
    # gap if overlap_or_gap < 0
    overlap_or_gap = min(tw_end_1, tw_end_2) - max(tw_start_1, tw_start_2)
    return overlap_or_gap


def compute_pairwise_spatial_temportal_distance(node_1, node_2):
    # The callable should take two arrays from X as input and return a value indicating the distance between them.
    # node = [x, y, tw_start, tw_end]
    x1, y1, tw_start_1, tw_end_1 = node_1
    x2, y2, tw_start_2, tw_end_2 = node_2
    tw_width_1 = tw_end_1 - tw_start_1
    tw_width_2 = tw_end_2 - tw_start_2
    max_tw_width = max(tw_width_1, tw_width_2)

    euclidean_dist = euclidean([x1, y1], [x2, y2])
    overlap_or_gap = get_time_window_overlap_or_gap([tw_start_1, tw_end_1], [tw_start_2, tw_end_2])
    temporal_weight = 0
    if overlap_or_gap >= 0:
        overlap = overlap_or_gap
        temporal_weight = (euclidean_dist / max_tw_width) * overlap
    # else:
    #     # gap calculated by get_time_window_overlap_or_gap() is a negative number
    #     # if waiting time is not included in travel time, then large gap b/t time windows is an advantage
    #     # bc it provides more flexibility for routing
    #     # so leave it as a negative value will reduce spatial_temportal_distance
    #     gap = overlap_or_gap
    #     if OBJ_MIN_WAIT_TIME:
    #         # if waiting time IS included in travel time, then large gap b/t time windows is a penalty
    #         # so take its absolute value will increase spatial_temportal_distance
    #         gap = abs(gap)
    #     temporal_weight = (1 / euclidean_dist) * gap
    spatial_temportal_distance = euclidean_dist + temporal_weight

    # only non-negative values
    return max(0, spatial_temportal_distance)


def compute_spatial_temportal_distance_matrix(fv):
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html
    metric_callable = compute_pairwise_spatial_temportal_distance
    return pairwise_distances(fv, metric=metric_callable)


def normalize_feature_vectors(fv):
    """Normalize feature vectors using z-score."""
    fv = np.array(fv)
    # axis=0 -> row axis, runs down the rows, i.e. calculate the mean for each column/feature
    mean = np.mean(fv, axis=0)
    # ddof=1 -> degrees of freedom = N-1, i.e. sample std
    # ddof = 'delta degrees of freedom'
    # set ddof=0 for population std
    std = np.std(fv, axis=0, ddof=1)
    norm = (fv - mean) / std
    return norm


def build_feature_vectors(inst, include_time_windows=False):
    """Build feature vectors for clustering from instance data.
    
    Params:
    - inst: benchmark instance data
    - include_time_windows: True if time windows should be included in features, else False. Default is False

    Returns:
    - fv: a list of feature vectors representing the customers to be clustered, excluding the depot
    """
    fv = []
    for i in range(len(inst.coordinates)):
        row = []
        row.extend(inst.coordinates[i]) # x, y coords for customer i
        if include_time_windows:
            row.append(inst.earliest[i]) # earliest service start time for customer i
            row.append(inst.latest[i]) # lastest service start time for customer i
        fv.append(row)

    # By CVRPLIB convention, index 0 is always depot; depot should not be clustered
    return np.array(fv[1:])


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


def run_k_medoids(fv, n_clusters, include_time_windows=False):
    # https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.KMedoids.html
    print('Running k-medoids...')
    if include_time_windows:
        print('using time windows...')
        metric = 'precomputed' # for 'precomputed' must pass the fit() method a distance matrix instead of a fv
        dist_matrix = compute_spatial_temportal_distance_matrix(fv)
        X = dist_matrix
    else:
        metric = 'euclidean' #  or a callable
        X = fv
    method = 'pam'
    init = 'k-medoids++' # {‘random’, ‘heuristic’, ‘k-medoids++’, ‘build’}, default='build'
    kmedoids = KMedoids(n_clusters=n_clusters, metric=metric, method=method, init=init).fit(X)
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
    affinity = 'precomputed' # {'precomputed', 'euclidean'}
    # affinity matrix is the negative of distance matrix
    affinity_matrix = compute_spatial_temportal_distance_matrix(fv) * -1
    ap = AP(affinity=affinity).fit(affinity_matrix)
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
                                    f"python {os.path.basename(__file__)} -b=1 -n='C206' -k=3 -t | "
                                    f"python {os.path.basename(__file__)} -b=2 -n='C1_2_1' -k=3 -t")
    parser.add_argument('-n', '--instance_name', required=True,
                        help='benchmark instance name without file extension, e.g. "C206"')
    parser.add_argument('-b', '--benchmark', default=1, choices=[1, 2], type=int,
                        help='benchmark dataset to use: 1=Solomon (1987), 2=Homberger and Gehring (1999); Default=1')
    parser.add_argument('-k', '--num_clusters', required=True, type=int,
                        help='number of clusters')
    parser.add_argument('-t', '--include_time_windows', action='store_true',
                        help='use time windows for clustering or not')
    args = parser.parse_args()


    benchmark = SOLOMON if args.benchmark == 1 else HG
    path = f'CVRPLIB/{benchmark}/{args.instance_name}'
    inst, sol = cvrplib.read(instance_path=f'{path}.txt', solution_path=f'{path}.sol')
    # print(np.array(inst.distances).shape)

    # fv = build_feature_vectors(inst)
    fv = build_feature_vectors(inst, include_time_windows=args.include_time_windows)
    # print(fv[0:4])

    start = time.time()
    num_clusters = args.num_clusters
    # clusters = run_k_means(fv, num_clusters) # same result with 2 or 3 clusters
    clusters = run_k_medoids(fv, num_clusters, include_time_windows=args.include_time_windows) # cluster sizes are better balanced than k-means; better result with 3 clusters
    # clusters = run_k_medoids_fasterpam(fv, num_clusters) # actually runs slower; same result as standard medoids as expected
    # clusters = run_ap(fv) # had 9 clusters; TODO: control num of clusters? set preference based on k-means++?

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
