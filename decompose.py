import os
import argparse
import time

from sklearn.cluster import KMeans
import cvrplib
import hgs.tools as tools
import numpy as np

import solver_hgs as hgs

HG = 'Vrp-Set-HG' # n=[200, 1000]
SOLOMON = 'Vrp-Set-Solomon' # n=100


def build_feature_vectors_from_cvrplib(inst: cvrplib.Instance.VRPTW):
    """Build feature vectors for clustering from instance data read from cvrplib.
    
    Params:
    - inst: benchmark instance data read from cvrplib

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


def run_k_means(fv, n_clusters):
    """Run the k-means algorithm with the given feature vectors and number of clusters.
    
    Params:
    - fv: a list of feature vectors representing the customers to be clustered
    - n_cluster: number of clusters

    Returns:
    - labels: a list of labels indicating which cluster a customer belongs to
    - clusters: a list of clusters of customer IDs
    """
    kmeans = KMeans(n_clusters=n_clusters, n_init=10).fit(fv)
    labels = kmeans.labels_
    # a dict of clustered customer IDs
    clusters = {f'cluster{i}': [] for i in range(n_clusters)}
    for i in range(len(labels)):
        # customer id is shifted by 1 bc index 0 is depot
        clusters[f'cluster{labels[i]}'].append(i+1)

    return labels, clusters


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

    decomposed_instance = dict(
        coords=coords,
        demands=demands,
        vehicle_cap=inst.capacity,
        time_windows=time_windows,
        service_durations=service_durations,
        duration_matrix=duration_matrix,
        release_times=[0] * len(coords), # not used but required by hgspy.Params
    )
    
    return decomposed_instance


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

        fv = build_feature_vectors_from_cvrplib(inst)
        start = time.time()
        labels, clusters = run_k_means(fv, 3)
        end = time.time()
        # print('Total run time:', end-start)
        # print('labels:\n', labels)
        print('clusters:\n', clusters)

        # asymmetric diff
        # set(route1) - set(cluster0)
        # symmetric diff
        # set(cluster0).symmetric_difference(set(route1))

        total_cost = 0
        total_routes = []
        for _, cluster in clusters.items():
            decomp_inst = build_decomposed_instance(inst, cluster)
            print('cluster size: ', len(cluster))
            print('coords: ', np.array(decomp_inst['coords']).shape)
            print('demands: ', np.array(decomp_inst['demands']).shape)
            print('vehicle_cap: ', decomp_inst['vehicle_cap'])
            print('time_windows: ', np.array(decomp_inst['time_windows']).shape)
            print('service_durations: ', np.array(decomp_inst['service_durations']).shape)
            print('duration_matrix: ', np.array(decomp_inst['duration_matrix']).shape)
            print()

            cost, decomp_routes = hgs.call_hgs(decomp_inst)

            # map subproblem customer IDs back to the original customer IDs
            original_routes = []
            for route in decomp_routes:
                # shift cluster index by 1 bc customer IDs start at 1 (0 is the depot)
                # customer with id=1 in the subproblem is the customer with id=cluster[0] in the original problem
                route_with_original_customer_ids = [cluster[customer_id-1] for customer_id in route]
                original_routes.append(route_with_original_customer_ids)

            total_cost += cost
            total_routes.extend(original_routes)

        print("\n----- Solution -----")
        print("Total cost: ", total_cost)
        for i, route in enumerate(total_routes):
            print(f"Route {i}:", route)

    else: # ORTEC
        inst = tools.read_vrplib(os.path.join('hgs/instances', f'{args.instance_name}.txt'))
