# Author: Xu Ye <kan.ye@tum.de>

import os
import random
import pprint as pp
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import vrp.decomp.helpers as helpers
from vrp.decomp.solvers import HgsSolverWrapper, GortoolsSolverWrapper
from vrp.decomp.decomposers import KMedoidsDecomposer
from vrp.decomp.decomposition import DecompositionRunner
from vrp.decomp.decomposition import Node, VRPInstance
from vrp.decomp.constants import *
import vrp.decomp.distance_matrices as DM

MARKER_SIZE = 10


# mock data from Qi 2012
def mock_qi_2012():
    return [
        [50, 50, 0, 720, 0], # depot
        [10, 10, 60, 120, 10],
        [30, 10, 420, 480, 10],
        [30, 30, 60, 120, 10],
        [70, 70, 420, 480, 10],
        [70, 90, 60, 120, 10],
        [90, 90, 420, 480, 10],
    ]


# sample from R2_10_1
def s1():
    data = [[250, 250, 0, 7697, 0], # depot
    [94, 192, 4319, 4444, 10],
    [80, 357, 2093, 2271, 10],
    [480, 119, 2515, 2632, 10],
    [161, 275, 5370, 5524, 10],
    [187, 263, 191, 324, 10],
    [371, 163, 3328, 3414, 10],
    [332, 5, 4110, 4272, 10],
    [80, 251, 1215, 1334, 10],
    [382, 360, 696, 849, 10],
    [191, 215, 211, 338, 10]]
    return data


# sample from R2_10_1
def s2():
    data = [[250, 250, 0, 7697, 0],
    [241, 180, 6789, 6887, 10],
    [226, 316, 234, 328, 10],
    [418, 399, 1739, 1882, 10],
    [272, 420, 5515, 5691, 10],
    [456, 113, 4922, 5015, 10],
    [457, 280, 4013, 4105, 10],
    [34, 169, 1948, 2048, 10],
    [19, 38, 3376, 3521, 10],
    [32, 319, 1621, 1703, 10],
    [67, 24, 4083, 4184, 10],
    [352, 63, 4718, 4827, 10],
    [210, 113, 2778, 2864, 10],
    [390, 249, 2323, 2451, 10],
    [365, 119, 1482, 1692, 10],
    [450, 416, 2281, 2453, 10],
    [447, 184, 1699, 1816, 10],
    [247, 219, 81, 169, 10],
    [180, 283, 2963, 3107, 10],
    [174, 67, 5018, 5180, 10],
    [372, 39, 2886, 3022, 10]]
    return data

# sample from R2_10_1
def s3():
    data = [[250, 250, 0, 7697, 0], # 0
    [0, 188, 3709, 3867, 10], # 1
    [174, 347, 4047, 4191, 10], # 2
    [497, 54, 3918, 4027, 10], # 3
    [488, 173, 4836, 4966, 10], # 4
    [250, 328, 1001, 1108, 10], # 5
    [230, 487, 4552, 4605, 10], # 6
    [468, 360, 1104, 1208, 10], # 7
    [39, 162, 2269, 2373, 10], # 8
    [380, 350, 3462, 3589, 10], # 9
    [198, 85, 1032, 1189, 10], # 10
    [150, 68, 1710, 1793, 10], # 11
    [377, 432, 3343, 3429, 10], # 12
    [348, 348, 3705, 3893, 10], # 13
    [178, 420, 4290, 4401, 10], # 14
    [77, 465, 1029, 1179, 10], # 15
    [323, 34, 5244, 5394, 10], # 16
    [404, 207, 3634, 3771, 10], # 17
    [105, 140, 678, 779, 10], # 18
    [297, 131, 4640, 4803, 10], # 19
    [267, 374, 2425, 2501, 10]] # 20
    return data


def data_gen(dir_name, instance_name, sample_size):
    inst, converted_inst = helpers.read_instance(dir_name, instance_name)
    sample = random.sample(inst.customers, sample_size)
    data = [inst.depot]
    data.extend(sample)
    print(data)
    nodes = np.asarray(converted_inst.nodes)[data]
    sample_inst = VRPInstance(nodes, converted_inst.vehicle_capacity, converted_inst.extra)
    feature_vectors = helpers.build_feature_vectors(sample_inst)
    data = [feature_vectors.depot_data]
    data.extend(feature_vectors.data)
    data = np.asarray(data).tolist()
    pp.pprint(data)
    return data


def build_vrp_instance_from_mock(data):
    dist_matrix_func = DM.euclidean_vectorized
    fv_data = np.asarray(data)
    feature_vectors = helpers.FV(fv_data) # incl. depot
    decomposer = KMedoidsDecomposer(dist_matrix_func=dist_matrix_func)
    distances = decomposer.dist_matrix_func(feature_vectors, decomposer)
    distances = np.round(distances).astype(int)

    node_list = []
    for customer_id in range(len(data)):
        params = dict(
            x_coord = data[customer_id][0],
            y_coord = data[customer_id][1],
            demand = 1,
            distances = distances[customer_id],
            start_time = data[customer_id][2],
            end_time = data[customer_id][3],
            service_time = data[customer_id][4],
        )
        node = Node(**params)
        node_list.append(node)

    inst = VRPInstance(node_list, vehicle_capacity=10,
        extra={'name': '', 'num_vehicles': 10})

    return inst


def plot_instance(data, title):
    depot = data[0]
    cols = list(zip(*data[1:])) # customers only
    x, y = cols[0], cols[1]
    fig, ax = plt.subplots()
    # plot depot
    ax.scatter(depot[0], depot[1], label='depot', c='black', marker='s') # square
    # plot customers
    ax.scatter(x, y, label='customers')
    fig.legend(loc='upper left')

    # annotate customers
    for i in range(len(data[1:])):
        ax.annotate(i+1, (x[i], y[i]))

    ax.set_title(f'{title}')
    plt.show()


# TODO: tw distribution per cluster?
def plot_tw(dir_name, instance_name):
    inst, _ = helpers.read_instance(dir_name, instance_name)
    start_times = np.asarray(inst.earliest[1:])
    end_times = np.asarray(inst.latest[1:])
    tws = list(zip(start_times, end_times))
    # sort by start time
    tws.sort(key=lambda tw: tw[0])
    fig, ax = plt.subplots()
    for i, tw in enumerate(tws):
        ax.plot(tw, [i, i])

    fname = helpers.create_full_path_file_name(instance_name, TEST_DIR, 'plot', 'tw')
    fig.savefig(fname)
    plt.close()


def plot_tws():
    dir_name = HG
    input = {
        'C1': C1_10,
        'C2': C2_10,
        'R1': R1_10,
        'R2': R2_10,
        'RC1': RC1_10,
        'RC2': RC2_10,
    }
    for val in input.values():
        for instance_name in val:
            plot_tw(dir_name, instance_name)


def plot_multidim_scaling(data, dist_matrix_func):
    title = dist_matrix_func.__name__.removesuffix('_vectorized')

    fv_data = np.asarray(data)
    feature_vectors = helpers.FV(fv_data[1:], fv_data[0])
    decomposer = KMedoidsDecomposer(dist_matrix_func=dist_matrix_func)
    dist_matrix = decomposer.dist_matrix_func(feature_vectors, decomposer)

    edges = []
    n = len(dist_matrix)
    for i in range(n):
        for j in range(n):
            if i < j:
                edges.append((i, j, {'dist': dist_matrix[i, j]}))

    G = nx.Graph()
    G.add_edges_from(edges)
    dist_dict = defaultdict(dict)
    for src, dest, data in G.edges(data=True):
        dist_dict[src][dest] = data['dist']

    fig, ax = plt.subplots()
    pos = nx.kamada_kawai_layout(G, dist_dict)
    nx.draw_networkx_nodes(G, pos, node_size=MARKER_SIZE, ax=ax, label='customers')
    fig.legend(loc='upper left')
    ax.set_title(f'{title}')
    plt.show()


def cluster(vrp_inst, dist_matrix_func, sample_name, output_file_name, to_json=False):
    title = sample_name + ' - ' + dist_matrix_func.__name__.removesuffix('_vectorized')
    num_clusters = 2

    decomposer = KMedoidsDecomposer(
        dist_matrix_func=dist_matrix_func,
        num_clusters=num_clusters,
        use_overlap=True,
        use_gap=True,
        normalize=True
    )

    clusters = decomposer.decompose(vrp_inst)
    print()
    print(f'============ {title} clusters: {clusters} ============')

    if to_json:
        json_data = {
            'title': title,
            'clusters': clusters,
        }
        helpers.write_to_json(json_data, output_file_name)

    return clusters


class MockDecomposer:
    def __init__(self, clusters) -> None:
        self.clusters = clusters

    def decompose(self, inst):
        return self.clusters


def solve(vrp_inst, clusters, title, min_total, output_file_name, no_decomp=False, to_json=False, verbose=False):
    num_clusters = 2
    time_limit = 5

    if min_total:
        solver = GortoolsSolverWrapper(time_limit=time_limit, min_total=min_total)
    else:
        # solver = HgsSolverWrapper(time_limit=time_limit)
        solver = GortoolsSolverWrapper(time_limit=time_limit, min_total=min_total)
    if no_decomp:
        solution = solver.solve(vrp_inst)
        title = 'no decomp'
    else:
        decomposer = MockDecomposer(clusters)
        runner = DecompositionRunner(vrp_inst, decomposer, solver)
        runner.decomposer.clusters = clusters
        solution = runner._run_solver_parallel(num_workers=num_clusters)

    if solution.metrics[METRIC_COST] == float('inf'):
        print('No feasible solution found.')
    else:
        cost = solution.metrics[METRIC_COST]
        routes = solution.routes
        print()
        print(f'============ {title} cost: {cost} ============')
        driving_time, wait_time = helpers.print_solution(solution, vrp_inst, verbose=verbose)
        print()

        if to_json:
            json_data = {
                'title': title,
                'cost': round(cost),
                'driving_time': round(driving_time),
                'wait_time': round(wait_time),
                'num_routes': len(routes),
                'routes': routes,
            }
            helpers.write_to_json(json_data, output_file_name)

        return solution


def plot_clusters(clusters, title, clusters_dir):
    # plot depot
    depot = data[0]
    fig, ax = plt.subplots(layout='constrained')
    ax.scatter(depot[0], depot[1], label='depot', c='black', marker='s') # square

    # plot customers
    nodes = list(zip(*data)) # incl. depot
    x, y = np.asarray(nodes[0]), np.asarray(nodes[1])
    for i, cluster in enumerate(clusters):
        ax.scatter(x[cluster], y[cluster], label=f'cluster {i+1}')

    # annotate customers
    cols = list(zip(*data[1:])) # customers only
    x, y = cols[0], cols[1]
    for i in range(len(data[1:])):
        ax.annotate(i+1, (x[i], y[i]))

    fig.legend(loc='outside right upper') # 'outside' only works with constrained_layout
    ax.set_title(f'{title}')
    # plt.show()
    fname = os.path.join(clusters_dir, title)
    fig.savefig(fname)


def plot_routes(data, routes, title, cost, driving_time, wait_time, routes_dir):
    # plot depot
    depot = data[0]
    fig, ax = plt.subplots(layout='constrained')
    ax.scatter(depot[0], depot[1], label='depot', c='black', marker='s') # square

    # plot customers
    nodes = list(zip(*data)) # incl. depot
    x, y = np.asarray(nodes[0]), np.asarray(nodes[1])
    for i, route in enumerate(routes):
        # from depot to first node
        x_1 = [depot[0], x[route[0]]]
        y_1 = [depot[1], y[route[0]]]
        ax.plot(x_1, y_1, c='black', ls='dotted')

        ax.plot(x[route], y[route], label=f'route {i+1}', ls='-', marker='o')

    # annotate customers
    cols = list(zip(*data[1:])) # customers only
    x, y = cols[0], cols[1]
    for i in range(len(data[1:])):
        ax.annotate(i+1, (x[i], y[i]))

    fig.legend(loc='outside right upper') # 'outside' only works with constrained_layout
    ax.set_title(f'{title} (cost: {cost})\n(driving time: {driving_time}, wait time: {wait_time})')
    # plt.show()
    fname = os.path.join(routes_dir, title)
    fig.savefig(fname)


if __name__ == '__main__':
    # data_funcs = [mock_qi_2012, s1, s2, s3]
    data_funcs = [s3]
    dist_matrix_funcs = [DM.euclidean_vectorized, DM.v2_2_vectorized, DM.qi_2012_vectorized]
    # dist_matrix_funcs = [DM.euclidean_vectorized]

    '''STEP 0'''
    min_total = True
    ext = 'min total' if min_total else 'min driving'

    for data_func in data_funcs:
        data = data_func()
        sample_name = data_func.__name__

        # dir_name = HG
        # instance_name = 'R2_10_1'
        # sample_size = 20
        # data = data_gen(dir_name, instance_name, sample_size)

        vrp_inst = build_vrp_instance_from_mock(data)

        '''STEP 1'''
        # plot_instance(data, sample_name)

        clusters_dir = os.path.join(TEST_DIR, 'clusters')
        helpers.make_dirs(clusters_dir)
        clusters_output_file_name = os.path.join(clusters_dir, f'{sample_name}')

        routes_dir = os.path.join(TEST_DIR, 'routes', ext)
        helpers.make_dirs(routes_dir)
        routes_output_file_name = os.path.join(routes_dir, f'{sample_name}')

        '''STEP 2.1'''
        # for dist_matrix_func in dist_matrix_funcs:
        #     cluster(vrp_inst, dist_matrix_func, sample_name, clusters_output_file_name, to_json=True)

        clusters_data_gen = helpers.read_json_gen(clusters_output_file_name)
        for clusters_data in clusters_data_gen:
            title = clusters_data['title']
            clusters = clusters_data['clusters']

            '''STEP 2.2'''
            # plot_clusters(clusters, title, clusters_dir)

            '''STEP 3.1'''
            # solve(vrp_inst, clusters, title, min_total, routes_output_file_name, to_json=True, verbose=True)

        '''STEP 3.2'''
        routes_data_gen = helpers.read_json_gen(routes_output_file_name)
        for routes_data in routes_data_gen:
            title = routes_data['title']
            cost = routes_data['cost']
            driving_time = routes_data['driving_time']
            wait_time = routes_data['wait_time']
            routes = routes_data['routes']

            title = title + ' - ' + ext
            plot_routes(data, routes, title, cost, driving_time, wait_time, routes_dir)

