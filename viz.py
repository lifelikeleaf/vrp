# Author: Xu Ye <kan.ye@tum.de>

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

from manual_test import read_instance

MARKER_SIZE = 10


# mock data from Qi 2012
def mock_qi_2012():
    '''min total'''
    # euclidean cost: 400
    # [[3, 1], [2], [6, 4], [5]]

    # v2_2 cost: 400
    # [[6, 4], [5], [3, 1], [2]]

    # qi_2012 cost: 400
    # [[4, 6], [2], [3, 1], [5]]
    '''min driving'''
    # euclidean cost: 240
    # [[3, 1, 2], [5, 6, 4]]

    # v2_2 cost: 240
    # [[5, 6, 4], [3, 1, 2]]

    # qi_2012 cost: 400
    # [[4, 6], [2], [3, 1], [5]]
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
    '''min total'''
    # euclidean cost: 3120
    # [[5, 10], [4], [1], [2], [8], [7], [9], [6], [3]]

    # v2_2 cost: 3120
    # [[7], [6], [3], [5, 10], [4], [1], [2], [8], [9]]

    # qi_2012 cost: 2998
    # [[5, 10], [2], [8], [9], [7, 1], [6], [4], [3]]
    '''min driving'''
    # euclidean cost: 1660
    # [[10, 5, 8, 2, 1, 4], [9, 3, 6, 7]]

    # v2_2 cost: 1835
    # [[10, 5, 8, 2, 1, 4], [9], [3, 6, 7]]

    # qi_2012 cost: 1914
    # [[3, 6, 7, 1, 4], [10, 5, 8, 2], [9]]
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
    '''min total'''
    # euclidean cost: 6238
    # [[6], [14, 16], [3, 15], [2], [4], [5], [13], [11, 19], [18], [1], [17], [9, 7], [12, 20, 8], [10]]

    # v2_2 cost: 6116
    # [[4], [3, 15], [6], [11, 5], [14, 16], [17, 2], [1], [19], [18], [9, 7, 13], [12, 20, 8], [10]]

    # qi_2012 cost: 6472
    # [[11, 5], [8, 10], [1], [19, 4], [12], [20], [6], [14, 16], [17, 2], [3, 15], [9, 7, 13], [18]]
    '''min driving'''
    # euclidean cost: 3109
    # [[14, 16, 13, 6, 5], [2, 3, 15, 4], [18], [17, 12, 20, 11], [9, 7, 8, 10, 19, 1]]

    # v2_2 cost: 3035
    # [[3, 15, 6, 4], [17, 14, 16, 13, 12, 20, 11, 5], [2, 9, 7, 8, 10, 19, 1], [18]]

    # qi_2012 cost: 3678
    # [[2, 9, 7, 18], [3, 15, 6], [17, 14, 16, 13, 20], [11, 5, 4], [12, 8, 10, 19, 1]]
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
    '''min total'''
    # euclidean cost: 6704
    # [[2, 14, 6], [12, 9, 17, 13], [5], [15], [4], [7], [20], [16], [8], [3], [1], [18, 10], [11], [19]]

    # v2_2 cost: 6520
    # [[17, 3], [7], [16], [10], [19], [8], [18, 15], [11], [4], [1, 2, 14, 6], [12, 9, 13], [5], [20]]

    # qi_2012 cost: 6128
    # [[19, 4, 16], [1, 2, 14, 6], [12, 9, 13], [20], [17, 3], [8], [7], [10], [5], [18, 15], [11]]
    '''min driving'''
    # euclidean cost: 3718
    # [[3, 19, 16], [18, 10, 11, 8, 1], [5, 20, 12, 9, 13], [15, 2, 14, 6], [7, 17, 4]]

    # v2_2 cost: 3524
    # [[15, 14, 6], [7, 12, 9, 13], [5, 20, 2], [17, 3, 4, 16], [18, 10, 11, 8, 1, 19]]

    # qi_2012 cost: 4405
    # [[15], [7], [18, 10, 11, 8], [5], [17, 3, 4, 16], [1, 2, 14, 6], [20, 12, 9, 13, 19]]
    data = [[250, 250, 0, 7697, 0],
    [0, 188, 3709, 3867, 10],
    [174, 347, 4047, 4191, 10],
    [497, 54, 3918, 4027, 10],
    [488, 173, 4836, 4966, 10],
    [250, 328, 1001, 1108, 10],
    [230, 487, 4552, 4605, 10],
    [468, 360, 1104, 1208, 10],
    [39, 162, 2269, 2373, 10],
    [380, 350, 3462, 3589, 10],
    [198, 85, 1032, 1189, 10],
    [150, 68, 1710, 1793, 10],
    [377, 432, 3343, 3429, 10],
    [348, 348, 3705, 3893, 10],
    [178, 420, 4290, 4401, 10],
    [77, 465, 1029, 1179, 10],
    [323, 34, 5244, 5394, 10],
    [404, 207, 3634, 3771, 10],
    [105, 140, 678, 779, 10],
    [297, 131, 4640, 4803, 10],
    [267, 374, 2425, 2501, 10]]
    return data


def data_gen(dir_name, instance_name, sample_size):
    inst, converted_inst = read_instance(dir_name, instance_name)
    sample = random.sample(inst.customers, sample_size)
    data = [inst.depot]
    data.extend(sample)
    print(data)
    # data = [0, 230, 634, 248, 799, 997, 605, 266, 901, 625, 698] # sample_R2_10_1
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


def plot_clusters(data, dist_matrix_func, plot=False):
    title = dist_matrix_func.__name__.removesuffix('_vectorized')
    num_clusters = 2
    time_limit = 5

    vrp_inst = build_vrp_instance_from_mock(data)
    decomposer = KMedoidsDecomposer(dist_matrix_func=dist_matrix_func, use_overlap=True, use_gap=True, normalize=True)
    decomposer.num_clusters = num_clusters

    if not plot:
        solver = GortoolsSolverWrapper(time_limit=time_limit, min_total=True)
        runner = DecompositionRunner(vrp_inst, decomposer, solver)
        solution = runner.run(in_parallel=True, num_workers=num_clusters)
        if solution.metrics[METRIC_COST] == float('inf'):
            print('No feasible solution found.')
        else:
            cost = solution.metrics[METRIC_COST]
            routes = solution.routes
            print()
            print(f'============ {title} cost: {cost} ============')
            print(f'num routes: {len(routes)}')
            print(routes)
    else:
        clusters = decomposer.decompose(vrp_inst)
        print()
        print(f'============ {title} clusters: {clusters} ============')

        # plot depot
        depot = data[0]
        fig, ax = plt.subplots()
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

        fig.legend(loc='upper left')
        ax.set_title(f'{title}')
        plt.show()


def plot_routes(data):
    sample = 's3'
    prefix = 'min total - ' + sample

    # title = prefix + ' - euclidean'
    # routes = [[2, 14, 6], [12, 9, 17, 13], [5], [15], [4], [7], [20], [16], [8], [3], [1], [18, 10], [11], [19]]

    # title = prefix + ' - v2_2'
    # routes = [[17, 3], [7], [16], [10], [19], [8], [18, 15], [11], [4], [1, 2, 14, 6], [12, 9, 13], [5], [20]]

    # title = prefix + ' - qi'
    # routes = [[19, 4, 16], [1, 2, 14, 6], [12, 9, 13], [20], [17, 3], [8], [7], [10], [5], [18, 15], [11]]

    prefix = 'min driving - ' + sample

    # title = prefix + ' - euclidean'
    # routes = [[3, 19, 16], [18, 10, 11, 8, 1], [5, 20, 12, 9, 13], [15, 2, 14, 6], [7, 17, 4]]

    # title = prefix + ' - v2_2'
    # routes = [[15, 14, 6], [7, 12, 9, 13], [5, 20, 2], [17, 3, 4, 16], [18, 10, 11, 8, 1, 19]]

    title = prefix + ' - qi'
    routes = [[15], [7], [18, 10, 11, 8], [5], [17, 3, 4, 16], [1, 2, 14, 6], [20, 12, 9, 13, 19]]

    # plot depot
    depot = data[0]
    fig, ax = plt.subplots()
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

    fig.legend(loc='upper left')
    ax.set_title(f'{title}')
    plt.show()


if __name__ == '__main__':
    # data_funcs = [mock_qi_2012, s1, s2, s3]
    data_funcs = [s3]

    for data_func in data_funcs:
        data = data_func()

        # dir_name = HG
        # instance_name = 'R2_10_1'
        # sample_size = 20
        # data = data_gen(dir_name, instance_name, sample_size)

        # plot_instance(data, title=f'{data_func.__name__}')

        # plot_multidim_scaling(data, dist_matrix_func)

        # dist_matrix_funcs = [DM.euclidean_vectorized, DM.v2_2_vectorized, DM.qi_2012_vectorized]
        # for dist_matrix_func in dist_matrix_funcs:
        #     plot_clusters(data, dist_matrix_func, plot=True)

        plot_routes(data)

