# Author: Xu Ye <kan.ye@tum.de>

"""Auxiliary code for manual tests. Not guaranteed to be up to date."""

import os
import sqlite3
from collections import defaultdict
import itertools
from difflib import SequenceMatcher
from scipy.integrate import quad

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import Levenshtein

'''To use hgs tools, you must have hgs installed, see vrp/decomp/solvers.py for details.'''
from vrp.third_party.solver.hgs import tools

import vrp.third_party.cvrplib as cvrplib
import vrp.decomp.helpers as helpers
from vrp.decomp.solvers import HgsSolverWrapper, GortoolsSolverWrapper
from vrp.decomp.decomposers import (
    KMedoidsDecomposer, APDecomposer, KMeansDecomposer, HierarchicalDecomposer
)
from vrp.decomp.decomposition import DecompositionRunner, VRPInstance
from vrp.decomp.constants import *
import vrp.decomp.distance_matrices as DM


def list_benchmark_names():
    benchmark = cvrplib.list_names(low=100, high=100, vrp_type='vrptw')
    print(benchmark)


def benchmarks_to_sqlite(db_name: str):
    '''Example query:
    select service_time from nodes as n join instances as i
    where n.instance_name = i.instance_name and i.instance_type = 'C1' group by 1;
    '''
    con = sqlite3.connect(db_name)
    cur = con.cursor()

    sql_create_instances_table = '''
    CREATE TABLE instances (
        instance_name TEXT PRIMARY KEY,
        instance_type TEXT,
        instance_size INTEGER,
        vehicle_number INTEGER,
        vehicle_capacity INTEGER
    )
    '''

    sql_create_nodes_table = '''
    CREATE TABLE nodes (
        instance_name TEXT REFERENCES instances,
        node_id INTEGER,
        x_coord INTEGER,
        y_coord INTEGER,
        demand INTEGER,
        start_time INTEGER,
        end_time INTEGER,
        service_time INTEGER
    )
    '''

    cur.execute(sql_create_instances_table)
    cur.execute(sql_create_nodes_table)

    n = 10
    dir_name = HG
    insert_list = []
    inst_gen = helpers.get_hg_instance_names(n)
    for instance_name in inst_gen:
    # for instance_name in ['C1_2_1']:
        instance_type = instance_name.split('_')[0]
        inst = helpers.read_instance(dir_name, instance_name)[0]
        instance_size = inst.n_customers
        vehicle_number = inst.n_vehicles
        vehicle_capacity = inst.capacity

        params = (instance_name, instance_type, instance_size, vehicle_number, vehicle_capacity)
        cur.execute('INSERT INTO instances VALUES (?, ?, ?, ?, ?)', params)

        for i in range(len(inst.coordinates)):
            node_id = i
            x_coord, y_coord = inst.coordinates[i]
            demand = inst.demands[i]
            start_time = inst.earliest[i]
            end_time = inst.latest[i]
            service_time = inst.service_times[i]

            row = (instance_name, node_id, x_coord, y_coord, demand,
                    start_time, end_time, service_time)

            insert_list.append(row)

    cur.executemany('INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?)', insert_list)
    con.commit()
    con.close()


def test_standardize_fv():
    fv = [
        [1,2,3,4],
        [4,3,2,1],
        [5,6,7,8],
    ]
    fv = helpers.standardize_feature_vectors(fv)
    print(fv)


def test_normalize_matrix():
    m = [
        [1,2],
        [4,3],
    ]
    m = helpers.normalize_matrix(m)
    print(m)
    print(type(m))

def test_read_json():
    file_name = 'test'
    data = helpers.read_json(file_name)
    print(len(data))


def test_read_json_gen():
    file_name = 'test'
    json_gen = helpers.read_json_gen(file_name)
    # print(next(json_gen))
    data = []
    for item in json_gen:
        data.append(item)
    print(len(data))


def test_get_clusters():
    decomposer = KMedoidsDecomposer(dist_matrix_func=None)
    labels = [0, 1, 1, 3, 0, 0, 3, 3, 3, 1, 1]
    clusters = decomposer.get_clusters(labels)
    assert clusters == [[1, 5, 6], [2, 3, 10, 11], [4, 7, 8, 9]]
    print(clusters)


def summary_fv(to_excel=False, summary_to_excel=False):
    dir_name = SOLOMON
    names = ['C101']
    # names.extend(FOCUS_GROUP_C1)
    # names.extend(FOCUS_GROUP_RC2)
    norm = False

    for instance_name in names:
        _, converted_inst = helpers.read_instance(dir_name, instance_name)

        feature_vectors = helpers.build_feature_vectors(converted_inst)
        df = pd.DataFrame(feature_vectors.data, columns=['x', 'y', 'start', 'end'])
        print(df.iloc[0]['x'])
        print(len(df))
        # min = df.start.min()
        # max = df.end.max()
        # fv_np = np.asarray(feature_vectors.data)
        # min = fv_np.min(axis=0)[2]
        # max = fv_np.max(axis=0)[3]
        # print(f'min: {min}')
        # print(f'max: {max}')
        exit()
        df['time_duration'] = df['end'] - df['start']
        df_desc = df.describe().round(2)
        df_desc.drop(index=['count', '25%', '50%', '75%'], inplace=True)
        print(f'\n{instance_name}')
        print(df_desc)
        if to_excel:
            ext = '_norm' if norm else ''
            helpers.make_dirs(TEST_DIR)
            file_name = os.path.join(TEST_DIR, f'FV')
            helpers.write_to_excel(df, file_name=f'{file_name}{ext}', sheet_name=instance_name, overlay=False)
            if summary_to_excel:
                file_name = os.path.join(TEST_DIR, f'FV_summary')
                helpers.write_to_excel(df_desc, file_name=f'{file_name}{ext}', sheet_name=instance_name, overlay=False, index=True)


def test_pairwise_distance():
    dir_name = SOLOMON
    instance_name = 'C101'
    gap = False

    _, converted_inst = helpers.read_instance(dir_name, instance_name)
    decomposer = KMedoidsDecomposer(dist_matrix_func=None, use_overlap=True, use_gap=gap)
    feature_vectors = helpers.build_feature_vectors(converted_inst)
    fv = feature_vectors.data

    fv_i = fv[0]
    fv_j = fv[20]

    pd = DM._PairwiseDistance(fv_i, fv_j, decomposer)
    pd_td = pd.temporal_dist_v1()
    pd_std = pd.spatial_temporal_dist_v1()
    tw_i = pd.tw_width_i
    tw_j = pd.tw_width_j
    tw = pd.max_tw_width
    ed = pd.euclidean_dist
    olg = pd.overlap_or_gap

    td = ol = gap = ol_td = gap_td = 0
    if olg >= 0:
        ol = olg
        td = ol_td = helpers.safe_divide(ed, tw) * olg
    else:
        gap = olg
        td = gap_td = helpers.safe_divide(1, ed) * olg

    print(f'FV_i = {fv_i}, FV_j = {fv_j}')
    print(f'TW_i = {tw_i}, TW_j = {tw_j}, max TW = {tw}')
    print(
        f'Overlap or gap: {round(olg, 2)}; '
        f'euclidean dist: {round(ed, 2)}; temporal dist: {round(td, 2)}'
    )
    print(f'temporal dist from PD = {round(pd_td, 2)}')
    print(
        f'Overlap ST dist v1 = {round(ed + ol_td, 2)} \n'
        f'Overlap ST dist v2 = {round(ed + ol, 2)} \n'
        f'Gap ST dist v1 = {round(ed + gap_td, 2)} \n'
        f'Gap ST dist v2 = {round(ed + gap, 2)}'
    )
    print(f'ST dist from PD = {round(pd_std, 2)}')


def trial_formulas(stats):
    '''Formula v1: temporal dist'''
    # stats['TD_OL'] = stats['euclidean_dist'] / stats['max_tw_width'] * stats['overlap']
    # stats['dist_OL'] = stats['euclidean_dist'] + stats['TD_OL']

    ## gap could be so large that even with normalization dist_G would be driven to < 0
    # stats['TD_G'] = 1 / stats['euclidean_dist'] * stats['gap']
    # stats['dist_G'] = stats['euclidean_dist'] - stats['TD_G']

    '''Formula v2.1: temporal weight'''
    ## up to ~50% weight bc by definition overlap <= max_tw_width
    ## even though after normalization it could be a little higher
    # stats['temp_w8_OL'] = stats['overlap'] / (stats['overlap'] + stats['max_tw_width'])

    '''Formula v2.2: relative to the planning horizon'''
    # stats['temp_w8_OL'] = stats['relative_overlap'] * (1 - stats['relative_tw_width'])
    # stats['dist_OL'] = stats['euclidean_dist'] * (1 + stats['temp_w8_OL'])

    # stats['temp_w8_G'] = stats['gap'] / (stats['gap'] + stats['euclidean_dist'])
    # stats['dist_G'] = stats['euclidean_dist'] * (1 - stats['temp_w8_G'])

    return stats


def test_get_constituents(to_excel=False):
    dir_name = SOLOMON
    instance_name = 'C101'
    file_name = os.path.join(TEST_DIR, 'stats_matrix')
    norm = True

    _, converted_inst = helpers.read_instance(dir_name, instance_name)
    decomposer = KMedoidsDecomposer(dist_matrix_func=None, normalize=norm)
    feature_vectors = helpers.build_feature_vectors(converted_inst)
    fv = feature_vectors.data

    # constituents_matrix = DM._get_constituents_matrix(fv, decomposer)
    constituents_matrix = DM._get_constituents_vectorized(fv, decomposer, as_matrix=True)

    # flatten bc per-column arrays must each be 1-dimensional
    constituents_df = pd.DataFrame({key: val.flatten() for key, val in constituents_matrix.items()})
    df = trial_formulas(constituents_df)
    df_desc = df.describe()
    # df_no_zeros = df.replace(0, np.NaN)
    # df_desc = df_no_zeros.describe()
    df_desc.drop(index=['count', '25%', '50%', '75%'], inplace=True)
    print(f'\n{instance_name}')
    print(df_desc.round(2))

    if to_excel:
        ext = ''
        ext += '_norm' if norm else ''
        helpers.make_dirs(TEST_DIR)
        helpers.write_to_excel(df, file_name=f'{file_name}{ext}', sheet_name=f'{instance_name}{ext}', overlay=False, index=True)


def dist_matrix_to_excel():
    dir_name = SOLOMON
    instance_name = 'C101'
    instance_size = 100
    dist_matrix_func = DM.v2_6_vectorized
    file_name = os.path.join(TEST_DIR, f'DM_{dist_matrix_func.__name__}')
    '''
    if all 3 are False, it's plain old euclidean
    if only norm is True, it's normalized euclidean
    without normalization, gap could reduce dist to unreasonably small values even with v2 formulas
    while overlap could increase dist by 50% to almost 100%
    '''
    overlap = False
    gap = True
    norm = True

    ext = ''
    ext += '_overlap' if overlap else ''
    ext += '_gap' if gap else ''
    ext += '_norm' if norm else ''

    _, converted_inst = helpers.read_instance(dir_name, instance_name)
    decomposer = KMedoidsDecomposer(dist_matrix_func=dist_matrix_func, use_overlap=overlap, use_gap=gap, normalize=norm)
    feature_vectors = helpers.build_feature_vectors(converted_inst)
    dist_matrix = decomposer.dist_matrix_func(feature_vectors, decomposer)
    '''
    max_tw_width (by construct) and euclidean_dist (by nature) would only be 0 with itself, so could make sense to remove 0s;
    but overlap and gap could legitimately be 0 b/t node pairs, albeit not very common;
    and overlap with itself is always the full tw_width of itself,
    whereas gap with itself is always 0
    '''
    dist_matrix = np.asarray(dist_matrix)
    customer_ids = [i for i in range(1, instance_size + 1)]
    df = pd.DataFrame(dist_matrix, columns=customer_ids, index=customer_ids)
    helpers.make_dirs(TEST_DIR)
    helpers.write_to_excel(df, file_name=f'{file_name}{ext}', sheet_name=f'{instance_name}{ext}', overlay=False, index=True)
    print(dist_matrix.shape)
    '''
    Visual inspection for instance C101:
    Overlap
        - non-normalized: lots of overlaps; dist v1 < dist v2
        - normalized: only a few (10) overlaps; dist v1 > dist v2
    Gap:
        - non-normalized: lots of gaps; gap size much > euclidean dist;
            lots of negative (thus 0) distances; v2 more negative than v1 (v2 < v1)
        - normalized: lots of gaps;

    Example temporal dist and euclidean dist comparison:
    FV: node 1 [45, 68, 912, 967] and node 21 [30, 52, 914, 965]
        - euclidean_dist = ((30-45)**2 + (52-68)**2)**0.5 = 21.93171219946131
        - node 1 TW width: 55; node 2 TW width: 51
        - overlap = 51
        - temporal_dist = euclidean_dist / max_tw_width * overlap
            = 21.93171219946131 / 55 * 51 = 20.336678584955035
        - spatial_temporal_dist = euclidean_dist + temporal_dist
            = 21.93171219946131 + 20.336678584955035 = 42.26839078441634
        - almost doubles euclidean_dist
    '''
    # print(dist_matrix[0, 20])
    # print(dist_matrix[10, 10]) # dist to self should always be 0


def calc_omega_factor(inst: cvrplib.Instance.VRPTW, cluster=None):
    '''Van Landeghem, H. R. G. (1988)'''
    if cluster is None:
        start_times = np.asarray(inst.earliest[1:])
        end_times = np.asarray(inst.latest[1:])
    else:
        start_times = np.asarray(inst.earliest)[cluster]
        end_times = np.asarray(inst.latest)[cluster]

    n = len(start_times)
    tw_sizes = end_times - start_times
    avg_tw_size = tw_sizes.sum() / n
    planning_horizon = max(end_times) - min(start_times)
    omega_factor = avg_tw_size / planning_horizon * 100
    return avg_tw_size, omega_factor


def calc_omega_factors():
    dir_name = HG
    input = {
        # 'test': ['C1_2_1', 'C2_2_1'],
        'C1': C1_10,
        'C2': C2_10,
        'R1': R1_10,
        'R2': R2_10,
        'RC1': RC1_10,
        'RC2': RC2_10,
    }
    for val in input.values():
        for instance_name in val:
            inst, _ = helpers.read_instance(dir_name, instance_name)
            avg_tw_size, omega_factor = calc_omega_factor(inst)
            print(f'{instance_name} avg TW size: {round(avg_tw_size, 2)}')
            print(f'{instance_name} omega factor: {round(omega_factor, 2)}%')
        print()


def calc_omega_factors_per_cluster():
    dir_name = HG
    instance_name = 'C1_10_1'
    num_clusters = 10
    inst, converted_inst = helpers.read_instance(dir_name, instance_name)
    dist_matrix_func = DM.v2_2_vectorized

    decomposer = KMedoidsDecomposer(
        dist_matrix_func=dist_matrix_func,
        num_clusters=num_clusters,
        use_overlap=True,
        use_gap=True,
        normalize=True,
        # penalize_gap=True,
    )
    clusters = decomposer.decompose(converted_inst)

    for i, cluster in enumerate(clusters):
        avg_tw_size, omega_factor = calc_omega_factor(inst, cluster)
        print(f'{instance_name} cluster {i+1} avg TW size: {round(avg_tw_size, 2)}')
        # print(f'{instance_name} cluster {i+1} omega factor: {round(omega_factor, 2)}%')


def test_decompose():
    dir_name = SOLOMON
    instance_name = 'C101'
    _, converted_inst = helpers.read_instance(dir_name, instance_name)
    dist_matrix_func = DM.euclidean_vectorized

    # decomposer = KMedoidsDecomposer(dist_matrix_func=dist_matrix_func, use_overlap=True)
    decomposer = HierarchicalDecomposer(dist_matrix_func=dist_matrix_func, use_overlap=True)
    decomposer.num_clusters = 3
    clusters = decomposer.decompose(converted_inst)
    print(clusters)


def test_solver():
    dir_name = HG
    instance_name = 'C1_10_2'
    _, converted_inst = helpers.read_instance(dir_name, instance_name)
    time_limit = 10

    print(f'instance: {instance_name}')
    # solver = HgsSolverWrapper(time_limit=time_limit, init_sol=True)
    # solver = GortoolsSolverWrapper(time_limit=time_limit, min_total=False)
    solver = GortoolsSolverWrapper(time_limit=time_limit, min_total=True)
    solution = solver.solve(converted_inst)
    if solution.metrics[METRIC_COST] == float('inf'):
        print('No feasible solution found.')
    else:
        helpers.print_solution(solution, converted_inst)


def test_framework():
    dir_name = HG
    instance_name = 'C1_10_3'
    num_clusters = 10
    _, converted_inst = helpers.read_instance(dir_name, instance_name)
    lam = 0.1
    dist_matrix_func = DM.get_dist_matrix_func_v2_2(lam, lam)
    time_limit = 10

    print(f'instance: {instance_name}')
    # solver = HgsSolverWrapper(time_limit=time_limit)
    # solver = GortoolsSolverWrapper(time_limit=time_limit, min_total=False)
    solver = GortoolsSolverWrapper(time_limit=time_limit, min_total=True)

    decomposer = KMedoidsDecomposer(
        dist_matrix_func=dist_matrix_func,
        num_clusters=num_clusters,
        use_overlap=True,
        # use_gap=True,
        normalize=True
    )
    runner = DecompositionRunner(converted_inst, decomposer, solver)
    solution = runner.run(in_parallel=True, num_workers=num_clusters)
    if solution.metrics[METRIC_COST] == float('inf'):
        print('No feasible solution found.')
    else:
        helpers.print_solution(solution, converted_inst, verbose=False)


def test_antiderivative_vs_quad():

    def def_integral(antiderivative, lower_limit, upper_limit, **kwargs):
        return antiderivative(upper_limit, **kwargs) - antiderivative(lower_limit, **kwargs)

    k1 = 1
    k2 = 1.5
    k3 = 2

    # For instance C101
    # (i, j) = (1, 2)
    a_prime = 1004 # a' = lower_limit of integral
    b_prime = 1059 # b' = upper_limit of integral
    start_time_j = 825 # c
    end_time_j = 870 # d

    # (i, j) = (2, 1)
    # a_prime = 917 # a' = lower_limit of integral
    # b_prime = 962 # b' = upper_limit of integral
    # start_time_j = 912 # c
    # end_time_j = 967 # d

    # For mock data from Qi 2012
    # (i, j) = (1, 2)
    # a_prime = 90 # a' = lower_limit of integral
    # b_prime = 150 # b' = upper_limit of integral
    # start_time_j = 420 # c
    # end_time_j = 480 # d

    # (i, j) = (2, 1)
    # a_prime = 450 # a' = lower_limit of integral
    # b_prime = 510 # b' = upper_limit of integral
    # start_time_j = 60 # c
    # end_time_j = 120 # d

    # x=t' in [c, d]
    def k1_integrand(x, k1, d):
        return -k1 * x + k1 * d

    k1_quad_result = quad(k1_integrand, a_prime, b_prime, args=(k1, end_time_j))
    print(k1_quad_result)

    def k1_antiderivative(x, k1, d):
        # antiderivative of k1_integrand
        return -k1 * x ** 2 / 2 + k1 * d * x

    k1_result = def_integral(k1_antiderivative, a_prime, b_prime, k1=k1, d=end_time_j)
    print(k1_result)

    # x=t' < c
    def k2_integrand(x, k1, k2, c, d):
        return k2 * x + k1 * d - (k1 + k2) * c

    k2_quad_result = quad(k2_integrand, a_prime, b_prime, args=(k1, k2, start_time_j, end_time_j))
    print(k2_quad_result)

    def k2_antiderivative(x, k1, k2, c, d):
        return k2 * x ** 2 / 2 + k1 * d * x - (k1 + k2) * c * x

    k2_result = def_integral(k2_antiderivative, a_prime, b_prime, k1=k1, k2=k2, c=start_time_j, d=end_time_j)
    print(k2_result)

    # x=t' > d
    def k3_integrand(x, k3, d):
        return -k3 * x + k3 * d

    k3_quad_result = quad(k3_integrand, a_prime, b_prime, args=(k3, end_time_j))
    print(k3_quad_result)

    def k3_antiderivative(x, k3, d):
        return -k3 * x ** 2 / 2 + k3 * d * x

    k3_result = def_integral(k3_antiderivative, a_prime, b_prime, k3=k3, d=end_time_j)
    print(k3_result)


def test_dist_matrix_qi_2012(use_mock_data=False):
    dir_name = SOLOMON
    instance_name = 'C101'
    dist_matrix_func = DM.qi_2012_vectorized
    _, converted_inst = helpers.read_instance(dir_name, instance_name)
    decomposer = KMedoidsDecomposer(dist_matrix_func=dist_matrix_func)

    if not use_mock_data:
        feature_vectors = helpers.build_feature_vectors(converted_inst)
        dist_matrix = decomposer.dist_matrix_func(feature_vectors, decomposer)
        print(dist_matrix[3, 3].round(2))
        print(dist_matrix[0, 1].round(2))
        print(dist_matrix[1, 0].round(2))
    else:
        # mock data from Qi 2012
        fv_data = [
            [50, 50, 0, 720, 0], # depot
            [10, 10, 60, 120, 10],
            [30, 10, 420, 480, 10],
            [30, 30, 60, 120, 10],
            [70, 70, 420, 480, 10],
            [70, 90, 60, 120, 10],
            [90, 90, 420, 480, 10],
        ]
        fv_data = np.asarray(fv_data)
        feature_vectors = helpers.FV(fv_data[1:], fv_data[0])

        dist_matrix = decomposer.dist_matrix_func(feature_vectors, decomposer)
        print(dist_matrix.round(2))


def validate_routes():
    dir_name = HG
    instance_name = 'RC1_10_1'
    num_clusters = 6
    inst, converted_inst = helpers.read_instance(dir_name, instance_name)
    dist_matrix_func = DM.v2_2_vectorized
    time_limit = 10

    print(f'instance: {instance_name}')
    # solver = HgsSolverWrapper(time_limit=time_limit)
    # solver = GortoolsSolverWrapper(time_limit=time_limit, min_total=False)
    solver = GortoolsSolverWrapper(time_limit=time_limit, min_total=True)

    decomposer = KMedoidsDecomposer(dist_matrix_func=dist_matrix_func, num_clusters=num_clusters)
    runner = DecompositionRunner(converted_inst, decomposer, solver)
    solution = runner.run(in_parallel=True, num_workers=num_clusters)

    if solution.metrics[METRIC_COST] == float('inf'):
        print('No feasible solution found.')
    else:
        helpers.print_solution(solution, converted_inst)
        instance = dict(
            coords = np.array(inst.coordinates),
            demands = np.array(inst.demands),
            capacity = inst.capacity,
            duration_matrix = np.array(inst.distances),
            time_windows = np.array(list(zip(inst.earliest, inst.latest))),
            service_times = np.array(inst.service_times),
        )
        driving_time = tools.validate_static_solution(instance, solution.routes)
        print(f'driving time excluding waiting time: {driving_time}')


def compare_routes_to_bk(decomp=True):
    '''Compare Euclidean "min total" routes to that of the best known "min driving"
    solution: an indication of how (un)important the time dimension is.
    '''
    n = 10 # instance size
    num_clusters = 10
    dist_matrix_func = DM.euclidean_vectorized
    dir_name = HG
    close_match_threshold = 0.9


    def compare(bk_sol, my_routes, my_cost):
        num_full_match = 0
        num_close_match = 0
        for bk_route in bk_sol.routes:
            for my_route in my_routes:
                # normalized similarity b/t routes
                norm_sim = Levenshtein.ratio(my_route, bk_route)

                if norm_sim > close_match_threshold:
                    num_close_match += 1

                if my_route == bk_route:
                    num_full_match += 1

        num_bk_routes = len(bk_sol.routes)
        num_my_routes = len(my_routes)
        bk_cost = bk_sol.cost
        percent_full_match = round(num_full_match / num_bk_routes * 100, 2)
        percent_close_match = round(num_close_match / num_bk_routes * 100, 2)

        excel_data = {
            KEY_INSTANCE_NAME: [instance_name],
            'num_bk_routes': [num_bk_routes],
            'num_my_routes': [num_my_routes],
            'bk_cost': [bk_cost],
            'my_cost': [my_cost],
            'num_full_match': [num_full_match],
            'percent_full_match': [f'{percent_full_match}%'],
            'num_close_match': [num_close_match],
            'percent_close_match': [f'{percent_close_match}%'],
        }
        df = pd.DataFrame(excel_data)
        sheet_name = instance_name.split('_')[0]
        helpers.write_to_excel(df, f'compare_routes_to_bk{ext}', sheet_name=sheet_name)

        print()
        print(f'instance: {instance_name}')
        print(f'num bk routes: {num_bk_routes}; cost: {bk_cost}')
        print(f'num my routes: {num_my_routes}; cost: {my_cost}')
        print(f'num full match routes: {num_full_match}')
        print(f'% full match bk routes: {percent_full_match}%')
        print(f'num close match routes: {num_close_match}')
        print(f'% close match bk routes: {percent_close_match}%')
        print()


    if decomp:
        time_limit = 10
        ext = f'_decomp_n={n}'
    else:
        time_limit = 30
        ext = f'_n={n}'

    solver = GortoolsSolverWrapper(time_limit=time_limit, min_total=True)
    decomposer = KMedoidsDecomposer(
        dist_matrix_func=dist_matrix_func,
        num_clusters=num_clusters,
        normalize=True,
    )
    json_file_name = 'euclidean_routes' + ext

    if os.path.isfile(json_file_name + '.json'):
        # file exists, read solution from it
        for py_obj in helpers.read_json_gen(json_file_name):
            instance_name = py_obj[KEY_INSTANCE_NAME]
            _, _, bk_sol = helpers.read_instance(dir_name, instance_name, include_bk=True)
            routes = py_obj[KEY_ROUTES]
            cost = py_obj[KEY_COST]
            compare(bk_sol, routes, cost)
    else:
        # file does not exist, solve instances and write solution to json file
        inst_gen = helpers.get_hg_instance_names(n)
        for instance_name in inst_gen:
        # for instance_name in ['C1_2_1']:
            _, converted_inst, bk_sol = helpers.read_instance(dir_name, instance_name, include_bk=True)

            if decomp:
                runner = DecompositionRunner(converted_inst, decomposer, solver)
                solution = runner.run(in_parallel=True, num_workers=num_clusters)
            else:
                solution = solver.solve(converted_inst)

            routes = solution.routes
            cost = solution.metrics[METRIC_COST]

            json_data = {
                KEY_INSTANCE_NAME: instance_name,
                KEY_ROUTES: routes,
                KEY_COST: cost,
            }
            helpers.write_to_json(json_data, json_file_name)

            compare(bk_sol, routes, cost)

            if decomp:
                helpers.sleep(10)


if __name__ == '__main__':
    # test_normalize_matrix()
    # test_read_json()
    # test_read_json_gen()
    # test_get_clusters()
    # summary_fv(to_excel=False, summary_to_excel=False)
    # test_pairwise_distance()
    # test_get_constituents()
    # dist_matrix_to_excel()
    # test_decompose()
    # test_solver()
    # test_framework()
    # plot_instance()
    # plot_dist_matrix()
    # plot_clusters()
    # analyze_overlap_gap_effect()
    # test_antiderivative_vs_quad()
    # test_dist_matrix_qi_2012(use_mock_data=True)
    # validate_routes()
    # calc_omega_factors()
    # calc_omega_factors_per_cluster()
    compare_routes_to_bk()
    # benchmarks_to_sqlite('vrptw.db')
