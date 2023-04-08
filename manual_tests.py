# Author: Xu Ye <kan.ye@tum.de>

"""Quick and dirty manual tests. Some functions might not be up to date."""

import os
from collections import defaultdict
import itertools
from scipy.integrate import quad

import cvrplib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from vrp.third_party.solver.hgs import tools
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


def calc_omega_factor(inst, cluster=None): # (cvrplib.Instance.VRPTW, list)
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
    return omega_factor


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
            omega_factor = calc_omega_factor(inst)
            print(f'{instance_name} omega factor: {round(omega_factor, 2)}%')
        print()


def analyze_overlap_gap_effect():
    dir_name = HG
    '''n=200'''
    # FOCUS_GROUP_C1 = ['C1_2_1', 'C1_2_4', 'C1_2_8']
    # FOCUS_GROUP_C2 = ['C2_2_1', 'C2_2_4', 'C2_2_8']
    # FOCUS_GROUP_R1 = ['R1_2_1', 'R1_2_4', 'R1_2_8']
    # FOCUS_GROUP_R2 = ['R2_2_1', 'R2_2_4', 'R2_2_8']
    # FOCUS_GROUP_RC1 = ['RC1_2_1', 'RC1_2_4', 'RC1_2_8']
    # FOCUS_GROUP_RC2 = ['RC2_2_1', 'RC2_2_4', 'RC2_2_8']
    input = {
        # 'test': ['RC2_10_8'],

        ## n=1000
        # 'focus_C1': FOCUS_GROUP_C1,
        # 'focus_C2': FOCUS_GROUP_C2,
        # 'focus_R1': FOCUS_GROUP_R1,
        # 'focus_R2': FOCUS_GROUP_R2,
        # 'focus_RC1': FOCUS_GROUP_RC1,
        # 'focus_RC2': FOCUS_GROUP_RC2,

        ## n=1000
        'C1': C1_10,
        'C2': C2_10,
        'R1': R1_10,
        'R2': R2_10,
        'RC1': RC1_10,
        'RC2': RC2_10,
    }
    decomposer = KMedoidsDecomposer(dist_matrix_func=None, normalize=False)
    for val in input.values():
        for instance_name in val:
            inst, converted_inst = helpers.read_instance(dir_name, instance_name)
            feature_vectors = helpers.build_feature_vectors(converted_inst)
            fv = feature_vectors.data
            x, y, start, end = np.asarray(list(zip(*fv)))
            planning_horizon = end.max() - start.min()

            constituents_matrix = DM._get_constituents_vectorized(fv, decomposer, as_matrix=False)
            df = pd.DataFrame({key: val.flatten() for key, val in constituents_matrix.items()})
            overlap_count = df.loc[df['overlap'] > 0, ['overlap']].count()['overlap']
            gap_count = df.loc[df['gap'] > 0, ['gap']].count()['gap']
            overlap_p = overlap_count / df['overlap'].count()
            overlap_p = round(overlap_p * 100, 2)
            gap_p = gap_count / df['gap'].count()
            gap_p = round(gap_p * 100, 2)
            print(f'\n{instance_name}')
            # % of node pairs in an instance with _any_ overlap/gap
            print(f'overlap % = {overlap_p}%')
            print(f'gap % = {gap_p}%')

            # pairwise size of overlap/gap compared to the planning horizon
            overlaps = constituents_matrix['overlap']
            # overlaps = overlaps[overlaps != 0]
            gaps = constituents_matrix['gap']
            # gaps = gaps[gaps != 0]
            overlaps.sort()
            gaps.sort()

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5.4))
            y_max = planning_horizon * 1.05
            text_x = len(overlaps) * 0.4 # same for gaps when 0s are not filtered out
            text_y = planning_horizon * 0.2

            ax1.plot(overlaps)
            ax1.fill_between(np.arange(len(overlaps)), overlaps)
            # Use double curly braces to "escape" literal curly braces that only LaTeX understands
            ax1.text(text_x, text_y, rf'$\sum_{{overlaps}} = {overlaps.sum():,}$')
            ax1.set_ylim(top=y_max)

            ax1.set_title('Amount of Overlaps')
            # draw a horizontal line
            ax1.axhline(planning_horizon, ls='--', c='red')
            arrow_head_x = len(overlaps) * 0.3 # middle shifted to the left by 20%
            arrow_head_y = planning_horizon
            arrow_tail_x = len(overlaps) * 0.45
            arrow_tail_y = planning_horizon * 0.8
            ax1.annotate('planning horizon', xy=(arrow_head_x, arrow_head_y),
                        xytext=(arrow_tail_x, arrow_tail_y),
                        arrowprops=dict(facecolor='black', shrink=0.05))

            ax2.plot(gaps)
            ax2.fill_between(np.arange(len(gaps)), gaps)
            ax2.text(text_x, text_y, rf'$\sum_{{gaps}}$ = {gaps.sum():,}')
            ax2.set_ylim(top=y_max)

            ax2.set_title('Amount of Gaps')
            ax2.axhline(planning_horizon, ls='--', c='red')
            fig.suptitle(f'{inst.name} ({overlap_p}% of pairs have overlaps, {gap_p}% gaps)') #, fontweight='bold')
            # plt.show()
            path = os.path.join(TEST_DIR, 'plot', 'overlap_gap_effect (1k)')
            helpers.make_dirs(path)
            fname = os.path.join(path, instance_name)
            fig.savefig(fname)


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
    instance_name = 'C2_10_1'
    num_clusters = 4
    _, converted_inst = helpers.read_instance(dir_name, instance_name)
    dist_matrix_func = DM.euclidean_vectorized
    time_limit = 10

    print(f'instance: {instance_name}')
    # solver = HgsSolverWrapper(time_limit=time_limit)
    # solver = GortoolsSolverWrapper(time_limit=time_limit, min_total=False)
    solver = GortoolsSolverWrapper(time_limit=time_limit, min_total=True)

    decomposer = KMedoidsDecomposer(
        dist_matrix_func=dist_matrix_func,
        num_clusters=num_clusters,
        use_overlap=True,
        use_gap=True,
        normalize=True
    )
    runner = DecompositionRunner(converted_inst, decomposer, solver)
    solution = runner.run(in_parallel=True, num_workers=num_clusters)
    if solution.metrics[METRIC_COST] == float('inf'):
        print('No feasible solution found.')
    else:
        helpers.print_solution(solution, converted_inst, verbose=True)


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
    calc_omega_factors()
