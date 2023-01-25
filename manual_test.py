import os
import vrp.decomp.helpers as helpers
import cvrplib
from vrp.decomp.solvers import HgsSolverWrapper
from vrp.decomp.decomposers import KMedoidsDecomposer
from vrp.decomp.decomposition import DecompositionRunner
from vrp.decomp.constants import *
import vrp.decomp.distance_matrices as DM
import numpy as np
import pandas as pd


TEST_DIR = 'Test'


def compute_route_wait_time(route, inst, verbose=False):
    # `route` doesn't include depot; `inst` does include depot = 0

    depot = 0
    first_stop = route[0]
    route_wait_time = 0
    route_dist = inst.distances[depot][first_stop]

    # don't count the wait time at the first stop
    # bc the vehicle could always be dispatched later from the depot
    # so as to not incur any wait time and it doesn't affect feasibility
    # NOTE: can't simply start at the earliest start time of the first node,
    # bc the earliest start time of the first node could be 0 and we can't
    # start at the first stop at time 0, bc we have to first travel from
    # deopt to the first stop
    depot_earliest_departure_time = inst.earliest[depot] + inst.service_times[depot]
    # earliest possible arrival time at the first stop
    # = earliest possible time to leave the depot + travel time from depot to the first stop
    travel_time = inst.distances[depot][first_stop]
    arrival_time = depot_earliest_departure_time + travel_time
    # logical earliest start time at the first stop
    # = the later of arrival time and TW earliest start time
    tw_earliest_start = inst.earliest[first_stop]
    logical_earliest_start = max(arrival_time, tw_earliest_start)
    # departure time from the first node
    service_time = inst.service_times[first_stop]
    departure_time = logical_earliest_start + service_time

    if verbose:
        print(f'first stop = {first_stop}', end=', ')
        print(f'TW earliest start = {tw_earliest_start}', end=', ')
        print(f'travel time from depot to first stop = {travel_time}', end=', ')
        print(f'logical earliest start = {logical_earliest_start}', end=', ')
        print(f'service time = {service_time}', end=', ')
        print(f'departure time = {departure_time}')
        print()

    prev_stop = first_stop
    for stop in route[1:]: # start counting wait time from the 2nd stop
        travel_time = inst.distances[prev_stop][stop]
        route_dist += travel_time
        tw_earliest_start = inst.earliest[stop]
        arrival_time = departure_time + travel_time
        # Wait if we arrive before earliest start
        wait_time = max(0, tw_earliest_start - arrival_time)
        route_wait_time += wait_time
        logical_earliest_start = arrival_time + wait_time
        service_time = inst.service_times[stop]
        departure_time = logical_earliest_start + service_time

        if verbose:
            print(f'travel time from {prev_stop} to {stop} = {travel_time}', end=', ')
            print(f'coords for prev stop {prev_stop} = {inst.coordinates[prev_stop]}', end=', ')
            print(f'coords for stop {stop} = {inst.coordinates[stop]}')
            print(f'stop = {stop}', end=', ')
            print(f'arrival time = {arrival_time}', end=', ')
            print(f'TW earliest start = {tw_earliest_start}', end=', ')
            print(f'wait time={wait_time}', end=', ')
            print(f'logical earliest start = {logical_earliest_start}', end=', ')
            print(f'service time = {service_time}', end=', ')
            print(f'departure time = {departure_time}')
            print()

        prev_stop = stop

    route_dist += inst.distances[prev_stop][depot]

    return route_wait_time, route_dist


def list_benchmark_names():
    benchmark = cvrplib.list_names(low=100, high=100, vrp_type='vrptw')
    print(benchmark)


def test_normalize_fv():
    fv = [
        [1,2,3,4],
        [4,3,2,1],
        [5,6,7,8],
    ]
    fv = helpers.normalize_feature_vectors(fv)
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
    file_name = 'k_medoids'
    data = helpers.read_json(file_name)
    print(len(data))


def test_get_clusters():
    decomposer = KMedoidsDecomposer(dist_matrix_func=DM.v1)
    labels = [0, 1, 1, 3, 0, 0, 3, 3, 3, 1, 1]
    clusters = decomposer.get_clusters(labels)
    assert clusters == [[1, 5, 6], [2, 3, 10, 11], [4, 7, 8, 9]]
    print(clusters)


def read_instance(dir_name, instance_name):
    file_name = os.path.join(CVRPLIB, dir_name, instance_name)
    inst = cvrplib.read(instance_path=f'{file_name}.txt')
    converted_inst = helpers.convert_cvrplib_to_vrp_instance(inst)
    return inst, converted_inst


def summary_fv(to_excel=False, summary_to_excel=False):
    dir_name = SOLOMON
    names = ['C101']
    # names.extend(FOCUS_GROUP_C1)
    # names.extend(FOCUS_GROUP_RC2)
    for instance_name in names:
        _, converted_inst = read_instance(dir_name, instance_name)

        decomposer = KMedoidsDecomposer(None, use_gap=True)
        feature_vectors = decomposer.build_feature_vectors(converted_inst, use_tw=True, normalize=False)
        df = pd.DataFrame(feature_vectors.data, columns=['x', 'y', 'start', 'end'])
        df['time_duration'] = df['end'] - df['start']
        df_desc = df.describe()
        df_desc.drop(index=['count', '25%', '50%', '75%'], inplace=True)
        print(f'\n{instance_name}')
        print(df_desc)
        if to_excel:
            helpers.make_dirs(TEST_DIR)
            file_name = os.path.join(TEST_DIR, f'FV')
            helpers.write_to_excel(df, file_name=file_name, sheet_name=instance_name, overlay=False)
            if summary_to_excel:
                file_name = os.path.join(TEST_DIR, f'FV_summary')
                helpers.write_to_excel(df_desc, file_name=file_name, sheet_name=instance_name, overlay=False, index=True)


def test_pairwise_distance():
    dir_name = SOLOMON
    instance_name = 'C101'
    gap = False
    norm = False

    _, converted_inst = read_instance(dir_name, instance_name)
    decomposer = KMedoidsDecomposer(dist_matrix_func=None, use_tw=True, use_gap=gap)
    feature_vectors = decomposer.build_feature_vectors(converted_inst, use_tw=True, normalize=norm)
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


def dist_matrix_to_excel():
    dir_name = SOLOMON
    instance_name = 'C101'
    instance_size = 100
    dist_matrix_func = DM.v4
    file_name = os.path.join(TEST_DIR, f'DM_{dist_matrix_func.__name__}')
    gap = False
    norm = False

    _, converted_inst = read_instance(dir_name, instance_name)
    decomposer = KMedoidsDecomposer(dist_matrix_func=dist_matrix_func, use_tw=True, use_gap=gap)
    feature_vectors = decomposer.build_feature_vectors(converted_inst, use_tw=True, normalize=norm)
    df = dist_matrix = decomposer.dist_matrix_func(feature_vectors, decomposer)
    df_desc = df.describe()
    '''
    max_tw_width (by construct) and euclidean_dist (by nature) would only be 0 with itself, so could make sense to remove 0s;
    but overlap and gap could legitimately be 0 b/t node pairs, albeit not very common;
    and overlap with itself is always the full tw_width of itself,
    whereas gap with itself is always 0
    '''
    # df_no_zeros = df.replace(0, np.NaN)
    # df_desc = df_no_zeros.describe()
    df_desc.drop(index=['count', '25%', '50%', '75%'], inplace=True)
    print(f'\n{instance_name}')
    print(df_desc.round(2))
    # dist_matrix = np.array(dist_matrix)
    # customer_ids = [i for i in range(1, instance_size + 1)]
    # df = pd.DataFrame(dist_matrix, columns=customer_ids, index=customer_ids)
    ext = ''
    ext += '_gap' if gap else ''
    ext += '_norm' if norm else ''
    helpers.make_dirs(TEST_DIR)
    helpers.write_to_excel(df, file_name=file_name, sheet_name=f'{instance_name}{ext}', overlay=False, index=True)
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


def test_decompose():
    dir_name = SOLOMON
    instance_name = 'C101'
    inst, converted_inst = read_instance(dir_name, instance_name)

    decomposer = KMedoidsDecomposer(dist_matrix_func=DM.v1, use_tw=True)
    decomposer.num_clusters = 5
    decomposer.decompose(converted_inst)


def test_framework():
    dir_name = SOLOMON
    instance_name = 'RC206'
    num_clusters = 2
    inst, converted_inst = read_instance(dir_name, instance_name)

    solver = HgsSolverWrapper(time_limit=5)
    # solution = solver.solve(converted_inst)

    decomposer = KMedoidsDecomposer(dist_matrix_func=DM.v1, num_clusters=num_clusters, use_tw=True, use_gap=True)
    runner = DecompositionRunner(converted_inst, decomposer, solver)
    solution = runner.run(in_parallel=True, num_workers=num_clusters)

    cost = solution.metrics[METRIC_COST]
    dist = solution.metrics[METRIC_DISTANCE]
    wait_time = solution.metrics[METRIC_WAIT_TIME]
    routes = solution.routes
    extra = solution.extra


    total_wait_time = 0
    total_dist = 0
    for route in routes:
        route_wait_time, route_dist = compute_route_wait_time(route, inst, verbose=False)
        total_wait_time += route_wait_time
        total_dist += route_dist


    print(f'cost: {cost}')
    print(f'distance: {dist}')
    print(f'computed total distance: {total_dist}')
    print(f'wait time: {wait_time}')
    print(f'computed total wait time: {total_wait_time}')
    print(f'extra: {extra}')
    print(f'routes: {routes}')


if __name__ == '__main__':
    # test_normalize_matrix()
    # test_get_clusters()
    # summary_fv(to_excel=True, summary_to_excel=False)
    # test_pairwise_distance()
    dist_matrix_to_excel()
    # test_decompose()
    # test_framework()

