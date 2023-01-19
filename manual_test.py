import os
import vrp.decomp.helpers as helpers
import cvrplib
from vrp.decomp.solvers import HgsSolverWrapper
from vrp.decomp.decomposers import KMedoidsDecomposer
from vrp.decomp.decomposition import DecompositionRunner
from vrp.decomp.constants import *


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


def test_normalize():
    fv = [
        [1,2,3,4],
        [4,3,2,1],
        [5,6,7,8],
    ]
    fv = helpers.normalize_feature_vectors(fv)
    print(fv)


def test_read_json():
    file_name = 'k_medoids'
    data = helpers.read_json(file_name)
    print(len(data))


def test_framework():
    dir_name = SOLOMON
    instance_name = 'C105'
    num_clusters = 3
    file_name = os.path.join(CVRPLIB, dir_name, instance_name)
    inst = cvrplib.read(instance_path=f'{file_name}.txt')
    converted_inst = helpers.convert_cvrplib_to_vrp_instance(inst)

    solver = HgsSolverWrapper(time_limit=5)
    decomposer = KMedoidsDecomposer(num_clusters=num_clusters)
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
    test_framework()

