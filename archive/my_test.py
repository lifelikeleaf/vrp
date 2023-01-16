import os
import sys
from pathlib import Path
import numpy as np

# Add parent dir of this current dir to sys.path so python can find the hgs module.
path = Path(os.path.dirname(__file__))
# Only strings should be added to sys.path; all other data types are ignored during import.
PARENT_DIR = str(path.parent.resolve())
sys.path.append(PARENT_DIR)

import vrp.third_party.solver.hgs.tools as tools
from vrp.third_party.solver.hgs.baselines.hgs_vrptw import hgspy
from wurlitzer import pipes


def compute_route_wait_time(route, dist, timew, service_t):
    # route doesn't include depot
    # dist, timew and service_t do include depot = 0

    route_wait_time = 0

    # don't count the wait time at the first stop
    # bc the vehicle could always be dispatched later from the depot
    # so that it arrives exactly at the earliest arrival time of the first stop
    # and it doesn't affect feasibility
    first_stop = route[0]
    first_stop_earliest_start, first_stop_latest_arrival = timew[first_stop]
    current_time = first_stop_earliest_start + service_t[first_stop]

    prev_stop = first_stop
    for stop in route[1:]: # start counting wait time from the 2nd stop
        earliest_arrival, latest_arrival = timew[stop]
        arrival_time = current_time + dist[prev_stop, stop]
        # Wait if we arrive before earliest_arrival
        current_time = max(arrival_time, earliest_arrival)
        wait_time = earliest_arrival - arrival_time
        route_wait_time += max(0, wait_time)
        current_time += service_t[stop]
        prev_stop = stop

    return route_wait_time


instance_name = 'ORTEC-VRPTW-ASYM-0bdff870-d1-n458-k35' #'ORTEC-VRPTW-ASYM-1bdf25a7-d1-n531-k43'
path = os.path.join(PARENT_DIR, 'vrp/third_party/solver/hgs/instances', f'{instance_name}.txt')
instance = tools.read_vrplib(path)

x = {
    k: v.shape if isinstance(v, np.ndarray) else None
    for k, v in instance.items()
}
# output:
# {
#   'is_depot': (459,), 'coords': (459, 2), 'demands': (459,), 'capacity': None, 
#   'time_windows': (459, 2), 'service_times': (459,), 'duration_matrix': (459, 459)
# }

# def to_giant_tour_str(routes, with_depot=True):
#     return " ".join(map(str, tools.to_giant_tour(routes, with_depot)))

# Trivial initial solution: one route per request
# initial_solution = [[i] for i in range(1, instance['coords'].shape[0])]

# Capture C-level stdout/stderr
with pipes() as (out, err):
    # Define configuration
    config = hgspy.Config(
        #seed=1234,
        nbVeh=-1, # nbVehicles = nbClients; line 464 in Params.cpp
        timeLimit=10,
        useWallClockTime=True,)
        #initialSolution=to_giant_tour_str(initial_solution),
        #useDynamicParameters=True)

    # Convert instance so it is suitable for HGS and define params object
    'C++/Python bindings defined in hgs/baselines/hgs_vrptw/src/bindings.cpp'
    'actual C++ implementation in Params.cpp'
    ## tools.inst_to_vars(instance) converts dict of numpy objects obtained from tools.read_vrplib
    ## to a dict of standard python list objects, which are passed in to Params as kwargs
    params = hgspy.Params(config, **tools.inst_to_vars(instance))

    split = hgspy.Split(params)
    ls = hgspy.LocalSearch(params)

    pop = hgspy.Population(params, split, ls)
    algo = hgspy.Genetic(params, split, pop, ls)
    algo.run()
    best = pop.getBestFound()

# print(f'Output from C++: \n {out.read()}')


# Avoid mixing up cpp and python output
## flush any buffered output from cpp (in Population.cpp) and print to stdout/terminal
## before printing the following solution output in python
# sys.stdout.flush() # not needed when using wurlitzer to capture C-level output


'''
see tools.validate_route_time_windows() for how to calculate waiting time (line 138)
validate_static_solution
- validate_all_customers_visited
- validate_route_capacity
- validate_route_time_windows

also see `struct CostSol` in hgs/baselines/hgs_vrptw/include/Individual.h
- hgs/baselines/hgs_vrptw/src/bindings.cpp (line 203)
// added by Leif: exposing distance and waitTime from CPP Individual object to Python
.def_property_readonly("distance", [](Individual &indiv) { return indiv.myCostSol.distance; })
.def_property_readonly("waitTime", [](Individual &indiv) { return indiv.myCostSol.waitTime; })
// added by Leif
'''
print("\n----- Solution -----")
print("Cost: ", best.cost) # best.cost == driving_time
driving_time = tools.compute_solution_driving_time(instance, best.routes)
print("Driving time excluding waiting time: ", driving_time)
validated_driving_time = tools.validate_static_solution(instance, best.routes)
print("Validated driving time excluding waiting time: ", validated_driving_time)

# print(type(best)) # <class 'hgspy.Individual'>
print('distance:', best.distance)
print('wait time:', best.waitTime) # best.waitTime == total_wait_time

total_wait_time = 0
for route in best.routes:
    # don't count the wait time at the first stop
    # bc the vehicle could always be dispatched later from the depot
    # so that it arrives exactly at the earliest arrival time of the first stop
    # and it doesn't affect feasibility
    total_wait_time += compute_route_wait_time(route, instance['duration_matrix'], instance['time_windows'], instance['service_times'])

print('calculated total wait time:', total_wait_time)
# for i, route in enumerate(best.routes):
#     print(f"Route {i}:", route)

