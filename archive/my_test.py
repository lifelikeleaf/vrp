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

instance_name = 'ORTEC-VRPTW-ASYM-0bdff870-d1-n458-k35' #'ORTEC-VRPTW-ASYM-1bdf25a7-d1-n531-k43'
instance = tools.read_vrplib(os.path.join(PARENT_DIR, 'vrp/third_party/solver/hgs/instances', f'{instance_name}.txt'))
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
    ## C++/Python bindings defined in hgs/baselines/hgs_vrptw/src/bindings.cpp
    ## actual C++ implementation in Params.cpp
    ## tools.inst_to_vars(instance) converts dict of numpy objects obtained from tools.read_vrplib
    ## to a dict of standard python list objects, which are passed in to Params as kwargs
    params = hgspy.Params(config, **tools.inst_to_vars(instance))

    split = hgspy.Split(params)
    ls = hgspy.LocalSearch(params)

    pop = hgspy.Population(params, split, ls)
    algo = hgspy.Genetic(params, split, pop, ls)
    algo.run()
    best = pop.getBestFound()

print(f'Output from C++: \n {out.read()}')

print(type(best))

# Avoid mixing up cpp and python output
## flush any buffered output from cpp (in Population.cpp) and print to stdout/terminal
## before printing the following solution output in python
# sys.stdout.flush() # not needed when using wurlitzer to capture C-level output
print("\n----- Solution -----")
# Print cost and routes of best solution
print("Cost: ", best.cost) # best.cost == driving_time
driving_time = tools.compute_solution_driving_time(instance, best.routes)
# see tools.validate_route_time_windows() for how to include waiting time: line 138
# also see struct CostSol in hgs/baselines/hgs_vrptw/include/Individual.h
print("Driving time excluding waiting time: ", driving_time)
for i, route in enumerate(best.routes):
    print(f"Route {i}:", route)

