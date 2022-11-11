import os
import sys
import numpy as np

import hgs.tools as tools
from hgs.baselines.hgs_vrptw import hgspy

instance_name = 'ORTEC-VRPTW-ASYM-0bdff870-d1-n458-k35'
instance = tools.read_vrplib(os.path.join('hgs/instances', f'{instance_name}.txt'))
{
    k: v.shape if isinstance(v, np.ndarray) else None
    for k, v in instance.items()
}

def to_giant_tour_str(routes, with_depot=True):
    return " ".join(map(str, tools.to_giant_tour(routes, with_depot)))

# Trivial initial solution: one route per request
initial_solution = [[i] for i in range(1, instance['coords'].shape[0])]

# Define configuration
config = hgspy.Config(
    seed=1234, 
    nbVeh=-1,
    timeLimit=10,
    useWallClockTime=True,
    initialSolution=to_giant_tour_str(initial_solution),
    useDynamicParameters=True)

# Convert instance so it is suitable for HGS and define params object
params = hgspy.Params(config, **tools.inst_to_vars(instance))

split = hgspy.Split(params)
ls = hgspy.LocalSearch(params)
pop = hgspy.Population(params, split, ls)
algo = hgspy.Genetic(params, split, pop, ls)
algo.run()
best = pop.getBestFound()

# Avoid mixing up cpp and python output
sys.stdout.flush()
print("----- Solution -----")
# Print cost and routes of best solution
print("Cost: ", best.cost)
for i, route in enumerate(best.routes):
    print(f"Route {i}:", route)

