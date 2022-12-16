import os

import hgs.tools as tools
from hgs.baselines.hgs_vrptw import hgspy
from wurlitzer import pipes

import cvrplib

HG = 'Vrp-Set-HG'
SOLOMON = 'Vrp-Set-Solomon'

def call_hgs(instance, cpp_output=False):
    """Calls the HGS solver with default config and passing in problem instance data"""

    # Capture C-level stdout/stderr
    with pipes() as (out, err):
        config = hgspy.Config(
            nbVeh=-1,
            timeLimit=10,
            useWallClockTime=True
        )

        params = hgspy.Params(config, **instance)
        split = hgspy.Split(params)
        ls = hgspy.LocalSearch(params)
        pop = hgspy.Population(params, split, ls)
        algo = hgspy.Genetic(params, split, pop, ls)
        algo.run()
        # get the best found solution (type Individual) from the population pool
        solution = pop.getBestFound()

    if cpp_output:
        print(f'Output from C++: \n {out.read()}')

    # return solution
    # for some reason returning solution alone makes solution.routes = []
    return solution.cost, solution.routes


def build_instance_for_hgs(inst: cvrplib.Instance.VRPTW):
    """Converts cvrplib format to argument types accepted by hgspy.Params

    hgspy.Params(
        config: hgspy.Config,
        coords: List[Tuple[int, int]],
        demands: List[int],
        vehicle_cap: int,
        time_windows: List[Tuple[int, int]],
        service_durations: List[int],
        duration_matrix: List[List[int]],
        release_times: List[int]
    )
    """
    return dict(
        coords=[(x, y) for x, y in inst.coordinates],
        demands=inst.demands,
        vehicle_cap=inst.capacity,
        time_windows=[(inst.earliest[i], inst.latest[i]) for i in range(len(inst.earliest))],
        service_durations=inst.service_times,
        duration_matrix=inst.distances,
        release_times=[0] * len(inst.coordinates),
    )


if __name__ == "__main__":
    # ORTEC benchmarks
    # instance_name = 'ORTEC-VRPTW-ASYM-0bdff870-d1-n458-k35'
    # inst = tools.read_vrplib(os.path.join('hgs/instances', f'{instance_name}.txt'))
    # # converts dict of numpy objects to a dict of standard python list objects suitable for HGS consumption
    # instance = tools.inst_to_vars(inst)

    # CVRPLIB benchmarks
    instance_name = 'C206'
    inst = cvrplib.read(f'CVRPLIB/{SOLOMON}/{instance_name}.txt')
    # # print([k for k, v in vars(inst).items()])
    instance = build_instance_for_hgs(inst)

    # solution = call_hgs(instance)
    # cost, routes = solution.cost, solution.routes
    cost, routes = call_hgs(instance)

    print("\n----- Solution -----")
    print("Cost: ", cost)
    for i, route in enumerate(routes):
        print(f"Route {i}:", route)
