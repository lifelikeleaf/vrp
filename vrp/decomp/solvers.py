from ..third_party.solver.hgs.baselines.hgs_vrptw import hgspy
from wurlitzer import pipes

from .decomposition import AbstractSolverWrapper, VRPInstance

class HgsSolverWrapper(AbstractSolverWrapper):
    def __init__(self, time_limit=10, cpp_output=False) -> None:
        self.time_limit = time_limit
        self.cpp_output = cpp_output


    def build_instance_for_hgs(self, inst: VRPInstance):
        """Converts a `VRPInstance` to argument types
        accepted by `hgspy.Params`:

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
        coords = []
        demands = []
        time_windows = []
        service_durations = []
        duration_matrix = []
        for i in range(len(inst.nodes)):
            node = inst.nodes[i]
            coords.append((node.x_coord, node.y_coord))
            demands.append(node.demand)
            time_windows.append((node.start_time, node.end_time))
            service_durations.append(node.service_time)
            duration_matrix.append(node.distances)

        return dict(
            coords = coords,
            demands = demands,
            vehicle_cap = inst.vehicle_capacity,
            time_windows = time_windows,
            service_durations = service_durations,
            duration_matrix = duration_matrix,
            # not used but required by hgspy.Params
            release_times=[0] * len(inst.nodes),
        )


    def solve(self, inst: VRPInstance):
        # Calls the HGS solver with default config and passing in a
        # VRP problem instance.

        instance = self.build_instance_for_hgs(inst)

        # Capture C-level stdout/stderr
        with pipes() as (out, err):
            config = hgspy.Config(
                nbVeh=-1,
                timeLimit=self.time_limit,
                useWallClockTime=True
            )

            params = hgspy.Params(config, **instance)
            split = hgspy.Split(params)
            ls = hgspy.LocalSearch(params)
            pop = hgspy.Population(params, split, ls)
            algo = hgspy.Genetic(params, split, pop, ls)
            algo.run()
            # get the best found solution (type Individual) from
            # the population pool
            solution = pop.getBestFound()

        if self.cpp_output:
            print(f'Output from C++: \n {out.read()}')

        # return solution
        # for some reason returning solution alone makes solution.routes = []
        return solution.cost, solution.routes

