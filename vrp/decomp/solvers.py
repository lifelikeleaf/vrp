from ..third_party.solver.hgs.baselines.hgs_vrptw import hgspy
from ..third_party.solver.hgs import tools
from wurlitzer import pipes

from .decomposition import AbstractSolverWrapper, VRPInstance, VRPSolution
from .logger import logger
from . import helpers
from .constants import *

logger = logger.getChild(__name__)


class HgsSolverWrapper(AbstractSolverWrapper):
    def __init__(self, time_limit=10, cpp_output=False, trivial_init_sol=False) -> None:
        self.time_limit = time_limit
        self.cpp_output = cpp_output
        self.trivial_init_sol = trivial_init_sol


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


    @helpers.log_run_time
    def solve(self, inst: VRPInstance):
        # Calls the HGS solver with default config and passing in a
        # VRP problem instance.

        instance = self.build_instance_for_hgs(inst)

        initial_solution = ''
        if self.trivial_init_sol:
            def to_giant_tour_str(routes, with_depot=True):
                return " ".join(map(str, tools.to_giant_tour(routes, with_depot)))

            # trivial initial solution: one tour per customer
            initial_solution = [[i] for i in range(1, len(instance['coords']))]
            initial_solution = to_giant_tour_str(initial_solution)

        # Capture C-level stdout/stderr
        with pipes() as (out, err):
            config = hgspy.Config(
                nbVeh=-1,
                timeLimit=self.time_limit,
                useWallClockTime=True,
                initialSolution=initial_solution,
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
            logger.debug(f'Output from C++: \n {out.read()}')

        infinity = float('inf')
        if solution is None:
            # no feasible solution found
            metrics = {
                METRIC_COST: infinity,
                METRIC_DISTANCE: infinity,
                METRIC_WAIT_TIME: infinity,
            }
            # when called on decomposed instances, if one single
            # instance finds no feasible solution, aggregated cost will
            # be inf, indicating no feasible solution found for the original
            # instance, regardless of how many routes from other instances
            # may have been collected.
            sol = VRPSolution([], metrics)
            return sol

        # return solution
        # returning solution object alone makes solution.routes = []
        # see bindings.cpp
        metrics = {
            METRIC_COST: solution.cost,
            METRIC_DISTANCE: solution.distance,
            METRIC_WAIT_TIME: solution.waitTime,
        }
        sol = VRPSolution(solution.routes, metrics)
        return sol

