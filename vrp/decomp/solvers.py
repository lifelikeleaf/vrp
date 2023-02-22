from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from ..third_party.solver.hgs.baselines.hgs_vrptw import hgspy
from ..third_party.solver.hgs import tools
from wurlitzer import pipes

from .decomposition import AbstractSolverWrapper, VRPInstance, VRPSolution
from .logger import logger
from . import helpers
from .constants import *

logger = logger.getChild(__name__)
infinity = float('inf')


class HgsSolverWrapper(AbstractSolverWrapper):
    '''Wraps HGS VRPTW solver.'''

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
    def solve(self, inst: VRPInstance) -> VRPSolution:
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

        if solution is None:
            # no feasible solution found
            metrics = {
                METRIC_COST: infinity,
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
            METRIC_WAIT_TIME: solution.waitTime,
        }
        sol = VRPSolution(solution.routes, metrics)
        return sol


class GortoolsSolverWrapper(AbstractSolverWrapper):
    '''Wraps Google OR-Tools VRPTW solver. This is primarily for
    experiments where wait time is included in the objective function because
    HGS solver doesn't include wait time in its objective function.
    '''

    def __init__(self, time_limit=10, wait_time_in_obj_func=True) -> None:
        self.time_limit = time_limit
        # this flag should be kept True for actual experiments,
        # it's used here mainly for testing purposes.
        self.wait_time_in_obj_func = wait_time_in_obj_func


    def build_data_for_gortools(self, inst: VRPInstance):
        demand = []
        time_window = []
        service_time = []
        time_matrix = []
        for i in range(len(inst.nodes)):
            node = inst.nodes[i]
            demand.append(node.demand)
            time_window.append((node.start_time, node.end_time))
            service_time.append(node.service_time)
            time_matrix.append(node.distances)

        return dict(
            demand = demand,
            vehicle_cap = inst.vehicle_capacity,
            time_window = time_window,
            service_time = service_time,
            time_matrix = time_matrix,
        )


    def get_time_callback(self, manager, time_matrix, service_time):

        def time_callback(from_index, to_index):
            """
            Returns the driving time between the two nodes + service time
            at end node.
            """
            # Convert from ortools internal routing variable index to
            # time matrix node index.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            # NOTE: add service time so the time dimension can correctly
            # keep track of cumulative time at each node.
            return time_matrix[from_node][to_node] + service_time[to_node]

        return time_callback


    def get_demand_callback(self, manager, demand):

        def demand_callback(index):
            """Returns the demand of the node."""
            node = manager.IndexToNode(index)
            return demand[node]

        return demand_callback


    def print_solution(self, data, manager, routing, solution):
        '''For debugging purposes.'''
        tw = data['time_window']
        st = data['service_time']
        print(f'Objective: {solution.ObjectiveValue()}')
        time_dimension = routing.GetDimensionOrDie(DIMENSION_TIME)
        total_time = 0
        print(f'num vehicles: {routing.vehicles()}')
        for vehicle_id in range(routing.vehicles()):
            if routing.IsVehicleUsed(solution, vehicle_id):
                # here start node is always depot
                index = routing.Start(vehicle_id)
                node_idx = manager.IndexToNode(index)
                time_var = time_dimension.CumulVar(index)
                route_start = solution.Min(time_var)
                plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
                while not routing.IsEnd(index):
                    time_var = time_dimension.CumulVar(index)
                    # sw: solution window
                    # tw: time window
                    min_sw = solution.Min(time_var)
                    max_sw = solution.Max(time_var)
                    # assert min_sw == max_sw
                    min_tw = tw[node_idx][0] + st[node_idx]
                    max_tw = tw[node_idx][1] + st[node_idx]
                    if min_sw == min_tw:
                        plan_output += '***'
                    plan_output += '{0} sw({1},{2}) tw[{3},{4}] -> '.format(
                        node_idx,
                        min_sw,
                        max_sw,
                        min_tw,
                        max_tw,
                    )
                    index = routing.Next(solution, index)
                    node_idx = manager.IndexToNode(index)

                time_var = time_dimension.CumulVar(index)
                plan_output += '{0} sw({1},{2}) tw[{3},{4}]\n'.format(
                    node_idx,
                    solution.Min(time_var),
                    solution.Max(time_var),
                    tw[node_idx][0] + st[node_idx],
                    tw[node_idx][1] + st[node_idx],
                )
                route_end = solution.Min(time_var)
                route_time = route_end - route_start
                plan_output += 'Time of the route: {}min\n'.format(route_time)
                print(plan_output)
                total_time += route_time
        print('Total time of all routes: {}min'.format(total_time))


    @helpers.log_run_time
    def solve(self, inst: VRPInstance) -> VRPSolution:
        depot_node_idx = 0
        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(
            len(inst.nodes), # num_nodes
            inst.extra['num_vehicles'],
            depot_node_idx
        )

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        data = self.build_data_for_gortools(inst)
        time_matrix = data['time_matrix']
        service_time = data['service_time']
        demand = data['demand']
        vehicle_cap = data['vehicle_cap']

        demand_callback = self.get_demand_callback(manager, demand)
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

        # Create capacity dimension for tracking cumulative vehicle capacity.
        # here all vehicles have the same capacity;
        # if vehiclces have diff capacities, use AddDimensionWithVehicleCapacity,
        # which takes a vector of capacities.
        routing.AddDimension(
            demand_callback_index,
            0,                          # no slack for capacity
            vehicle_cap,                # max vehicle capacity
            True,                       # fix starting cumulative var to zero
            DIMENSION_CAPACITY,         # name of dimension
        )

        time_callback = self.get_time_callback(manager, time_matrix, service_time)
        time_callback_index = routing.RegisterTransitCallback(time_callback)

        # Create time dimension for tracking cumulative time.
        depot_node = inst.nodes[depot_node_idx]
        # upper bound for wait time and vehicle travel time
        depot_tw_size = depot_node.end_time - depot_node.start_time
        routing.AddDimension(
            time_callback_index,
            depot_tw_size,          # allow wait time (slack)
            depot_tw_size,          # max time per vehicle
            False,                  # Don't fix starting cumulative var to zero.
                                    # In VRPTW a vehicle might not start at time 0.
            DIMENSION_TIME,         # name of dimension
        )
        time_dimension = routing.GetDimensionOrDie(DIMENSION_TIME)

        if self.wait_time_in_obj_func:
            logger.info('Considers total time (wait time included)')
            time_dimension.SetSpanCostCoefficientForAllVehicles(1)
        else:
            logger.info('Only considers driving time')
            routing.SetArcCostEvaluatorOfAllVehicles(time_callback_index)

        # Add hard time window constraints, shifted by service time so that
        # for early arrival, wait time (slack in ortools term) can be
        # correctly calculated; and late arrival is only truly late if vehicle
        # can't arrive and finish service before the end time, bc time_callback
        # already includes service time
        for node_idx, time_window in enumerate(data['time_window']):
            index = manager.NodeToIndex(node_idx)
            time_dimension.CumulVar(index).SetRange(
                time_window[0] + service_time[node_idx],
                time_window[1] + service_time[node_idx],
            )

        # secondary objectives to try to reduce wait time even if OF
        # only considers driving time.
        # if OF already includes wait time, these secondary objectives are
        # not that important.
        if not self.wait_time_in_obj_func:
            for vehicle_id in range(routing.vehicles()):
                # leave the depot as late as possible
                routing.AddVariableMaximizedByFinalizer(
                    time_dimension.CumulVar(routing.Start(vehicle_id)))

                # return to depot as early as possible
                routing.AddVariableMinimizedByFinalizer(
                    time_dimension.CumulVar(routing.End(vehicle_id)))

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_INSERTION)

        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.seconds = self.time_limit

        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)
        logger.info(f'status: {routing.status()} = {SOLVER_STATUS[routing.status()]}')

        if solution:
            # exclude service time from solution cost
            cost = solution.ObjectiveValue() - sum(service_time)
            metrics = {
                METRIC_COST: cost,
                # wait time is expected to be included in objective function,
                # so wait time alone is not calculated here, but can be
                # calculated downstream based on routes. Include it here in
                # `metrics` for compatibility with existing ExperimentRunner,
                # where it's expected.
                METRIC_WAIT_TIME: 0,
            }
            routes = []
            route_starts = []
            for vehicle_id in range(routing.vehicles()):
                if routing.IsVehicleUsed(solution, vehicle_id):
                    route = []
                    # here starting node is always depot
                    index = routing.Start(vehicle_id)
                    time_var = time_dimension.CumulVar(index)
                    # departure time from the depot for this route
                    route_start = solution.Max(time_var)
                    route_starts.append(route_start)
                    # depot is skipped - not included in the returned route list
                    index = routing.Next(solution, index)
                    while not routing.IsEnd(index):
                        node_idx = manager.IndexToNode(index)
                        route.append(node_idx)
                        index = routing.Next(solution, index)

                    routes.append(route)

            sol = VRPSolution(routes, metrics)
            # for validation purposes
            sol.extra = {
                EXTRA_ROUTE_STARTS: route_starts,
            }
        else:
            # no feasible solution found
            metrics = {
                METRIC_COST: infinity,
                METRIC_WAIT_TIME: infinity,
            }
            sol = VRPSolution([], metrics)

        return sol

