"""A mini framework for decomposition of VRP and VRPTW problems.

This work only considers a single depot and homogeneous fleet, but users could
extend it to include multi-depot and/or heterogeneous fleet problems as well.

By CVRPLIB convention, the depot always has index 0 and customers have
indices/IDs from 1 to n. The depot should always be included in decomposed
subproblems, but should never be included in the decomposition process.
That is, only customers should be clustered, but the depot should be
reincorporated to form independent subproblems that are smaller VRP instances.

A clarification of terminology:
- A cluster is a subset of customers.
- A subproblem is a cluster + the depot.

"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from copy import deepcopy
from multiprocessing import Pool

from .logger import logger

logger = logger.getChild(__name__)

@dataclass
class Node:
    """Encapsulates each customer/depot node and its attributes.

    Parameters
    ----------
    x_coord: float
        A node's x coordinate.
    y_coord: float
        A node's y coordinate.
    demand: int
        A node's demand.
    distances: list[float]
        Distances from this node to any other node.

    Optional:
        start_time: int
            Earliest service start time for this node.
        end_time: int
            Latest service start time for this node.
        service_time: int
            Service time for this node.

    """
    x_coord: float
    y_coord: float
    demand: int
    distances: list[float]

    ## fields below only exist for VRPTW

    # time window
    start_time: int = None
    end_time: int = None

    service_time: int = None


    def get_decomposed_distances(self, cluster: list):
        """Gets a decomposed distance list based on the provided `cluster` list.
        Only includes distances from this node to other nodes if those nodes
        are present in the cluster list. Distance to the depot is always
        included.

        Parameters
        ----------
        cluster: list
            A list of customer IDs/indices that represents a cluster
            of customers.

        Returns
        -------
        distances: list
            Distances from this node to other nodes only if they are present
            in the `cluster` list. Distance to the depot is always included.

        """
        # distance to the depot is always included
        depot = 0
        distances = [self.distances[depot]]

        for customer_id in cluster:
            distances.append(self.distances[customer_id])

        return distances


"""Objects which are instances of user-defined classes are hashable by default.
But Data Classes require some special handling.

https://docs.python.org/3/library/dataclasses.html#module-contents
If eq and frozen are both true, by default dataclass() will
generate a __hash__() method for you. If eq is true and frozen
is false, __hash__() will be set to None, marking it unhashable
(which it is, since it is mutable).

If eq is false, __hash__() will be left untouched meaning the
__hash__() method of the superclass will be used (if the superclass
is object, this means it will fall back to id-based hashing).
"""
@dataclass(eq=False)
class VRPInstance:
    """A VRP problem instance representation more intuitive for decomposition.
    Other benchmark instance readers (e.g. cvrplib) tend to represent a VRP
    instance on a 'column' basis (i.e. keyed on attributes). Here a VRP instance
    is represeneted on a 'row' basis, with customers/depot and their attributes
    encapsulated as a list of `Node` objects. This class only explicitly
    outlines the bare minimum data fields: `nodes` and `vehicle_capacity`.
    The user should use the `extra` attribute to add more data fields if needed.
    If the data field should belong to a customer/depot node, then the user
    should extend the `Node` class instead.

    Parameters
    ----------
    nodes: list[Node]
        A list of customer nodes. Also includes the depot.
        By CVRPLIB convention, the depot always has index 0.

    vehicle_capacity: int
        Vehicle capcity for a homogeneous fleet.

    Optional:
        extra: dict
            A dict of additional data fields to include.

    """
    nodes: list[Node]
    vehicle_capacity: int
    extra: dict = None


class AbstractDecomposer(ABC):
    """Abstract base class that provides an interface for a `Decomposer`.
    User should extend this class, implement its abstract method `decompose()`
    and pass a concrete decomposer to `DecompositionRunner`.
    """

    @property
    def clusters(self):
        return self._clusters


    @clusters.setter
    def clusters(self, clusters):
        self._clusters = clusters


    @abstractmethod
    def decompose(self, inst: VRPInstance):
        """Decomposes the given VRP problem instance. The caller (e.g.
        `DecompositionRunner`) is responsible for calling this method and
        passing in the required `inst` object.
        Note: the depot should not be considered in the decomposition process.

        Parameters
        ----------
        inst: `VRPInstance`
            A VRP problem instance for the decomposer to decompose.

        Returns
        -------
        clusters: list[list[int]]
            A list of clustered customer IDs. Note: clusters should never
            include the depot.

            E.g. [[4, 2, 5], [3, 1]] means there are 2 clusters:
            - cluster 1 contains customers [2, 4, 5],
            - and cluster 2 contains customers [1, 3].

        """
        pass


class AbstractSolverWrapper(ABC):
    """Wrapper to an arbitrary VRP solver, e.g. HGS, Google OR-Tools, etc.
    Abstract base class that provides an interface for a `Solver`.
    User should extend this class, implement its abstract method `solve()`
    and pass a concrete solver to `DecompositionRunner`.
    """
    @abstractmethod
    def solve(self, inst: VRPInstance):
        """Solves the given VRP problem instance. The caller (e.g.
        `DecompositionRunner`) is responsible for calling this method and
        passing in the required `inst` object.

        Parameters
        ----------
        inst: `VRPInstance`
            A VRP problem instance for the solver to solve. The user is
            responsible for converting it to the proper format accepted by
            the underlying VRP solver.

        Returns
        -------
        cost: float
            Cost of the best found solution.

        routes: list[list[int]]
            Routes of the best found solution.

        """
        pass


class DecompositionRunner:
    """Manages the end-to-end decomposition and solving flow. Takes care of
    common tasks and delegates custom tasks to the decomposer and solver.
    """
    def __init__(
        self,
        inst: VRPInstance,
        decomposer: AbstractDecomposer,
        solver: AbstractSolverWrapper
    ) -> None:
        """Creates a `DecompositionRunner`.

        Parameters
        ----------
        inst: `VRPInstance`
            A VRP problem instance.

        decomposer: a subclass with concrete implementation of
        `AbstractDecomposer`.
            An instance of a concrete subclass of `AbstractDecomposer`.

        solver: a subclass with concrete implementation of
        `AbstractSolverWrapper`.
            An instance of a concrete subclass of `AbstractSolverWrapper`.

        """
        # TODO: use descriptor
        self.inst = inst
        self.decomposer = decomposer
        self.solver = solver


    def _check_type(self, inst):
        if type(inst) is not VRPInstance:
            raise TypeError(
                f'Argument must be of type '
                f'VRPInstance, but got type {type(inst)}.'
            )


    @property
    def inst(self):
        return self._inst


    @inst.setter
    def inst(self, inst: VRPInstance):
        self._check_type(inst)
        self._inst = inst


    def run(self, in_parallel=False, num_workers=1):
        """Solves the VRP problem by first decomposing it into smaller
        subproblems, then solving each subproblem independently, and finally
        returning the combined results.

        Parameters
        ----------
        in_parallel: bool
            Whether to solve the subproblems in parallel. Default is False, i.e.
            solve subproblems sequentially.

        num_workers: int
            The number of workers to use for parallel run. Only used if
            `in_parallel=True`. Default is 1, i.e. no parallelism.

        Returns
        -------
        total_cost: float
            Total cost of the original VRP problem, i.e. combined cost of
            decomposed subproblems.
        total_routes: list[list[int]]
            Routes for the original VRP problem, i.e. combined routes of
            decomposed subproblems.

        """
        self.decomposer.clusters = self.decomposer.decompose(self.inst)
        if in_parallel:
            # TODO: validate num_workers > 0, is not None
            # potential feature: is num_workers <= 0, use appropriate default
            return self._run_solver_parallel(num_workers)
        else:
            return self._run_solver_sequential()


    def _run_solver_sequential(self):
        """Run solver on decomposed subproblems sequentially."""
        total_cost = 0
        total_routes = []
        for cluster in self.decomposer.clusters:
            decomp_inst = self._build_decomposed_instance(cluster)
            cost, routes = \
                self._run_solver_on_decomposed_instance(decomp_inst, cluster)

            total_cost += cost
            total_routes.extend(routes)

        return total_cost, total_routes


    def _run_solver_parallel(self, num_workers):
        """Run solver on decomposed subproblems in parallel with the given
        number of workers."""
        decomp_inst_list = []
        for cluster in self.decomposer.clusters:
            decomp_inst_list.append(self._build_decomposed_instance(cluster))

        # start worker processes
        # TODO: how many workers are appropriate?
        # num_workers = min(os.cpu_count(), len(self.decomposer.clusters))
        with Pool(processes=num_workers) as pool:
            results = pool.starmap(
                self._run_solver_on_decomposed_instance,
                list(zip(decomp_inst_list, self.decomposer.clusters))
            )

        total_cost = 0
        total_routes = []
        for cost, routes in results:
            total_cost += cost
            total_routes.extend(routes)

        return total_cost, total_routes


    def _build_decomposed_instance(self, cluster) -> VRPInstance:
        """Build a subproblem instance including only the depot and
        customers present in the `cluster` list."""
        inst = self.inst
        # create a new VRPInstance and initialize it with the depot included
        depot = 0
        decomp_inst = VRPInstance(
            [deepcopy(inst.nodes[depot])],
            inst.vehicle_capacity,
            deepcopy(inst.extra),
        )
        # update the distance vector of the depot node
        decomp_inst.nodes[depot].distances = \
            inst.nodes[depot].get_decomposed_distances(cluster)

        for customer_id in cluster:
            decomp_inst.nodes.append(deepcopy(inst.nodes[customer_id]))
            # update the distance vector of the node just appended
            decomp_inst.nodes[-1].distances = \
                inst.nodes[customer_id].get_decomposed_distances(cluster)

        return decomp_inst


    def _run_solver_on_decomposed_instance(self, decomp_inst, cluster):
        logger.debug(f"Process ID: {os.getpid()}")
        cost, decomp_routes = self.solver.solve(decomp_inst)
        original_routes = self._map_decomposed_to_original_customer_ids(
            decomp_routes, cluster
        )
        return cost, original_routes


    def _map_decomposed_to_original_customer_ids(self, decomp_routes, cluster):
        '''Map subproblem customer IDs back to the original customer IDs.'''
        original_routes = []
        for route in decomp_routes:
            '''param `cluster` contains the customer IDs of the original problem.
            shift cluster index by 1 because customer indices start at 1 in
            the subproblem, where 0 is the depot, and the `cluster` does not
            contain the depot, but the subproblem does.
            e.g. customer with id = 3 in the subproblem
            is the customer with id = cluster[2] in the original problem

            subproblem nodes:          [0   1   2   3   4]
            (0 is depot)

            cluster index:                  0   1   2   3

            cluster values
            (original customer IDs):       [5   4   7   10]

            decomp_routes:                 [3,  1,  4,  2]

            original_routes:               [7,  5,  10, 4]

            '''
            route_with_original_customer_ids = \
                [cluster[customer_id - 1] for customer_id in route]
            original_routes.append(route_with_original_customer_ids)

        return original_routes
