from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Node:
    """Encapsulates each customer node and its attributes as
    a d-dimensional data point for clustering.

    Parameters
    ----------
        x_coord: float
            A customer node's x coordinate.
        y_coord: float
            A customer node's y coordinate.
        demand: int
            A customer node's demand.
        distances: list[float]
            Distances from this node to any other node (including the depot).

        Optional:
            start_time: int
                Earliest service start time for this customer.
            end_time: int
                Latest service start time for this customer.
            service_time: int
                Service time for this customer.

    """
    # spatial coordinates
    x_coord: float
    y_coord: float

    demand: int

    # distances from this node to any other node (including the depot)
    distances: list[float]

    ## fields below only exist for VRPTW

    # time window
    start_time: int = None
    end_time: int = None

    service_time: int = None


    def get_decomposed_distances(self, cluster: list):
        """Gets a decomposed distance list based on the provided `cluster` list.
        Only includes distances from this node to other nodes if those nodes
        are present in the cluster list.

        Parameters
        ----------
        cluster: list
            A list of customer IDs that represents a cluster of customers.
            Depot should be excluded.

        Returns
        -------
        distances: list
            Distances from this node to other nodes only if they are present
            in the `cluster` list.

        """
        # distance to the depot is always included
        depot = 0
        distances = [self.distances[depot]]

        for customer_id in cluster:
            if customer_id == depot:
                # cluster should only include customer IDs, but if it
                # accidentally included depot, distance to depot is already
                # included above, so skip it
                continue
            distances.append(self.distances[customer_id])

        return distances


@dataclass
class VRPInstance:
    """A VRP problem instance representation more suitable for decomposition.
    Other benchmark instance readers (e.g. cvrplib) tend to represent a VRP
    instance on a 'column' basis (i.e. keyed on attributes). Here a VRP instance
    is represeneted on a 'row' basis, with customers and their attributes
    encapsulated as a list of `Node` objects.
    """
    customers: list[Node]
    vehicle_capacity: int


class AbstractDecomposer(ABC):
    """Abstract base class that provides an interface for a `Decomposer`.
    User should extend this class, implement its abstract method and pass
    a concrete decomposer to `DecompositionRunner`.
    """
    def __init__(self, inst: VRPInstance) -> None:
        """Subclasses that override this constructor should pass in
        an `VRPInstance` as the first argument and call
        `super().__init__(inst)`.
        
        Parameters
        ----------
        inst: `VRPInstance`
            A VRP problem instance. Required: subclasses that override this
            constructor must pass in an `VRPInstance` as the first argument.

        """
        if not issubclass(type(inst), VRPInstance):
            raise TypeError(f'First positional argument must be of type '
                f'vrp_instance.VRPInstance or its subclass, '
                f'but got type {type(inst)}.')

        self.inst = inst


    @abstractmethod
    def decompose(self):
        """Decomposition method. Note: depot should not be considered
        in the decomposition.

        Returns
        -------
        clusters: list[list[int]]
            A list of clustered customer IDs. E.g. [[4, 2, 5], [3, 1]]
            means there are 2 clusters: cluster 1 contains customers
            [2, 4, 5], and cluster 2 contains customers [1, 3].

        """
        pass


class AbstractSolverWrapper(ABC):
    """Wrapper to an arbitrary VRP solver, e.g. HGS, Google OR-Tools, etc."""
    @abstractmethod
    def solve(self, inst):
        """Solves the given VRP problem instance.
        
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

        routes: list[int]
            Routes of the best found solution.

        """
        pass


class DecompositionRunner:
    """Manages the end-to-end decomposition and solving flow. Takes care of
    common tasks and delegates custom tasks to the decomposer and solver.
    """
    def __init__(self, decomposer, solver) -> None:
        """Creates a `DecompositionRunner`.

        Parameters
        ----------
        decomposer: a subclass with concrete implementation of
        `AbstractDecomposer`.
            An instance of a concrete subclass of `AbstractDecomposer`.

        solver: a subclass with concrete implementation of
        `AbstractSolverWrapper`.
            An instance of a concrete subclass of `AbstractSolverWrapper`.

        """
        self.decomposer = decomposer
        self.solver = solver


    def decompose(self):
        self.decomposer.decompose()


    def solve(self):
        self.solver.solve()

