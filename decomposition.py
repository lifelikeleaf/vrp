from abc import ABC, abstractmethod


class AbstractConverter(ABC):
    @staticmethod
    @abstractmethod
    def convert_to_vrp_instance(benchmark):
        """Converts the benchmark instance from the format returned by a benchmark reader (e.g. cvrplib) to a `VRPInstance`.
        """
        pass

    # def convert_dict_to_solver_format():
    #     """Converts a VRP problem instance in dict format to a format accepted by the underlying VRP solver.
    #     """
    #     pass


class AbstractDecomposer(ABC):
    """Abstract base class that provides an interface for a Decomposer.
    User must extend this class, implement its abstract methods and pass a concrete `Decomposer` to `DecompositionRunner`.
    """
    @abstractmethod
    def decompose(self):
        """Decomposition method. *Note*: depot should not be considered in the decomposition.

        Returns
        -------
        clusters: list[list[int]]
            A list of clustered customer IDs. E.g. [[4, 2, 5], [3, 1]] means there are 2 clusters:
            cluster 1 contains customers [2, 4, 5], and cluster 2 contains customers [1, 3].

        """
        pass


class AbstractSolverWrapper(ABC):
    """Wrapper to an arbitrary VRP solver, e.g. HGS, Google OR-Tools, etc."""
    @abstractmethod
    def solve(self, inst):
        """Solves the given VRP problem instance.
        
        Parameters
        ----------
        inst: VRPInstance
            A VRP problem instance for the solver to solve. The solver wrapper is responsible for converting
            it to the proper format accepted by the underlying VRP solver.

        Returns
        -------
        cost: float
            Cost of the best found solution.

        routes: list[int]
            Routes of the best found solution.

        """
        pass


class DecompositionRunner:
    """Manages the end-to-end decomposition and solving flow. Takes care of common tasks and delegates
    custom tasks to the decomposer and solver.
    """
    def __init__(self, decomposer, solver) -> None:
        """Creates a `DecompositionRunner`.

        Parameters
        ----------
        decomposer: a subclass with concrete implementation of `AbstractDecomposer`.
            An instance of a concrete subclass of `AbstractDecomposer`.

        solver: a subclass with concrete implementation of `AbstractSolverWrapper`.
            An instance of a concrete subclass of `AbstractSolverWrapper`.

        """
        self.decomposer = decomposer
        self.solver = solver


    def decompose(self):
        self.decomposer.decompose()


    def solve(self):
        self.solver.solve()

