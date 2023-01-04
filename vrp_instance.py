from dataclasses import dataclass

@dataclass
class Node:
    """Encapsulate each customer node and its attributes as a d-dimensional data point for clustering.

    Parameters
    ----------
        x_coord: float
            TODO: stub
        y_coord: float
            stub
        demand: int
            stub
        distances: list[float]
            stub

        Optional:
            start_time: int
                stub
            end_time: int
                stub
            service_time: int
                stub

    """
    # spatial coordinates
    x_coord: float
    y_coord: float

    demand: int

    # distance from this node to any other node (including depot)
    distances: list[float]

    ## fields below only exist for VRPTW

    # time window
    start_time: int = None
    end_time: int = None

    service_time: int = None


    def get_decomposed_distances(self, cluster: list):
        """Gets a decomposed distance list based on the provided `cluster` list. Only includes distances
        from this node to other nodes if those nodes are present in the cluster list.

        Parameters
        ----------
        cluster: list
            A list of customer IDs that represents a cluster of customers. Depot should be excluded.

        Returns
        -------
        distances: list
            Distances from this node to other nodes only if they are present in the `cluster` list.

        """
        # distance to the depot is always included
        depot = 0
        distances = [self.distances[depot]]

        for customer_id in cluster:
            if customer_id == depot:
                # cluster should only include customer IDs, but if it accidentally included depot,
                # distance to depot is already included, skip it
                continue
            distances.append(self.distances[customer_id])

        return distances


@dataclass
class VRPInstance:
    customers: list[Node]
    vehicle_capacity: int
