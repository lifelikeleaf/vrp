import math
import numpy as np
import time
from typing import Callable

from .decomposition import Node, VRPInstance
from .logger import logger

def get_min_tours(inst):
    """Returns the minimum number of tours (i.e. vehicles required) for routing the given instance.

    Params:
    - inst: benchmark instance data in dict format
    """
    # total demand of all customers / vehicle capacity
    return math.ceil(sum(inst['demands']) / inst['capacity'])


def normalize_feature_vectors(fv):
    """Normalize feature vectors using z-score."""
    fv = np.array(fv)
    # axis=0 -> row axis, runs down the rows, i.e. calculate the mean for each column/feature
    mean = np.mean(fv, axis=0)
    # ddof=1 -> degrees of freedom = N-1, i.e. sample std
    # ddof = 'delta degrees of freedom'
    # set ddof=0 for population std
    std = np.std(fv, axis=0, ddof=1)
    norm = (fv - mean) / std
    return norm


def convert_cvrplib_to_vrp_instance(benchmark) -> VRPInstance:
    """Converts a `cvrplib.Instance.VRPTW` object to a
    `decomposition.VRPInstance` object.
    
    Parameters
    ----------
    benchmark: `cvrplib.Instance.VRPTW`
        A benchmark VRPTW problem instance returned by `cvrplib.read()`.
    
    Returns
    -------
    inst: `decomposition.VRPInstance`
        A `VRPInstance` object representing the VRP problem instance.

    """
    node_list = []
    for customer_id in range(len(benchmark.coordinates)):
        params = dict(
            x_coord = benchmark.coordinates[customer_id][0],
            y_coord = benchmark.coordinates[customer_id][1],
            demand = benchmark.demands[customer_id],
            distances = benchmark.distances[customer_id],
            start_time = benchmark.earliest[customer_id],
            end_time = benchmark.latest[customer_id],
            service_time = benchmark.service_times[customer_id],
        )
        node = Node(**params)
        node_list.append(node)

    inst = VRPInstance(node_list, benchmark.capacity)
    # Example: how to tag extra data fields to VRPInstance
    # inst.extra = {'num_vehicle': 20, 'distance_limit': 100}
    return inst


def get_time_window_overlap_or_gap(tw_1, tw_2):
    """Computes the amount of overlap or gap between 2 time windows.
        - Overlap: if overlap_or_gap > 0
        - Gap: if overlap_or_gap < 0

    For example:
        - let tw_1 = [0, 15], tw_2 = [5, 20], then there's an overlap of 10.
        - let tw_1 = [0, 10], tw_2 = [5, 9], then there's an overlap of 4.
        - let tw_1 = [0, 10], tw_2 = [15, 20], then there's a gap of 5.
    """
    tw_start_1, tw_end_1 = tw_1
    tw_start_2, tw_end_2 = tw_2
    overlap_or_gap = min(tw_end_1, tw_end_2) - max(tw_start_1, tw_start_2)
    return overlap_or_gap


def log_run_time(func: Callable):
    """Decorator for logging the run time of a function or method."""
    log = logger.getChild(func.__module__)

    def decorator(*args, **kwargs):
        start = time.time()
        return_val = func(*args, **kwargs)
        end = time.time()
        log.info(f'{func.__qualname__}() run time = {end-start} sec')
        return return_val

    return decorator

