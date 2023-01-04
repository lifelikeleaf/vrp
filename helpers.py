import math
import numpy as np


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

