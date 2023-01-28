from functools import lru_cache
from scipy.spatial.distance import euclidean as scipy_euclidean
import numpy as np
import pandas as pd

from . import helpers


### Public Section ###

def v1(feature_vectors, decomposer):
    '''temporal dist'''
    return _dist_matrix_symmetric(feature_vectors, decomposer, _pairwise_dist_v1)


def v2_1(feature_vectors, decomposer, trial=False):
    '''temporal weight'''
    return _dist_matrix_symmetric_normalizable(feature_vectors, decomposer, _pairwise_dist_v2_1, trial=trial)


def v2_2(feature_vectors, decomposer, trial=False):
    '''temporal weight'''
    return _dist_matrix_symmetric_normalizable(feature_vectors, decomposer, _pairwise_dist_v2_2, trial=trial)


def euclidean(feature_vectors, decomposer):
    return _dist_matrix_symmetric(feature_vectors, decomposer, _pairwise_euclidean_dist)


### Private Section ###

class _PairwiseDistance:
    def __init__(self, node_i, node_j, decomposer) -> None:
        self.node_i = node_i
        self.node_j = node_j
        self.decomposer = decomposer
        self.tw_width_i = 0
        self.tw_width_j = 0
        self.max_tw_width = 0
        self.euclidean_dist = 0
        self.overlap_or_gap = 0
        self.overlap = 0
        self.gap = 0
        self.compute_constituents()

    # TODO: have dist_matrix func call this and pass it to temporal_dist func
    # bc constituents are always the same, only diff is if they're normalized or not.
    # BUT for non-normalized, you don't have to loop over all n x n nodes twice,
    # so it's a trade-off b/t efficiency and cleaner code.
    def compute_constituents(self):
        # node_i and node_j are feature_vectors from 2 different nodes,
        # in the format: [x, y, tw_start, tw_end]
        x_i, y_i, start_i, end_i = self.node_i
        x_j, y_j, start_j, end_j = self.node_j
        self.tw_width_i = end_i - start_i
        self.tw_width_j = end_j - start_j

        self.max_tw_width = max(self.tw_width_i, self.tw_width_j)
        self.euclidean_dist = scipy_euclidean([x_i, y_i], [x_j, y_j])
        self.overlap_or_gap = helpers.get_time_window_overlap_or_gap(
            [start_i, end_i],
            [start_j, end_j]
        )
        self.overlap = self.overlap_or_gap if self.overlap_or_gap > 0 else 0
        self.gap = abs(self.overlap_or_gap) if self.overlap_or_gap < 0 else 0

        # collect constituents for this pair
        constituents = {
            'max_tw_width': self.max_tw_width,
            'euclidean_dist': self.euclidean_dist,
            'overlap': self.overlap,
            'gap': self.gap,
        }
        return constituents


    def spatial_temporal_dist_v1(self):
        # temporal_dist is the penalty or reduction applied to euclidean_dist
        # depending on whether there's a TW overlap or gap b/t 2 nodes
        temporal_dist = 0

        if self.overlap > 0 and self.decomposer.use_overlap:
            # temporal_dist = euclidean_dist / max_tw_width * overlap = overlap / max_tw_width * euclidean_dist
            temporal_dist = helpers.safe_divide(self.euclidean_dist, self.max_tw_width) * self.overlap
        elif self.gap > 0 and self.decomposer.use_gap:
            # if wait time is not included in objective function,
            # then large gap b/t time windows is an advantage
            # bc it provides more flexibility for routing,
            # so it should reduce spatial_temporal_dist
            # temporal_dist = -1 * gap / euclidean_dist
            temporal_dist = -1 * helpers.safe_divide(1, self.euclidean_dist) * self.gap

            # TODO: experiment with including wait time in OF
            # value - it's not considered by HGS solver and it's not
            # trivial to add it; check GOR Tools?
            if self.decomposer.minimize_wait_time:
                # but if wait time IS included in objective function,
                # then large gap b/t time windows is a penalty,
                # so it should increase spatial_temporal_dist
                # temporal_dist = gap / euclidean_dist
                temporal_dist = abs(temporal_dist)

        spatial_temporal_dist = self.euclidean_dist + temporal_dist
        # only non-negative distances are valid
        return max(0, spatial_temporal_dist)


def _pairwise_dist_v1(node_i, node_j, decomposer):
    pwd = _PairwiseDistance(node_i, node_j, decomposer)
    return pwd.spatial_temporal_dist_v1()


def _pairwise_dist_v2_1(stats, decomposer):
    '''Formula v2.1: temporal weight'''
    overlap = stats['overlap']
    max_tw_width = stats['max_tw_width']
    gap = stats['gap']
    euclidean_dist = stats['euclidean_dist']

    temporal_weight = 0
    spatial_temporal_dist = euclidean_dist
    if overlap > 0 and decomposer.use_overlap:
        # temporal_weight = overlap / (overlap + max_tw_width)
        # up to ~50% weight bc by definition overlap <= max_tw_width
        # even though after normalization it could be a little higher
        temporal_weight = helpers.safe_divide(overlap, (overlap + max_tw_width))
        spatial_temporal_dist = euclidean_dist * (1 + temporal_weight)
    elif gap > 0 and decomposer.use_gap:
        # temporal_weight = gap / (gap + euclidean_dist)
        temporal_weight = helpers.safe_divide(gap, (gap + euclidean_dist))
        spatial_temporal_dist = euclidean_dist * (1 - temporal_weight)

        if decomposer.minimize_wait_time:
            spatial_temporal_dist = euclidean_dist * (1 + temporal_weight)

    return spatial_temporal_dist


def _pairwise_dist_v2_2(stats, decomposer):
    '''Formula v2.2: tw relative to the planning horizon, overlap relative to tw'''
    overlap = stats['overlap']
    relative_overlap = stats['relative_overlap']
    relative_tw_width = stats['relative_tw_width']
    gap = stats['gap']
    euclidean_dist = stats['euclidean_dist']

    temporal_weight = 0
    spatial_temporal_dist = euclidean_dist
    if overlap > 0 and decomposer.use_overlap:
        # TODO: refactor? this is the only line that differs b/t v2.1 and v2.2
        temporal_weight = relative_overlap * (1 - relative_tw_width)
        spatial_temporal_dist = euclidean_dist * (1 + temporal_weight)
    elif gap > 0 and decomposer.use_gap:
        # temporal_weight = gap / (gap + euclidean_dist)
        temporal_weight = helpers.safe_divide(gap, (gap + euclidean_dist))
        spatial_temporal_dist = euclidean_dist * (1 - temporal_weight)

        if decomposer.minimize_wait_time:
            spatial_temporal_dist = euclidean_dist * (1 + temporal_weight)

    return spatial_temporal_dist


def _pairwise_euclidean_dist(node_i, node_j, decomposer):
    x_i, y_i, tw_start_i, tw_end_i = node_i
    x_j, y_j, tw_start_j, tw_end_j = node_j
    euclidean_dist = scipy_euclidean([x_i, y_i], [x_j, y_j])
    return euclidean_dist


@lru_cache(maxsize=1)
@helpers.log_run_time
def _dist_matrix(feature_vectors, decomposer, pairwise_dist_callable):
    fv = feature_vectors.data
    n = len(fv)
    dist_matrix = []

    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                # dist to self should always be 0
                row.append(0)
            else:
                row.append(pairwise_dist_callable(fv[i], fv[j], decomposer))

        dist_matrix.append(row)

    return dist_matrix


@lru_cache(maxsize=1)
@helpers.log_run_time
def _dist_matrix_symmetric(feature_vectors, decomposer, pairwise_dist_callable):
    fv = feature_vectors.data
    n = len(fv)
    dist_matrix = np.zeros((n, n)) # n x n matrix of zeros

    for i in range(n):
        for j in range(n):
            if i == j:
                # do nothing: dist to self should always be 0 anyway
                continue
            elif i < j:
                dist_matrix[i, j] = pairwise_dist_callable(fv[i], fv[j], decomposer)
            else: # i > j
                dist_matrix[i, j] = dist_matrix[j, i]

    return dist_matrix


@lru_cache(maxsize=1)
@helpers.log_run_time
def _dist_matrix_symmetric_normalizable(feature_vectors, decomposer, pairwise_dist_callable, trial=False):
    fv = feature_vectors.data
    n = len(fv)
    dist_matrix = np.zeros((n, n)) # n x n matrix of zeros
    max_tw_width_matrix = np.zeros((n, n))
    euclidean_dist_matrix = np.zeros((n, n))
    overlap_matrix = np.zeros((n, n))
    gap_matrix = np.zeros((n, n))
    # stats = pd.DataFrame()

    fv_df = pd.DataFrame(fv, columns=['x', 'y', 'start', 'end'])
    tw_start_min = fv_df.start.min()
    tw_end_max = fv_df.end.max()
    # rough planning horizon
    time_horizon = tw_end_max - tw_start_min

    # get pairwise constituents first, for normalization and trial purposes
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            elif i < j:
                pwd = _PairwiseDistance(fv[i], fv[j], decomposer)
                ## using df here is ~5x slower
                # pwd = _PairwiseDistance(fv_df.iloc[i], fv_df.iloc[j], decomposer)
                constituents = pwd.compute_constituents()
                ## using df here is ~20-30x slower
                # df = pd.DataFrame(constituents, index=[0])
                # stats = pd.concat([stats, df], ignore_index=True)
                max_tw_width_matrix[i, j] = constituents['max_tw_width']
                euclidean_dist_matrix[i, j] = constituents['euclidean_dist']
                overlap_matrix[i, j] = constituents['overlap']
                gap_matrix[i, j] = constituents['gap']
            else: # i > j
                max_tw_width_matrix[i, j] = max_tw_width_matrix[j, i]
                euclidean_dist_matrix[i, j] = euclidean_dist_matrix[j, i]
                overlap_matrix[i, j] = overlap_matrix[j, i]
                gap_matrix[i, j] = gap_matrix[j, i]

    # must do this before normalization
    # and do not normalze bc they're relative ratios
    relative_tw_width_matrix = max_tw_width_matrix / time_horizon
    # NOTE: overlap could be > max_tw_width after normalization
    # even though that's impossible before
    relative_overlap_matrix = np.divide(
        overlap_matrix,
        max_tw_width_matrix,
        out=np.zeros_like(overlap_matrix),
        where=(max_tw_width_matrix != 0)
    )

    if decomposer.normalize:
        max_tw_width_matrix = helpers.normalize_matrix(max_tw_width_matrix)
        euclidean_dist_matrix = helpers.normalize_matrix(euclidean_dist_matrix)
        overlap_matrix = helpers.normalize_matrix(overlap_matrix)
        gap_matrix = helpers.normalize_matrix(gap_matrix)

    stats_matrix = {
        # v2.1
        'overlap': overlap_matrix,
        'max_tw_width': max_tw_width_matrix,
        # v2.2
        'relative_overlap': relative_overlap_matrix,
        'relative_tw_width': relative_tw_width_matrix,
        'gap': gap_matrix,
        'euclidean_dist': euclidean_dist_matrix,
    }

    if not trial:
        # get distance matrix
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                elif i < j:
                    pairwise_stats = {key: val[i, j] for key, val in stats_matrix.items()}
                    dist_matrix[i, j] = pairwise_dist_callable(pairwise_stats, decomposer)
                else: # i > j
                    dist_matrix[i, j] = dist_matrix[j, i]

        return dist_matrix
    else:
        # flatten bc per-column arrays must each be 1-dimensional
        stats_df = pd.DataFrame({key: val.flatten() for key, val in stats_matrix.items()})
        return _trial_dist_matrix_symmetric(stats_df)


def _trial_dist_matrix_symmetric(stats):
    '''Formula v1: temporal dist'''
    # stats['TD_OL'] = stats['euclidean_dist'] / stats['max_tw_width'] * stats['overlap']
    # stats['dist_OL'] = stats['euclidean_dist'] + stats['TD_OL']

    ## gap could be so large that even with normalization dist_G would be driven to < 0
    # stats['TD_G'] = 1 / stats['euclidean_dist'] * stats['gap']
    # stats['dist_G'] = stats['euclidean_dist'] - stats['TD_G']

    '''Formula v2.1: temporal weight'''
    ## up to ~50% weight bc by definition overlap <= max_tw_width
    ## even though after normalization it could be a little higher
    # stats['temp_w8_OL'] = stats['overlap'] / (stats['overlap'] + stats['max_tw_width'])

    '''Formula v2.2: relative to the planning horizon'''
    stats['temp_w8_OL'] = stats['relative_overlap'] * (1 - stats['relative_tw_width'])
    stats['dist_OL'] = stats['euclidean_dist'] * (1 + stats['temp_w8_OL'])

    stats['temp_w8_G'] = stats['gap'] / (stats['gap'] + stats['euclidean_dist'])
    stats['dist_G'] = stats['euclidean_dist'] * (1 - stats['temp_w8_G'])

    return stats

