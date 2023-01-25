from functools import lru_cache
from scipy.spatial.distance import euclidean as scipy_euclidean
import numpy as np
import pandas as pd

from . import helpers


### Public Section ###

def v1(feature_vectors, decomposer):
    '''V1: spatial_temporal_dist = euclidean_dist + temporal_dist'''
    return _dist_matrix_symmetric(feature_vectors, decomposer, _pairwise_dist_v1)


def v4(feature_vectors, decomposer):
    return _dist_matrix_symmetric_normalized(feature_vectors, decomposer)


def v2(feature_vectors, decomposer):
    '''V2: simple overlap | simple gap'''
    return _dist_matrix_symmetric(feature_vectors, decomposer, _pairwise_dist_v2)


def v3(feature_vectors, decomposer):
    '''V3: v1 overlap | simple gap'''
    return _dist_matrix_symmetric(feature_vectors, decomposer, _pairwise_dist_v3)


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


    def compute_constituents(self):
        # node_i and node_j are feature_vectors from 2 different nodes,
        # in the format: [x, y, tw_start, tw_end]
        # TODO: use df for FV?
        x_i, y_i, tw_start_i, tw_end_i = self.node_i
        x_j, y_j, tw_start_j, tw_end_j = self.node_j
        self.tw_width_i = tw_end_i - tw_start_i
        self.tw_width_j = tw_end_j - tw_start_j

        self.max_tw_width = max(self.tw_width_i, self.tw_width_j)
        self.euclidean_dist = scipy_euclidean([x_i, y_i], [x_j, y_j])
        self.overlap_or_gap = helpers.get_time_window_overlap_or_gap(
            [tw_start_i, tw_end_i],
            [tw_start_j, tw_end_j]
        )
        self.overlap = self.overlap_or_gap if self.overlap_or_gap > 0 else 0
        self.gap = self.overlap_or_gap if self.overlap_or_gap < 0 else 0

        # collect constituents for this pair
        constituents = {
            'max_tw_width': self.max_tw_width,
            'euclidean_dist': self.euclidean_dist,
            'overlap': self.overlap,
            'gap': self.gap,
        }
        return constituents


    def temporal_dist_v1(self):
        '''V1'''
        # temporal_dist is the penalty or reduction applied to euclidean_dist
        # depending on whether there's a TW overlap or gap b/t 2 nodes
        temporal_dist = 0
        if self.overlap_or_gap >= 0:
            # there's a time window overlap between these 2 nodes
            overlap = self.overlap_or_gap
            temporal_dist = helpers.safe_divide(self.euclidean_dist, self.max_tw_width) * overlap
        elif self.decomposer.use_gap:
            # there's a time window gap between these 2 nodes
            assert self.overlap_or_gap < 0
            gap = self.overlap_or_gap
            # if wait time is not included in objective function,
            # then large gap b/t time windows is an advantage
            # bc it provides more flexibility for routing,
            # so leave it as a negative value will reduce
            # spatial_temporal_dist
            if self.decomposer.minimize_wait_time:
                # TODO: experiment with including wait time in OF
                # value - it's not considered by HGS solver and it's not
                # trivial to add it; check GOR Tools?
                # but if wait time IS included in objective function,
                # then large gap b/t time windows is a penalty,
                # so take its absolute value will increase
                # spatial_temporal_dist
                gap = abs(gap)

            temporal_dist = helpers.safe_divide(1, self.euclidean_dist) * gap

        return temporal_dist


    def temporal_dist_v2(self):
        '''V2: temporal_dist = simple overlap | simple gap'''

        temporal_dist = 0
        if self.overlap_or_gap >= 0:
            overlap = self.overlap_or_gap
            temporal_dist = overlap
        elif self.decomposer.use_gap:
            assert self.overlap_or_gap < 0
            gap = self.overlap_or_gap
            temporal_dist = gap

        return temporal_dist


    def temporal_dist_v3(self):
        '''V3: temporal_dist = v1 overlap | simple gap'''

        temporal_dist = 0
        if self.overlap_or_gap >= 0:
            overlap = self.overlap_or_gap
            temporal_dist = helpers.safe_divide(self.euclidean_dist, self.max_tw_width) * overlap
        elif self.decomposer.use_gap:
            assert self.overlap_or_gap < 0
            gap = self.overlap_or_gap
            temporal_dist = gap

        return temporal_dist


    def spatial_temporal_dist_v1(self):
        '''V1: spatial_temporal_dist = euclidean_dist + temporal_dist_v1'''
        temporal_dist = self.temporal_dist_v1()
        spatial_temporal_dist = self.euclidean_dist + temporal_dist
        # only non-negative distances are valid
        return max(0, spatial_temporal_dist)


    def spatial_temporal_dist_v2(self):
        '''V2: ... + temporal_dist_v2'''
        temporal_dist = self.temporal_dist_v2()
        spatial_temporal_dist = self.euclidean_dist + temporal_dist
        return max(0, spatial_temporal_dist)


    def spatial_temporal_dist_v3(self):
        '''V3: ... + temporal_dist_v3'''
        temporal_dist = self.temporal_dist_v3()
        spatial_temporal_dist = self.euclidean_dist + temporal_dist
        return max(0, spatial_temporal_dist)


def _pairwise_dist_v1(node_i, node_j, decomposer):
    pwd = _PairwiseDistance(node_i, node_j, decomposer)
    return pwd.spatial_temporal_dist_v1()


def _pairwise_dist_v2(node_i, node_j, decomposer):
    pwd = _PairwiseDistance(node_i, node_j, decomposer)
    return pwd.spatial_temporal_dist_v2()


def _pairwise_dist_v3(node_i, node_j, decomposer):
    pwd = _PairwiseDistance(node_i, node_j, decomposer)
    return pwd.spatial_temporal_dist_v3()


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
def _dist_matrix_symmetric_normalized(feature_vectors, decomposer):
    fv = feature_vectors.data
    n = len(fv)
    dist_matrix = np.zeros((n, n)) # n x n matrix of zeros
    max_tw_width_matrix = np.zeros((n, n))
    euclidean_dist_matrix = np.zeros((n, n))
    overlap_matrix = np.zeros((n, n))
    gap_matrix = np.zeros((n, n))
    # stats = pd.DataFrame()

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            elif i < j:
                pwd = _PairwiseDistance(fv[i], fv[j], decomposer)
                constituents = pwd.compute_constituents()
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

    max_tw_width_matrix = helpers.normalize_matrix(max_tw_width_matrix)
    euclidean_dist_matrix = helpers.normalize_matrix(euclidean_dist_matrix)
    overlap_matrix = helpers.normalize_matrix(overlap_matrix)
    gap_matrix = helpers.normalize_matrix(gap_matrix)

    stats = pd.DataFrame({
        'max_tw_width': max_tw_width_matrix.flatten(),
        'euclidean_dist': euclidean_dist_matrix.flatten(),
        'overlap': overlap_matrix.flatten(),
        'gap': gap_matrix.flatten(),
    })
    stats['TD_OL'] = stats['euclidean_dist'] / stats['max_tw_width'] * stats['overlap']
    stats['STD_OL'] = stats['euclidean_dist'] + stats['TD_OL']
    stats['TD_G'] = 1 / stats['euclidean_dist'] * stats['gap']
    stats['STD_G'] = stats['euclidean_dist'] + stats['TD_G']

    '''
    Create a new df column based on condtion:
    >>> a = {'A': [2, 3, 1], 'B': [2, -1, 3]}
    >>> df = pd.DataFrame(a)
    >>> df
       A  B
    0  2  2
    1  3  -1
    2  1  3
    - use indexing:
        >>> df.loc[df['A'] == df['B'], 'C'] = 0
        >>> df.loc[df['A'] > df['B'], 'C'] = 1
        >>> df.loc[df['A'] < df['B'], 'C'] = -1
    - vectorized numpy operation using np.where()
        >>> df['D'] = np.where(df['A'] == df['B'], 0, np.where(df['A'] >  df['B'], 1, -1))

    Replace values in a column based on condition:
    - use indexing:
        >>> df.loc[df.B < 0, 'B'] = 0
    - use df.mask()
        >>> df.B = df.B.mask(df.B.lt(0), 0)
        same as:
        >>> df.B = df.B.mask(df.B < 0, 0)
    '''

    return stats.round(2)

