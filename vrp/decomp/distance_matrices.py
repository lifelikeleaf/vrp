from functools import lru_cache
from scipy.spatial.distance import euclidean as scipy_euclidean
import numpy as np

from . import helpers


### Public Section ###

def v1(feature_vectors, decomposer):
    '''V1: spatial_temporal_dist = euclidean_dist + temporal_dist'''
    return _dist_matrix_symmetric(feature_vectors, decomposer, _pairwise_dist_v1)


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
        self.max_tw_width = 0
        self.euclidean_dist = 0
        self.overlap_or_gap = 0
        # TODO: collect stats
        self.stats = ['coords', 'max_tw_width', 'euclidean_dist', 'overlap_or_gap', 'temporal_dist']


    def compute_constituents(self):
        # node_i and node_j are feature_vectors from 2 different nodes,
        # in the format: [x, y, tw_start, tw_end]
        x_i, y_i, tw_start_i, tw_end_i = self.node_i
        x_j, y_j, tw_start_j, tw_end_j = self.node_j
        tw_width_i = tw_end_i - tw_start_i
        tw_width_j = tw_end_j - tw_start_j

        self.max_tw_width = max(tw_width_i, tw_width_j)
        self.euclidean_dist = scipy_euclidean([x_i, y_i], [x_j, y_j])
        self.overlap_or_gap = helpers.get_time_window_overlap_or_gap(
            [tw_start_i, tw_end_i],
            [tw_start_j, tw_end_j]
        )


    def temporal_dist_v1(self):
        '''V1'''
        self.compute_constituents()

        # temporal_dist is the penalty or reduction applied to euclidean_dist
        # depending on whether there's a TW overlap or gap b/t 2 nodes
        temporal_dist = 0
        if self.overlap_or_gap >= 0:
            # there's a time window overlap between these 2 nodes
            overlap = self.overlap_or_gap
            temporal_dist = helpers.safe_divide(self.euclidean_dist, self.max_tw_width) * overlap
            # print(
            #     f'Overlap: {round(overlap, 2)} b/t nodes ({round(x1, 2)}, {round(y1, 2)}) and ({round(x2, 2)}, {round(y2, 2)}); '
            #     f'euclidean dist: {round(self.euclidean_dist, 2)}; temporal weight: {round(temporal_dist, 2)} '
            # )
            # print(
            #     f'dist v1 = {round(self.euclidean_dist+temporal_dist, 2)} \n'
            #     f'dist v2 = {round(self.euclidean_dist+overlap, 2)}'
            # )
            # print()
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
            # TODO: descriptive stats
            # print(
            #     f'Gap: {round(gap, 2)} b/t nodes ({round(x1, 2)}, {round(y1, 2)}) and ({round(x2, 2)}, {round(y2, 2)}); '
            #     f'euclidean dist: {round(self.euclidean_dist, 2)}; temporal weight: {round(temporal_dist, 2)} '
            # )
            # print(
            #     f'dist v1 = {round(self.euclidean_dist+temporal_dist, 2)} \n'
            #     f'dist v2 = {round(self.euclidean_dist+gap, 2)}'
            # )
            # print()

        return temporal_dist


    def temporal_dist_v2(self):
        '''V2: temporal_dist = simple overlap | simple gap'''
        self.compute_constituents()

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
        self.compute_constituents()

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
    pd = _PairwiseDistance(node_i, node_j, decomposer)
    return pd.spatial_temporal_dist_v1()


def _pairwise_dist_v2(node_i, node_j, decomposer):
    pd = _PairwiseDistance(node_i, node_j, decomposer)
    return pd.spatial_temporal_dist_v2()


def _pairwise_dist_v3(node_i, node_j, decomposer):
    pd = _PairwiseDistance(node_i, node_j, decomposer)
    return pd.spatial_temporal_dist_v3()


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

