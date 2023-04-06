# Author: Xu Ye <kan.ye@tum.de>

from functools import lru_cache
from scipy.spatial.distance import euclidean as scipy_euclidean
import numpy as np

from . import helpers


### Public Section ###

def v1_vectorized(feature_vectors, decomposer):
    '''temporal dist'''
    return _dist_matrix_symmetric_vectorized(feature_vectors, decomposer, _vectorized_dist_v1)


def v2_1_vectorized(feature_vectors, decomposer):
    '''temporal weight'''
    return _dist_matrix_symmetric_vectorized(feature_vectors, decomposer, _vectorized_dist_v2_1)


def v2_2_vectorized(feature_vectors, decomposer):
    '''temporal weight'''
    return _dist_matrix_symmetric_vectorized(feature_vectors, decomposer, _vectorized_dist_v2_2)


def v2_3_vectorized(feature_vectors, decomposer):
    '''weighted version of 2.2'''
    return _dist_matrix_symmetric_vectorized(feature_vectors, decomposer, _vectorized_dist_v2_3)


def v2_4_vectorized(feature_vectors, decomposer):
    '''weighted version of 2.2'''
    return _dist_matrix_symmetric_vectorized(feature_vectors, decomposer, _vectorized_dist_v2_4)


def v2_5_vectorized(feature_vectors, decomposer):
    '''weighted version of 2.2'''
    return _dist_matrix_symmetric_vectorized(feature_vectors, decomposer, _vectorized_dist_v2_5)


def v2_6_vectorized(feature_vectors, decomposer):
    '''weighted version of 2.2'''
    return _dist_matrix_symmetric_vectorized(feature_vectors, decomposer, _vectorized_dist_v2_6)


def v2_7_vectorized(feature_vectors, decomposer):
    '''weighted version of 2.2'''
    return _dist_matrix_symmetric_vectorized(feature_vectors, decomposer, _vectorized_dist_v2_7)


def v2_8_vectorized(feature_vectors, decomposer):
    '''weighted version of 2.2'''
    return _dist_matrix_symmetric_vectorized(feature_vectors, decomposer, _vectorized_dist_v2_8)


def v2_9_vectorized(feature_vectors, decomposer):
    '''weighted version of 2.2'''
    return _dist_matrix_symmetric_vectorized(feature_vectors, decomposer, _vectorized_dist_v2_9)


def v2_10_vectorized(feature_vectors, decomposer):
    '''weighted version of 2.2'''
    return _dist_matrix_symmetric_vectorized(feature_vectors, decomposer, _vectorized_dist_v2_10)


def euclidean_vectorized(feature_vectors, decomposer):
    return _dist_matrix_symmetric_vectorized(feature_vectors, decomposer, _vectorized_euclidean_dist)


def qi_2012_vectorized(feature_vectors, decomposer=None):
    '''Based on:
    Qi, M., Lin, W. H., Li, N., & Miao, L. (2012). A spatiotemporal
    partitioning approach for large-scale vehicle routing problems
    with time windows. Transportation Research Part E: Logistics
    and Transportation Review, 48(1), 248-257.
    '''
    k1 = 1
    k2 = 1.5
    k3 = 2
    alpha1 = 0.5
    alpha2 = 0.5
    return _dist_matrix_qi_2012_vectorized(feature_vectors, k1, k2, k3, alpha1, alpha2)


def v3_1_vectorized(feature_vectors, decomposer):
    return _dist_matrix_transformed_tw_vectorized(feature_vectors, decomposer)


def v4_1_vectorized(feature_vectors, decomposer):
    return _dist_matrix_symmetric_vectorized(feature_vectors, decomposer, _vectorized_dist_v4_1)


def v4_2_vectorized(feature_vectors, decomposer):
    return _dist_matrix_symmetric_vectorized(feature_vectors, decomposer, _vectorized_dist_v4_2)


''' DEPRECATED: use vectorized versions instead'''

def v1(feature_vectors, decomposer):
    '''temporal dist'''
    return _dist_matrix_symmetric(feature_vectors, decomposer, _pairwise_dist_v1)


def v2_1(feature_vectors, decomposer):
    '''temporal weight'''
    return _dist_matrix_symmetric_normalizable(feature_vectors, decomposer, _pairwise_dist_v2_1)


def v2_2(feature_vectors, decomposer):
    '''temporal weight'''
    return _dist_matrix_symmetric_normalizable(feature_vectors, decomposer, _pairwise_dist_v2_2)


def v2_3(feature_vectors, decomposer):
    '''weighted version of 2.2'''
    return _dist_matrix_symmetric_normalizable(feature_vectors, decomposer, _pairwise_dist_v2_3)


def v2_4(feature_vectors, decomposer):
    '''weighted version of 2.2'''
    return _dist_matrix_symmetric_normalizable(feature_vectors, decomposer, _pairwise_dist_v2_4)


def euclidean(feature_vectors, decomposer):
    return _dist_matrix_symmetric(feature_vectors, decomposer, _pairwise_euclidean_dist)


### Private Section ###

def _vectorized_dist_v1(constituents, decomposer):
    euclidean_dists = constituents['euclidean_dist']
    max_tw_widths = constituents['max_tw_width']
    overlaps = constituents['overlap']
    gaps = constituents['gap']

    temporal_dists = 0

    if decomposer.use_overlap:
        # temporal_dist = euclidean_dist / max_tw_width * overlap
        #               = overlap / max_tw_width * euclidean_dist
        temporal_dists = np.divide(
            euclidean_dists,
            max_tw_widths,
            out=np.zeros(euclidean_dists.shape),
            where=(max_tw_widths != 0)
        ) * overlaps

    if decomposer.use_gap:
        # temporal_dist = -1 * gap / euclidean_dist
        temporal_dists += np.divide(
            1,
            euclidean_dists,
            out=np.zeros(euclidean_dists.shape),
            where=(euclidean_dists != 0)
        ) * gaps * -1

        if decomposer.penalize_gap:
            # temporal_dist = gap / euclidean_dist
            temporal_dists = np.absolute(temporal_dists)

    spatial_temporal_dists = euclidean_dists + temporal_dists
    # only non-negative distances are valid
    spatial_temporal_dists = np.maximum(0, spatial_temporal_dists)

    return spatial_temporal_dists


def _vectorized_dist_v2_1(constituents, decomposer):
    '''temporal weight'''
    overlaps = constituents['overlap']
    max_tw_widths = constituents['max_tw_width']
    gaps = constituents['gap']
    euclidean_dists = constituents['euclidean_dist']

    temporal_weights = 0
    spatial_temporal_dists = euclidean_dists
    if decomposer.use_overlap:
        # temporal_weight = overlap / (overlap + max_tw_width)
        # up to ~50% weight bc by definition overlap <= max_tw_width
        # even though after normalization it could be a little higher
        temporal_weights = np.divide(
            overlaps,
            (overlaps + max_tw_widths),
            out=np.zeros(overlaps.shape),
            where=((overlaps + max_tw_widths) != 0)
        )
        spatial_temporal_dists *= (1 + temporal_weights)

    # this is safe bc overlap and gap are by definition mutually exclusive,
    # i.e. a pair of nodes can either have a TW overlap or a TW gap,
    # or neither, but not both.
    if decomposer.use_gap:
        # temporal_weight = gap / (gap + euclidean_dist)
        temporal_weights = np.divide(
            gaps,
            (gaps + euclidean_dists),
            out=np.zeros(gaps.shape),
            where=((gaps + euclidean_dists) != 0)
        )

        if decomposer.penalize_gap:
            spatial_temporal_dists *= (1 + temporal_weights)
        else:
            spatial_temporal_dists *= (1 - temporal_weights)

    return spatial_temporal_dists


def _vectorized_dist_v2_2(constituents, decomposer, overlap_lambda=1, gap_lambda=1):
    '''tw relative to the planning horizon, overlap relative to tw'''
    relative_overlaps = constituents['relative_overlap']
    relative_tw_widths = constituents['relative_tw_width']
    gaps = constituents['gap']
    euclidean_dists = constituents['euclidean_dist']

    temporal_weights = 0
    spatial_temporal_dists = euclidean_dists
    if decomposer.use_overlap:
        # temporal_weight = relative_overlap * (1 - relative_tw_width) * lambda
        temporal_weights = relative_overlaps * (1 - relative_tw_widths) * overlap_lambda
        spatial_temporal_dists *= (1 + temporal_weights)

    # this is safe bc overlap and gap are by definition mutually exclusive,
    # i.e. a pair of nodes can either have a TW overlap or a TW gap,
    # or neither, but not both.
    if decomposer.use_gap:
        # temporal_weight = gap / (gap + euclidean_dist) * lambda
        temporal_weights = np.divide(
            gaps,
            (gaps + euclidean_dists),
            out=np.zeros(gaps.shape),
            where=((gaps + euclidean_dists) != 0)
        ) * gap_lambda

        if decomposer.penalize_gap:
            spatial_temporal_dists *= (1 + temporal_weights)
        else:
            spatial_temporal_dists *= (1 - temporal_weights)

    return spatial_temporal_dists


def _vectorized_dist_v2_3(constituents, decomposer):
    '''limit weight up to 50%'''
    return _vectorized_dist_v2_2(constituents, decomposer, overlap_lambda=0.5, gap_lambda=0.5)


def _vectorized_dist_v2_4(constituents, decomposer):
    '''limit weight up to 30%'''
    return _vectorized_dist_v2_2(constituents, decomposer, overlap_lambda=0.3, gap_lambda=0.3)


def _vectorized_dist_v2_5(constituents, decomposer):
    '''limit weight up to 15%'''
    return _vectorized_dist_v2_2(constituents, decomposer, overlap_lambda=0.15, gap_lambda=0.15)


def _vectorized_dist_v2_6(constituents, decomposer):
    '''limit weight up to 50% for overlap, 30% for gap'''
    return _vectorized_dist_v2_2(constituents, decomposer, overlap_lambda=0.5, gap_lambda=0.3)


def _vectorized_dist_v2_7(constituents, decomposer):
    '''limit weight up to 40%'''
    return _vectorized_dist_v2_2(constituents, decomposer, overlap_lambda=0.4, gap_lambda=0.4)


def _vectorized_dist_v2_8(constituents, decomposer):
    '''limit weight up to 20%'''
    return _vectorized_dist_v2_2(constituents, decomposer, overlap_lambda=0.2, gap_lambda=0.2)


def _vectorized_dist_v2_9(constituents, decomposer):
    '''limit weight up to 10%'''
    return _vectorized_dist_v2_2(constituents, decomposer, overlap_lambda=0.1, gap_lambda=0.1)


def _vectorized_dist_v2_10(constituents, decomposer):
    '''limit weight up to 5%'''
    return _vectorized_dist_v2_2(constituents, decomposer, overlap_lambda=0.05, gap_lambda=0.05)


def _vectorized_dist_v4_1(constituents, decomposer, overlap_lambda=1, gap_lambda=1):
    '''tw relative to the planning horizon, overlap relative to tw'''
    relative_overlaps = constituents['relative_overlap']
    relative_tw_widths = constituents['relative_tw_width']
    gaps = constituents['gap']
    euclidean_dists = constituents['euclidean_dist']

    temporal_weights = 0
    spatial_temporal_dists = euclidean_dists
    if decomposer.use_overlap:
        # temporal_weight = relative_overlap * (1 - relative_tw_width) * lambda
        temporal_weights = relative_overlaps * (1 - relative_tw_widths) * overlap_lambda
        spatial_temporal_dists *= (1 + temporal_weights)

    # this is safe bc overlap and gap are by definition mutually exclusive,
    # i.e. a pair of nodes can either have a TW overlap or a TW gap,
    # or neither, but not both.
    if decomposer.use_gap:
        # temporal_weight = gap / (gap + euclidean_dist) * lambda
        temporal_weights = np.divide(
            gaps,
            (gaps + euclidean_dists),
            out=np.zeros(gaps.shape),
            where=((gaps + euclidean_dists) != 0)
        ) * gap_lambda

        # ignoring service time,
        # if gap - euclidean dist > 0, then there's a guaranteed wait time, which is penalized
        # otherwise the gap is interpreted as a flexibility for routing, which is encouraged
        temporal_weights = np.where(gaps - euclidean_dists > 0, temporal_weights, -1 * temporal_weights)
        spatial_temporal_dists *= (1 + temporal_weights)

    return spatial_temporal_dists


def _vectorized_dist_v4_2(constituents, decomposer):
    return _vectorized_dist_v4_1(constituents, decomposer, overlap_lambda=0.5, gap_lambda=0.5)


def _vectorized_euclidean_dist(constituents, decomposer):
    return constituents['euclidean_dist']


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
        # in the format: [x, y, tw_start, tw_end, service_time]
        x_i = self.node_i[0]
        y_i = self.node_i[1]
        start_i = self.node_i[2]
        end_i = self.node_i[3]

        x_j = self.node_j[0]
        y_j = self.node_j[1]
        start_j = self.node_j[2]
        end_j = self.node_j[3]
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
            # temporal_dist = euclidean_dist / max_tw_width * overlap
            #               = overlap / max_tw_width * euclidean_dist
            temporal_dist = helpers.safe_divide(self.euclidean_dist, self.max_tw_width) * self.overlap
        elif self.gap > 0 and self.decomposer.use_gap:
            # temporal_dist = -1 * gap / euclidean_dist
            temporal_dist = -1 * helpers.safe_divide(1, self.euclidean_dist) * self.gap

            if self.decomposer.penalize_gap:
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

        if decomposer.penalize_gap:
            spatial_temporal_dist = euclidean_dist * (1 + temporal_weight)

    return spatial_temporal_dist


def _pairwise_dist_v2_2(stats, decomposer, weight_limit=1):
    '''Formula v2.2: tw relative to the planning horizon, overlap relative to tw'''
    overlap = stats['overlap']
    relative_overlap = stats['relative_overlap']
    relative_tw_width = stats['relative_tw_width']
    gap = stats['gap']
    euclidean_dist = stats['euclidean_dist']

    temporal_weight = 0
    spatial_temporal_dist = euclidean_dist
    if overlap > 0 and decomposer.use_overlap:
        temporal_weight = relative_overlap * (1 - relative_tw_width) * weight_limit
        spatial_temporal_dist = euclidean_dist * (1 + temporal_weight)
    elif gap > 0 and decomposer.use_gap:
        # temporal_weight = gap / (gap + euclidean_dist)
        temporal_weight = helpers.safe_divide(gap, (gap + euclidean_dist)) * weight_limit
        spatial_temporal_dist = euclidean_dist * (1 - temporal_weight)

        if decomposer.penalize_gap:
            spatial_temporal_dist = euclidean_dist * (1 + temporal_weight)

    return spatial_temporal_dist


def _pairwise_dist_v2_3(stats, decomposer):
    '''Formula v2.3: limit weight up to 50%'''
    return _pairwise_dist_v2_2(stats, decomposer, weight_limit=0.5)


def _pairwise_dist_v2_4(stats, decomposer):
    '''Formula v2.4: limit weight up to 30%'''
    return _pairwise_dist_v2_2(stats, decomposer, weight_limit=0.3)


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


@helpers.log_run_time
def _get_constituents_matrix(fv, decomposer):
    n = len(fv)
    max_tw_width_matrix = np.zeros((n, n))
    euclidean_dist_matrix = np.zeros((n, n))
    overlap_matrix = np.zeros((n, n))
    gap_matrix = np.zeros((n, n))
    # stats = pd.DataFrame()

    x = np.asarray(list(zip(*fv)))[0]
    y = np.asarray(list(zip(*fv)))[1]
    start = np.asarray(list(zip(*fv)))[2]
    end = np.asarray(list(zip(*fv)))[3]
    planning_horizon = end.max() - start.min()

    # get pairwise constituents, for normalization purposes
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
    relative_tw_width_matrix = max_tw_width_matrix / planning_horizon
    # overlap could be > max_tw_width after normalization
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

    constituents_matrix = {
        # v2.1
        'overlap': overlap_matrix,
        'max_tw_width': max_tw_width_matrix,
        # v2.2+
        'relative_overlap': relative_overlap_matrix,
        'relative_tw_width': relative_tw_width_matrix,

        'gap': gap_matrix,
        'euclidean_dist': euclidean_dist_matrix,
    }

    return constituents_matrix


@lru_cache(maxsize=1)
@helpers.log_run_time
def _dist_matrix_symmetric_normalizable(feature_vectors, decomposer, pairwise_dist_callable):
    fv = feature_vectors.data
    n = len(fv)
    constituents_matrix = _get_constituents_matrix(fv, decomposer)
    dist_matrix = np.zeros((n, n)) # n x n matrix of zeros

    # get distance matrix
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            elif i < j:
                pairwise_stats = {key: val[i, j] for key, val in constituents_matrix.items()}
                dist_matrix[i, j] = pairwise_dist_callable(pairwise_stats, decomposer)
            else: # i > j
                dist_matrix[i, j] = dist_matrix[j, i]

    return dist_matrix


def _get_transformed_tw(nodes, euclidean_dists):
    start_times = nodes[:, 2]
    end_times = nodes[:, 3]
    service_times = nodes[:, 4]

    transformed_start_times = start_times + service_times + euclidean_dists
    transformed_end_times = end_times + service_times + euclidean_dists

    return transformed_start_times, transformed_end_times


def _transformed_tw_temporal_dists_directional(nodes_i, nodes_j, euclidean_dists, planning_horizon):
    start_times_i, end_times_i = _get_transformed_tw(nodes_i, euclidean_dists)
    start_times_j = nodes_j[:, 2]
    end_times_j = nodes_j[:, 3]

    overlaps_or_gaps = np.minimum(end_times_i, end_times_j) - np.maximum(start_times_i, start_times_j)

    # conveniently, overlap is positive, gap is negative
    # with transformed tw, overlap means flexibility
    # gap means either early arrival or late arrival
    # so temporal_dist with overlap should be smaller and with gap larger
    temporal_dists = planning_horizon - overlaps_or_gaps

    return temporal_dists


def _transformed_tw_temporal_weights_directional(nodes_i, nodes_j, euclidean_dists, planning_horizon, decomposer):
    start_times_i, end_times_i = _get_transformed_tw(nodes_i, euclidean_dists)
    start_times_j = nodes_j[:, 2]
    end_times_j = nodes_j[:, 3]

    overlaps_or_gaps = np.minimum(end_times_i, end_times_j) - np.maximum(start_times_i, start_times_j)
    # overlaps = np.where(overlaps_or_gaps > 0, overlaps_or_gaps, 0)
    # gaps = np.where(overlaps_or_gaps < 0, overlaps_or_gaps, 0)

    # savings = 0
    # if decomposer.use_overlap and decomposer.use_gap:
    #     savings = overlaps_or_gaps
    # elif decomposer.use_overlap:
    #     savings = overlaps
    # elif decomposer.use_gap:
    #     savings = gaps

    # conveniently, overlap is positive, gap is negative
    # with transformed tw, overlap means flexibility
    # gap means either early arrival or late arrival
    temporal_weights = overlaps_or_gaps / planning_horizon

    return temporal_weights


@helpers.log_run_time
def _dist_matrix_transformed_tw_vectorized(feature_vectors, decomposer):
    fv = feature_vectors.data
    n = len(fv)

    # upper triangle indices of an n x n symmetric matrix
    # skip the main diagonal (k=1) bc dist to self is zero
    i, j = np.triu_indices(n, k=1)

    # all unique pairwise nodes
    nodes_i = fv[i]
    nodes_j = fv[j]

    x_coords_i = nodes_i[:, 0]
    y_coords_i = nodes_i[:, 1]
    x_coords_j = nodes_j[:, 0]
    y_coords_j = nodes_j[:, 1]
    euclidean_dists = np.sqrt((x_coords_j - x_coords_i) ** 2 + (y_coords_j - y_coords_i) ** 2)

    start = np.asarray(list(zip(*fv)))[2]
    end = np.asarray(list(zip(*fv)))[3]
    planning_horizon = end.max() - start.min()

    temporal_dists_ij = _transformed_tw_temporal_dists_directional(nodes_i, nodes_j, euclidean_dists, planning_horizon)
    temporal_dists_ji = _transformed_tw_temporal_dists_directional(nodes_j, nodes_i, euclidean_dists, planning_horizon)
    # temporal_weights_ij = _transformed_tw_temporal_weights_directional(nodes_i, nodes_j, euclidean_dists, planning_horizon, decomposer)
    # temporal_weights_ji = _transformed_tw_temporal_weights_directional(nodes_j, nodes_i, euclidean_dists, planning_horizon, decomposer)

    temporal_dists = np.maximum(temporal_dists_ij, temporal_dists_ji)
    # temporal_dists = np.minimum(temporal_dists_ij, temporal_dists_ji)
    # temporal_dists = temporal_dists_ij + temporal_dists_ji
    # temporal_weights = np.maximum(temporal_weights_ij, temporal_weights_ji)
    # temporal_weights = np.minimum(temporal_weights_ij, temporal_weights_ji)

    # temporal_matrix = np.zeros((n, n))
    # temporal_matrix[i, j] = temporal_dists
    # temporal_matrix += temporal_matrix.T
    # print('temporal matrix:')
    # print(temporal_matrix.round(2))
    # print()

    spatial_temporal_dists = helpers.normalize_matrix(euclidean_dists) + helpers.normalize_matrix(temporal_dists)
    # spatial_temporal_dists = euclidean_dists * (1 - temporal_weights)

    dist_matrix = np.zeros((n, n)) # n x n matrix of zeros
    # fill the upper triangle
    dist_matrix[i, j] = spatial_temporal_dists
    # fill the lower triangle
    dist_matrix += dist_matrix.T

    return dist_matrix


@helpers.log_run_time
def _get_constituents_vectorized(fv, decomposer, as_matrix=False):
    n = len(fv)

    # upper triangle indices of an n x n symmetric matrix
    # skip the main diagonal (k=1) bc dist to self is zero
    i, j = np.triu_indices(n, k=1)

    # all unique pairwise nodes
    nodes_i = fv[i]
    nodes_j = fv[j]

    x_coords_i = nodes_i[:, 0]
    y_coords_i = nodes_i[:, 1]
    start_times_i = nodes_i[:, 2]
    end_times_i = nodes_i[:, 3]

    x_coords_j = nodes_j[:, 0]
    y_coords_j = nodes_j[:, 1]
    start_times_j = nodes_j[:, 2]
    end_times_j = nodes_j[:, 3]

    tw_widths_i = end_times_i - start_times_i
    tw_widths_j = end_times_j - start_times_j

    max_tw_widths = np.maximum(tw_widths_i, tw_widths_j)
    euclidean_dists = np.sqrt((x_coords_j - x_coords_i) ** 2 + (y_coords_j - y_coords_i) ** 2)
    overlaps_or_gaps = np.minimum(end_times_i, end_times_j) - np.maximum(start_times_i, start_times_j)
    overlaps = np.where(overlaps_or_gaps > 0, overlaps_or_gaps, 0)
    gaps = np.where(overlaps_or_gaps < 0, np.absolute(overlaps_or_gaps), 0)

    x = np.asarray(list(zip(*fv)))[0]
    y = np.asarray(list(zip(*fv)))[1]
    start = np.asarray(list(zip(*fv)))[2]
    end = np.asarray(list(zip(*fv)))[3]
    planning_horizon = end.max() - start.min()

    if as_matrix:
        max_tw_width_matrix = np.zeros((n, n))
        euclidean_dist_matrix = np.zeros((n, n))
        overlap_matrix = np.zeros((n, n))
        gap_matrix = np.zeros((n, n))

        # fill the upper triangle
        max_tw_width_matrix[i, j] = max_tw_widths
        euclidean_dist_matrix[i, j] = euclidean_dists
        overlap_matrix[i, j] = overlaps
        gap_matrix[i, j] = gaps

        # fill the lower triangle
        max_tw_width_matrix += max_tw_width_matrix.T
        euclidean_dist_matrix += euclidean_dist_matrix.T
        overlap_matrix += overlap_matrix.T
        gap_matrix += gap_matrix.T

        relative_tw_width_matrix = max_tw_width_matrix / planning_horizon
        relative_overlap_matrix = np.divide(
            overlap_matrix,
            max_tw_width_matrix,
            # type of `out` matters:
            # np.zeros_like returns an array of zeros with
            # the same shape and type as input,
            # so if input were of type int64 instead of float64,
            # numpy would complain. But since `overlap_matrix`
            # is initialized to np.zeros((n, n)) with default
            # type numpy.float64, there's no problem. Otherwise
            # one could use `out=np.zeros(input.shape)`
            out=np.zeros_like(overlap_matrix),
            where=(max_tw_width_matrix != 0)
        )

        if decomposer.normalize:
            max_tw_width_matrix = helpers.normalize_matrix(max_tw_width_matrix)
            euclidean_dist_matrix = helpers.normalize_matrix(euclidean_dist_matrix)
            overlap_matrix = helpers.normalize_matrix(overlap_matrix)
            gap_matrix = helpers.normalize_matrix(gap_matrix)

        constituents_matrix = {
            # v2.1
            'overlap': overlap_matrix,
            'max_tw_width': max_tw_width_matrix,
            # v2.2+
            'relative_overlap': relative_overlap_matrix,
            'relative_tw_width': relative_tw_width_matrix,

            'gap': gap_matrix,
            'euclidean_dist': euclidean_dist_matrix,
        }

        return constituents_matrix

    else:
        relative_tw_widths = max_tw_widths / planning_horizon
        relative_overlaps = np.divide(
            overlaps,
            max_tw_widths,
            out=np.zeros(overlaps.shape),
            where=(max_tw_widths != 0)
        )

        if decomposer.normalize:
            max_tw_widths = helpers.normalize_matrix(max_tw_widths)
            euclidean_dists = helpers.normalize_matrix(euclidean_dists)
            overlaps = helpers.normalize_matrix(overlaps)
            gaps = helpers.normalize_matrix(gaps)

        constituents = {
            # v2.1
            'overlap': overlaps,
            'max_tw_width': max_tw_widths,
            # v2.2+
            'relative_overlap': relative_overlaps,
            'relative_tw_width': relative_tw_widths,

            'gap': gaps,
            'euclidean_dist': euclidean_dists,
        }

        return constituents


@lru_cache(maxsize=1)
@helpers.log_run_time
def _dist_matrix_symmetric_vectorized(feature_vectors, decomposer, vectorized_dist_callable):
    fv = feature_vectors.data

    as_matrix = True # get constituents as a 2-D matrix or 1-D array
    constituents = _get_constituents_vectorized(fv, decomposer, as_matrix=as_matrix)

    if as_matrix:
        dist_matrix = vectorized_dist_callable(constituents, decomposer)
    else:
        n = len(fv)
        i, j = np.triu_indices(n, k=1)
        dist_matrix = np.zeros((n, n)) # n x n matrix of zeros
        # fill the upper triangle
        dist_matrix[i, j] = vectorized_dist_callable(constituents, decomposer)
        # fill the lower triangle
        dist_matrix += dist_matrix.T

    return dist_matrix


@lru_cache(maxsize=1)
@helpers.log_run_time
def _dist_matrix_qi_2012_vectorized(feature_vectors, k1, k2, k3, alpha1, alpha2):
    fv = feature_vectors.data
    fv_depot_data = feature_vectors.depot_data

    n = len(fv)

    # upper triangle indices of an n x n symmetric matrix
    # skip the main diagonal (k=1) bc dist to self is zero
    i, j = np.triu_indices(n, k=1)

    # all unique pairwise nodes
    nodes_i = fv[i]
    nodes_j = fv[j]

    x_coords_i = nodes_i[:, 0]
    y_coords_i = nodes_i[:, 1]
    x_coords_j = nodes_j[:, 0]
    y_coords_j = nodes_j[:, 1]
    euclidean_dists = np.sqrt((x_coords_j - x_coords_i) ** 2 + (y_coords_j - y_coords_i) ** 2) # t_ij

    # euclidean_matrix = np.zeros((n, n))
    # euclidean_matrix[i, j] = euclidean_dists
    # euclidean_matrix += euclidean_matrix.T
    # print('euclidean matrix:')
    # print(euclidean_matrix.round(2))
    # print()

    temporal_dists_ij = _qi_2012_temporal_dists_directional(nodes_i, nodes_j, k1, k2, k3, euclidean_dists, fv_depot_data)
    temporal_dists_ji = _qi_2012_temporal_dists_directional(nodes_j, nodes_i, k1, k2, k3, euclidean_dists, fv_depot_data)
    '''
    NOTE: in section "4. Measuring temporal and spatiotemporal distance,"
    it is presented that the algorithm is to choose the `max` of the two
    directional temporal distances as the undirected pairwise temporal distance
    (Eq. 6).
    However, in section "6.1. An example for a small scale network," the
    numerical results are actually calculated using the `min` of the two
    directional temporal distances as the undirected pairwise temporal distance
    (Table 2).
    Use `min` for reconciliation with Table 2 and verification of correctness
    of my implementation.
    Use `max` for experiments and numerical comparison as it performs much
    better than `min`, indicating that Eq. 6 is the correct version of the
    algorithm intended by the authors of Qi et al 2012.
    '''
    temporal_dists = np.maximum(temporal_dists_ij, temporal_dists_ji)
    # temporal_dists = np.minimum(temporal_dists_ij, temporal_dists_ji)
    # temporal_dists = temporal_dists_ij + temporal_dists_ji

    # temporal_matrix = np.zeros((n, n))
    # temporal_matrix[i, j] = temporal_dists
    # temporal_matrix += temporal_matrix.T
    # print('temporal matrix:')
    # print(temporal_matrix.round(2))
    # print()

    spatiotemporal_dists = alpha1 * helpers.normalize_matrix(euclidean_dists) + alpha2 * helpers.normalize_matrix(temporal_dists)

    dist_matrix = np.zeros((n, n)) # n x n matrix of zeros
    # fill the upper triangle
    dist_matrix[i, j] = spatiotemporal_dists
    # fill the lower triangle
    dist_matrix += dist_matrix.T

    return dist_matrix


def _qi_2012_temporal_dists_directional(nodes_i, nodes_j, k1, k2, k3, euclidean_dists, fv_depot_data):
    start_times_i = nodes_i[:, 2] # a
    end_times_i = nodes_i[:, 3] # b
    service_times_i = nodes_i[:, 4] # s_i

    start_times_j = nodes_j[:, 2] # c
    end_times_j = nodes_j[:, 3] # d

    tw_widths_i = end_times_i - start_times_i
    tw_widths_j = end_times_j - start_times_j
    max_tw_widths = np.maximum(tw_widths_i, tw_widths_j)

    depot_start_time = fv_depot_data[2]
    depot_end_time = fv_depot_data[3]
    depot_tw_width = depot_end_time - depot_start_time
    # maximum time window width among all customers and the depot
    A = max(max(max_tw_widths), depot_tw_width)

    arrival_times_j_low = start_times_i + service_times_i + euclidean_dists # a' (a prime)
    arrival_times_j_high = end_times_i + service_times_i + euclidean_dists # b' (b prime)

    # definite integral
    def def_integral(antiderivative, lower_limit, upper_limit, **kwargs):
        return antiderivative(upper_limit, **kwargs) - antiderivative(lower_limit, **kwargs)

    # x=t' < c (penalize early arrival, i.e. wait time)
    def k2_integrand(x, k1, k2, c, d):
        return k2 * x + k1 * d - (k1 + k2) * c

    def k2_antiderivative(x, k1, k2, c, d):
        # antiderivative of k2_integrand
        return k2 * x ** 2 / 2 + k1 * d * x - (k1 + k2) * c * x

    k2_result = def_integral(
        k2_antiderivative,
        np.minimum(arrival_times_j_low, start_times_j),
        np.minimum(arrival_times_j_high, start_times_j),
        k1=k1, k2=k2, c=start_times_j, d=end_times_j
    )

    # x=t' in [c, d] (reward savings)
    def k1_integrand(x, k1, d):
        return -k1 * x + k1 * d

    def k1_antiderivative(x, k1, d):
        # antiderivative of k1_integrand
        return -k1 * x ** 2 / 2 + k1 * d * x

    k1_result = def_integral(
        k1_antiderivative,
        np.minimum(np.maximum(arrival_times_j_low, start_times_j), end_times_j),
        np.maximum(np.minimum(arrival_times_j_high, end_times_j), start_times_j),
        k1=k1, d=end_times_j
    )

    # x=t' > d (penalize late arrival)
    def k3_integrand(x, k3, d):
        return -k3 * x + k3 * d

    def k3_antiderivative(x, k3, d):
        # antiderivative of k3_integrand
        return -k3 * x ** 2 / 2 + k3 * d * x

    k3_result = def_integral(
        k3_antiderivative,
        np.maximum(arrival_times_j_low, end_times_j),
        np.maximum(arrival_times_j_high, end_times_j),
        k3=k3, d=end_times_j
    )

    temporal_dists = k1 * A - (k1_result + k2_result + k3_result) / (arrival_times_j_high - arrival_times_j_low)
    '''without penalizing early arrival'''
    # temporal_dists = k1 * A - (k1_result + k1 * tw_widths_j + k3_result) / (arrival_times_j_high - arrival_times_j_low)
    '''without penalizing late arrival'''
    # temporal_dists = k1 * A - (k1_result + k2_result) / (arrival_times_j_high - arrival_times_j_low)

    return temporal_dists

