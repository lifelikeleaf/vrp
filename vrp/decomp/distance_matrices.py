from functools import lru_cache
from scipy.spatial.distance import euclidean as scipy_euclidean

from . import helpers


### Public Section ###

def v1(feature_vectors, decomposer):
    '''V1: add
    spatial_temporal_distance = euclidean_dist + temporal_weight
    '''
    return _dist_matrix(feature_vectors, decomposer, _pairwise_dist_v1)


def v2(feature_vectors, decomposer):
    '''V2: simple overlap | simple gap'''
    return _dist_matrix(feature_vectors, decomposer, _pairwise_dist_v2)


def v3(feature_vectors, decomposer):
    '''V3: v1 overlap | simple gap'''
    return _dist_matrix(feature_vectors, decomposer, _pairwise_dist_v3)


def euclidean(feature_vectors, decomposer):
    return _dist_matrix(feature_vectors, decomposer, _pairwise_euclidean_dist)


### Private Section ###

class _PairwiseDistance:
    def __init__(self, fv_node_1, fv_node_2, decomposer) -> None:
        self.fv_node_1 = fv_node_1
        self.fv_node_2 = fv_node_2
        self.decomposer = decomposer
        self.euclidean_dist = 0


    def temporal_weight_v1(self):
        '''V1'''
        # fv_node_1 and fv_node_2 are feature_vectors from 2 different nodes,
        # in the format: [x, y, tw_start, tw_end]
        x1, y1, tw_start_1, tw_end_1 = self.fv_node_1
        x2, y2, tw_start_2, tw_end_2 = self.fv_node_2
        tw_width_1 = tw_end_1 - tw_start_1
        tw_width_2 = tw_end_2 - tw_start_2
        max_tw_width = max(tw_width_1, tw_width_2)

        self.euclidean_dist = scipy_euclidean([x1, y1], [x2, y2])
        overlap_or_gap = helpers.get_time_window_overlap_or_gap(
            [tw_start_1, tw_end_1],
            [tw_start_2, tw_end_2]
        )

        # temporal_weight is the penalty or reward applied to euclidean_dist
        # depending on whether there's a TW overlap or gap b/t 2 nodes
        temporal_weight = 0
        if overlap_or_gap >= 0:
            # there's a time window overlap between these 2 nodes
            overlap = overlap_or_gap
            temporal_weight = helpers.safe_divide(self.euclidean_dist, max_tw_width) * overlap
            # print(
            #     f'Overlap: {round(overlap, 2)} b/t nodes ({round(x1, 2)}, {round(y1, 2)}) and ({round(x2, 2)}, {round(y2, 2)}); '
            #     f'euclidean dist: {round(self.euclidean_dist, 2)}; temporal weight: {round(temporal_weight, 2)} '
            # )
            # print(
            #     f'dist v1 = {round(self.euclidean_dist+temporal_weight, 2)} \n'
            #     f'dist v2 = {round(self.euclidean_dist+overlap, 2)}'
            # )
            # print()
        elif self.decomposer.use_gap:
            # there's a time window gap between these 2 nodes
            assert overlap_or_gap < 0
            gap = overlap_or_gap
            # if wait time is not included in objective function,
            # then large gap b/t time windows is an advantage
            # bc it provides more flexibility for routing,
            # so leave it as a negative value will reduce
            # spatial_temporal_distance
            if self.decomposer.minimize_wait_time:
                # TODO: experiment with including wait time in OF
                # value - it's not considered by HGS solver and it's not
                # trivial to add it; check GOR Tools?
                # but if wait time IS included in objective function,
                # then large gap b/t time windows is a penalty,
                # so take its absolute value will increase
                # spatial_temporal_distance
                gap = abs(gap)

            temporal_weight = helpers.safe_divide(1, self.euclidean_dist) * gap
            # TODO: descriptive stats
            # print(
            #     f'Gap: {round(gap, 2)} b/t nodes ({round(x1, 2)}, {round(y1, 2)}) and ({round(x2, 2)}, {round(y2, 2)}); '
            #     f'euclidean dist: {round(self.euclidean_dist, 2)}; temporal weight: {round(temporal_weight, 2)} '
            # )
            # print(
            #     f'dist v1 = {round(self.euclidean_dist+temporal_weight, 2)} \n'
            #     f'dist v2 = {round(self.euclidean_dist+gap, 2)}'
            # )
            # print()

        return temporal_weight


    def temporal_weight_v2(self):
        '''V2: temporal_weight = simple overlap | simple gap'''
        x1, y1, tw_start_1, tw_end_1 = self.fv_node_1
        x2, y2, tw_start_2, tw_end_2 = self.fv_node_2
        tw_width_1 = tw_end_1 - tw_start_1
        tw_width_2 = tw_end_2 - tw_start_2
        max_tw_width = max(tw_width_1, tw_width_2)

        self.euclidean_dist = scipy_euclidean([x1, y1], [x2, y2])
        overlap_or_gap = helpers.get_time_window_overlap_or_gap(
            [tw_start_1, tw_end_1],
            [tw_start_2, tw_end_2]
        )

        temporal_weight = 0
        if overlap_or_gap >= 0:
            overlap = overlap_or_gap
            temporal_weight = overlap
        elif self.decomposer.use_gap:
            assert overlap_or_gap < 0
            gap = overlap_or_gap
            temporal_weight = gap

        return temporal_weight


    def temporal_weight_v3(self):
        '''V3: temporal_weight = v1 overlap | simple gap'''
        x1, y1, tw_start_1, tw_end_1 = self.fv_node_1
        x2, y2, tw_start_2, tw_end_2 = self.fv_node_2
        tw_width_1 = tw_end_1 - tw_start_1
        tw_width_2 = tw_end_2 - tw_start_2
        max_tw_width = max(tw_width_1, tw_width_2)

        self.euclidean_dist = scipy_euclidean([x1, y1], [x2, y2])
        overlap_or_gap = helpers.get_time_window_overlap_or_gap(
            [tw_start_1, tw_end_1],
            [tw_start_2, tw_end_2]
        )

        temporal_weight = 0
        if overlap_or_gap >= 0:
            overlap = overlap_or_gap
            temporal_weight = helpers.safe_divide(self.euclidean_dist, max_tw_width) * overlap
        elif self.decomposer.use_gap:
            assert overlap_or_gap < 0
            gap = overlap_or_gap
            temporal_weight = gap

        return temporal_weight


    def spatial_temporal_distance_v1(self):
        '''V1: add
        spatial_temporal_distance = euclidean_dist + temporal_weight
        '''
        temporal_weight = self.temporal_weight_v1()
        spatial_temporal_distance = self.euclidean_dist + temporal_weight
        # only non-negative distances are valid
        return max(0, spatial_temporal_distance)


    def spatial_temporal_distance_v2(self):
        '''V2: temporal_weight_v2'''
        temporal_weight = self.temporal_weight_v2()
        spatial_temporal_distance = self.euclidean_dist + temporal_weight
        return max(0, spatial_temporal_distance)


    def spatial_temporal_distance_v3(self):
        '''V3: temporal_weight_v3'''
        temporal_weight = self.temporal_weight_v3()
        spatial_temporal_distance = self.euclidean_dist + temporal_weight
        return max(0, spatial_temporal_distance)


def _pairwise_dist_v1(fv_node_1, fv_node_2, decomposer):
    pd = _PairwiseDistance(fv_node_1, fv_node_2, decomposer)
    return pd.spatial_temporal_distance_v1()


def _pairwise_dist_v2(fv_node_1, fv_node_2, decomposer):
    pd = _PairwiseDistance(fv_node_1, fv_node_2, decomposer)
    return pd.spatial_temporal_distance_v2()


def _pairwise_dist_v3(fv_node_1, fv_node_2, decomposer):
    pd = _PairwiseDistance(fv_node_1, fv_node_2, decomposer)
    return pd.spatial_temporal_distance_v3()


def _pairwise_euclidean_dist(fv_node_1, fv_node_2, decomposer):
    x1, y1, tw_start_1, tw_end_1 = fv_node_1
    x2, y2, tw_start_2, tw_end_2 = fv_node_2
    euclidean_dist = scipy_euclidean([x1, y1], [x2, y2])
    # print(f'Euclidean: {euclidean_dist}')
    return euclidean_dist


@lru_cache(maxsize=1)
def _dist_matrix(feature_vectors, decomposer, pairwise_dist_callable):
    dist_matrix = []

    for node_1 in feature_vectors.data:
        row = []
        for node_2 in feature_vectors.data:
            if node_1 == node_2:
                # dist to self should always be 0
                row.append(0)
            else:
                row.append(pairwise_dist_callable(node_1, node_2, decomposer))

        dist_matrix.append(row)

    return dist_matrix

