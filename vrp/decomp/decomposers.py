import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation as AP
from sklearn.metrics import pairwise_distances
from sklearn_extra.cluster import KMedoids
from scipy.spatial.distance import euclidean

from .decomposition import AbstractDecomposer
from . import helpers
from .logger import logger

logger = logger.getChild(__name__)

class BaseDecomposer(AbstractDecomposer):
    """Abstract base class that implements some common methods used by
    all decomposers implemented in this module.
    """
    def __init__(self, inst, num_clusters=2, include_tw=False) -> None:
        """

        Parameters
        ----------
        inst: `VRPInstance`
            A VRP problem instance.

        Optional:
            num_clusters: int > 0
                Number of clusters to decompose the VRP problem instance into.
                Default is 2 - anything less would be equivalent to the orginal
                problem without decomposition.
            include_tw: bool
                True if time windows should be included in features, else False.
                Default is False.

        """
        super().__init__(inst)
        # TODO: make sure num_clusters > 0
        self.num_clusters = num_clusters
        self.include_tw = include_tw
        # a list of feature vectors representing the customer nodes
        # to be clustered, excluding the depot
        self.feature_vectors = self.build_feature_vectors()


    def build_feature_vectors(self):
        """Build feature vectors for clustering from VRP problem instance."""
        feature_vectors = []
        nodes = self.inst.nodes
        for i in range(len(nodes)):
            row = []
            # x, y coords for customer i
            row.append(nodes[i].x_coord)
            row.append(nodes[i].y_coord)
            if self.include_tw:
                # earliest service start time for customer i
                row.append(nodes[i].start_time)
                # lastest service start time for customer i
                row.append(nodes[i].end_time)
            feature_vectors.append(row)

        # By CVRPLIB convention, index 0 is always depot;
        # depot should not be clustered
        return np.array(feature_vectors[1:])


    def get_clusters(self, labels):
        # array index in labels are customer IDs,
        # value at a given index is the cluster ID.

        # labels contains info on the actual number of clusters found,
        # whereas self.num_clusters is a user provided param.
        num_clusters = len(set(labels))
        clusters = [[] for i in range(num_clusters)]
        for i in range(len(labels)):
            # customer id is shifted by 1 bc index 0 is depot;
            # and depot is not clustered
            clusters[labels[i]].append(i + 1)

        logger.info(f'Num clusters: {num_clusters}')
        logger.info(f'Clusters: \n{clusters}')

        return clusters


class BaseDistanceMatrixBasedDecomposer(BaseDecomposer):
    """Abstract base class for distance matrix based decomposers."""
    def __init__(self, inst, num_clusters=2, include_tw=False,
                 use_gap=False, minimize_wait_time=False) -> None:
        super().__init__(inst, num_clusters, include_tw)
        # whether to consider gap b/t time windows for temporal_weight
        self.use_gap = use_gap
        # whether to minimize waiting time in objective function
        self.minimize_wait_time = minimize_wait_time


    def compute_pairwise_spatial_temportal_distance(self, fv_node_1, fv_node_2):
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html
        # The callable should take two arrays from X as input and return
        # a value indicating the distance between them.

        # fv_node_1 and fv_node_2 are feature_vectors from 2 different nodes,
        # in the format: [x, y, tw_start, tw_end]
        # see `self.build_feature_vectors()`
        x1, y1, tw_start_1, tw_end_1 = fv_node_1
        x2, y2, tw_start_2, tw_end_2 = fv_node_2
        tw_width_1 = tw_end_1 - tw_start_1
        tw_width_2 = tw_end_2 - tw_start_2
        max_tw_width = max(tw_width_1, tw_width_2)

        euclidean_dist = euclidean([x1, y1], [x2, y2])
        overlap_or_gap = helpers.get_time_window_overlap_or_gap(
            [tw_start_1, tw_end_1],
            [tw_start_2, tw_end_2]
        )
        temporal_weight = 0
        if overlap_or_gap >= 0:
            # there's a time window overlap between these 2 nodes
            overlap = overlap_or_gap
            temporal_weight = (euclidean_dist / max_tw_width) * overlap
        elif self.use_gap:
            # there's a time window gap between these 2 nodes
            assert overlap_or_gap < 0
            gap = overlap_or_gap
            # if waiting time is not included in objective function,
            # then large gap b/t time windows is an advantage
            # bc it provides more flexibility for routing,
            # so leave it as a negative value will reduce
            # spatial_temportal_distance
            if self.minimize_wait_time:
                # TODO: experiment with including waiting time in solution
                # value - currently it's not included
                # but if waiting time IS included in objective function,
                # then large gap b/t time windows is a penalty,
                # so take its absolute value will increase
                # spatial_temportal_distance
                gap = abs(gap)
            temporal_weight = (1 / euclidean_dist) * gap

        spatial_temportal_distance = euclidean_dist + temporal_weight

        # only non-negative values are valid
        return max(0, spatial_temportal_distance)


    def compute_spatial_temportal_distance_matrix(self):
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html
        metric_callable = self.compute_pairwise_spatial_temportal_distance
        return pairwise_distances(self.feature_vectors, metric=metric_callable)


class KMeansDecomposer(BaseDecomposer):
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    def decompose(self):
        logger.info('Running k-means...')
        kmeans = KMeans(n_clusters=self.num_clusters, n_init=10)
        kmeans.fit(self.feature_vectors)
        labels = kmeans.labels_
        return self.get_clusters(labels)


class KMedoidsDecomposer(BaseDistanceMatrixBasedDecomposer):
    # https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.KMedoids.html
    def decompose(self):
        logger.info('Running k-medoids...')
        if self.include_tw:
            logger.info('using time windows...')
            # for 'precomputed' must pass the fit() method a distance matrix
            # instead of a feature vector
            metric = 'precomputed'
            dist_matrix = self.compute_spatial_temportal_distance_matrix()
            X = dist_matrix
        else:
            metric = 'euclidean' #  or a callable
            X = self.feature_vectors
        method = 'pam'
        # {‘random’, ‘heuristic’, ‘k-medoids++’, ‘build’}, default='build'
        init = 'k-medoids++'
        kmedoids = KMedoids(
            n_clusters=self.num_clusters,
            metric=metric,
            method=method,
            init=init
        )
        kmedoids.fit(X)
        labels = kmedoids.labels_
        return self.get_clusters(labels)


class APDecomposer(BaseDistanceMatrixBasedDecomposer):
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html
    def decompose(self):
        logger.info('Running Affinity Propogation...')
        if self.include_tw:
            logger.info('using time windows...')
            affinity = 'precomputed'
            # affinity matrix is the negative of distance matrix
            affinity_matrix = -1 * self.compute_spatial_temportal_distance_matrix()
            X = affinity_matrix
        else:
            affinity = 'euclidean'
            X = self.feature_vectors
        ap = AP(affinity=affinity).fit(X)
        labels = ap.labels_
        # AP doesn't need num_clusters as an initial parameter,
        # it finds a certain number of clusters based on the algorithm.
        # So self.num_clusters from the constructor may not be correct,
        # thus assign self.num_clusters to the actual number of clusters
        # found by AP.
        # self.num_clusters = len(ap.cluster_centers_indices_)
        return self.get_clusters(labels)

