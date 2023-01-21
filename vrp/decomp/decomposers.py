from functools import lru_cache
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

    class FV():
        """lru_cache requires function arguments to be hashable.
        Wrap a feature_vectors list inside a user defined class
        to make it hashable.
        """
        def __init__(self, data: list) -> None:
            self.data = data


    """Abstract base class that implements some common methods used by
    all decomposers implemented in this module.
    """
    def __init__(self, name=None, num_clusters=2, use_tw=False, normalize=False) -> None:
        """
        Parameters
        ----------
        Optional:
            name: str
                The name of this decomposer. It can be used to identify
                an experiment.
                Default is None.

            num_clusters: int > 0
                Number of clusters to decompose the VRP problem instance into.
                Default is 2.

            use_tw: bool
                True if time windows should be included in features, else False.
                Default is False.

            normalize: bool
                True if feature vectors should be z standardized, else False.
                Default is False.

        """
        self.name = name
        self.num_clusters = num_clusters
        self.use_tw = use_tw
        self.normalize = normalize


    @staticmethod
    @lru_cache(maxsize=1)
    def build_feature_vectors(inst, use_tw, normalize) -> FV:
        """Build feature vectors for clustering from VRP problem instance.
        A list of feature vectors representing the customer nodes
        to be clustered, excluding the depot.
        """
        feature_vectors = []
        nodes = inst.nodes
        for i in range(len(nodes)):
            row = []
            # x, y coords for customer i
            row.append(nodes[i].x_coord)
            row.append(nodes[i].y_coord)
            if use_tw:
                # earliest service start time for customer i
                row.append(nodes[i].start_time)
                # lastest service start time for customer i
                row.append(nodes[i].end_time)
            feature_vectors.append(row)

        if normalize:
            feature_vectors = helpers.normalize_feature_vectors(feature_vectors)

        # By CVRPLIB convention, index 0 is always depot;
        # depot should not be clustered
        return __class__.FV(feature_vectors[1:])


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
        # TODO: dump to json?
        logger.debug(f'Clusters: \n{clusters}')

        return clusters


class BaseDistanceMatrixBasedDecomposer(BaseDecomposer):
    """Abstract base class for distance matrix based decomposers."""
    def __init__(
        self,
        name=None,
        num_clusters=2,
        use_tw=False,
        normalize=False,
        use_gap=False,
        minimize_wait_time=False
    ) -> None:
        """
        Parameters
        ----------
        Optional:
            use_gap: bool
                Whether to consider gap b/t time windows for temporal_weight.
                Default is False.

            minimize_wait_time: bool
                Whether to minimize wait time in objective function.
                Default is False.

        """
        super().__init__(name, num_clusters, use_tw, normalize)
        self.use_gap = use_gap
        self.minimize_wait_time = minimize_wait_time


    def compute_pairwise_spatial_temporal_distance(self, fv_node_1, fv_node_2):
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
            temporal_weight = helpers.safe_divide(euclidean_dist, max_tw_width) * overlap
        elif self.use_gap:
            # there's a time window gap between these 2 nodes
            assert overlap_or_gap < 0
            gap = overlap_or_gap
            # if wait time is not included in objective function,
            # then large gap b/t time windows is an advantage
            # bc it provides more flexibility for routing,
            # so leave it as a negative value will reduce
            # spatial_temporal_distance
            if self.minimize_wait_time:
                # TODO: experiment with including wait time in OF
                # value - it's not considered by HGS solver and it's not
                # trivial to add it; check GOR Tools?
                # but if wait time IS included in objective function,
                # then large gap b/t time windows is a penalty,
                # so take its absolute value will increase
                # spatial_temporal_distance
                gap = abs(gap)
            temporal_weight = helpers.safe_divide(1, euclidean_dist) * gap

        spatial_temporal_distance = euclidean_dist + temporal_weight

        # only non-negative values are valid
        # Precomputed distances need to have non-negative values,
        # else pairwise_distances() throws ValueError
        return max(0, spatial_temporal_distance)


    @staticmethod
    @lru_cache(maxsize=1)
    def compute_spatial_temporal_distance_matrix(feature_vectors, callable):
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html
        dist_matrix = pairwise_distances(feature_vectors.data, metric=callable)
        # TODO: dump to excel
        return dist_matrix


class KMeansDecomposer(BaseDecomposer):
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    @helpers.log_run_time
    def decompose(self, inst):
        feature_vectors = self.build_feature_vectors(inst, self.use_tw, self.normalize)
        logger.info('')
        logger.info('Running k-means...')
        kmeans = KMeans(n_clusters=self.num_clusters, n_init=10)
        kmeans.fit(feature_vectors.data)
        labels = kmeans.labels_
        return self.get_clusters(labels)


class KMedoidsDecomposer(BaseDistanceMatrixBasedDecomposer):
    # https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.KMedoids.html
    @helpers.log_run_time
    def decompose(self, inst):
        feature_vectors = self.build_feature_vectors(inst, self.use_tw, self.normalize)

        logger.info('')
        logger.info('Running k-medoids...')
        if self.use_tw:
            logger.info('using time windows...')
            # for 'precomputed' must pass the fit() method a distance matrix
            # instead of a feature vector
            metric = 'precomputed'
            dist_matrix = self.compute_spatial_temporal_distance_matrix(
                feature_vectors,
                self.compute_pairwise_spatial_temporal_distance
            )
            X = dist_matrix
        else:
            metric = 'euclidean' #  or a callable
            X = feature_vectors.data

        if self.use_gap:
            logger.info('and gap...')
        if self.minimize_wait_time:
            logger.info('and minimize wait time...')

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
    @helpers.log_run_time
    def decompose(self, inst):
        feature_vectors = self.build_feature_vectors(inst, self.use_tw, self.normalize)
        logger.info('')
        logger.info('Running Affinity Propogation...')
        if self.use_tw:
            logger.info('using time windows...')
            affinity = 'precomputed'
            # affinity matrix is the negative of distance matrix
            affinity_matrix = -1 * self.compute_spatial_temporal_distance_matrix(
                feature_vectors,
                self.compute_pairwise_spatial_temporal_distance
            )
            X = affinity_matrix
        else:
            affinity = 'euclidean'
            X = feature_vectors.data
        ap = AP(affinity=affinity).fit(X)
        labels = ap.labels_
        # AP doesn't need num_clusters as an initial parameter,
        # it finds a certain number of clusters based on the algorithm.
        # So self.num_clusters from the constructor may not be correct,
        # thus assign self.num_clusters to the actual number of clusters
        # found by AP.
        # self.num_clusters = len(ap.cluster_centers_indices_)
        return self.get_clusters(labels)

