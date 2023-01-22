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
        self.use_tw = use_tw # TODO: remove
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
            # TODO: remove this check and return consistent FV format
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

        # NOTE: the set of labels are not guaranteed to be consecutive
        # natural numbers, like [0, 1, 2, 3], even though most of the time
        # they are. But sometimes it skips an integer, like [0, 1, 3, 4],
        # which causes a hard-to-reproduce 'IndexError: list index out of range'
        # if the implementation used a simple list with the assumption
        # of consecutive natural numbers as indices.
        dict_clusters = {cluster_id: [] for cluster_id in set(labels)}

        logger.debug(f'Clusters container: {dict_clusters}')
        logger.debug(f'Num customers: {len(labels)}')
        for i in range(len(labels)):
            cluster_id = labels[i]
            # customer id is shifted by 1 bc index 0 is depot;
            # and depot is not clustered
            dict_clusters[cluster_id].append(i + 1)

        list_clusters = []
        for key in dict_clusters:
            list_clusters.append(dict_clusters[key])

        assert num_clusters == len(list_clusters)
        logger.info(f'Num clusters found: {num_clusters}')
        # TODO: dump to json?
        logger.debug(f'Clusters found:')
        for i, cluster in enumerate(list_clusters):
            logger.debug(f'cluster {i}: \n{cluster}')

        return list_clusters


class BaseDistanceMatrixBasedDecomposer(BaseDecomposer):
    """Abstract base class for distance matrix based decomposers."""
    def __init__(
        self,
        dist_matrix_func,
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
        dist_matrix_func: Callable
            A callable that returns a distance matrix.

        Optional:
            use_gap: bool
                Whether to consider gap b/t time windows for temporal_weight.
                Default is False.

            minimize_wait_time: bool
                Whether to minimize wait time in objective function.
                Default is False.

        """
        super().__init__(name, num_clusters, use_tw, normalize)
        self.dist_matrix_func = dist_matrix_func
        self.use_gap = use_gap
        self.minimize_wait_time = minimize_wait_time


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
            X = self.dist_matrix_func(feature_vectors, self)
            # TODO: dump dist_matrix to excel
            # instance_name
            # experiment_name = self.name
        else:
            # TODO: use DM.euclidean
            metric = 'euclidean' #  or a callable
            X = feature_vectors.data

        if self.use_gap:
            logger.info('and gap...')
        if self.minimize_wait_time:
            logger.info('and minimize wait time...')

        method = 'pam'
        # {‘random’, ‘heuristic’, ‘k-medoids++’, ‘build’}, default='build'
        init = 'k-medoids++'

        logger.debug(f'Num clusters pass into KMedoids: {self.num_clusters}')
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
            X = -1 * self.dist_matrix_func(feature_vectors, self)
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

