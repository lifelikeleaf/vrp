from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation as AP
from sklearn_extra.cluster import KMedoids

from .decomposition import AbstractDecomposer
from . import helpers
from .logger import logger

logger = logger.getChild(__name__)

class BaseDecomposer(AbstractDecomposer):
    """Abstract base class that implements some common methods used by
    all decomposers implemented in this module.
    """
    def __init__(self, name=None, num_clusters=2, standardize=False) -> None:
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

            standardize: bool
                True if feature vectors should be z standardized, else False.
                Default is False.

        """
        self.name = name
        self.num_clusters = num_clusters
        self.standardize = standardize


    def get_clusters(self, labels, inertia=0):
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
        logger.debug(f'Clusters found (with inertia = {inertia}):')
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
        standardize=False,
        use_overlap=False,
        use_gap=False,
        normalize=False,
        minimize_wait_time=False,
    ) -> None:
        """
        Parameters
        ----------
        dist_matrix_func: Callable
            A callable that returns a distance matrix.

        Optional:
            use_overlap: bool
                Whether to consider overlap b/t time windows for spatial
                temporal distance calculation.
                Default is False.

            use_gap: bool
                Whether to consider gap b/t time windows for spatial
                temporal distance calculation.
                Default is False.

            normalize: bool
                True if min-max scaling should be used for distance matrix
                calculation, else False.
                Default is False.

            minimize_wait_time: bool
                Whether to minimize wait time in objective function.
                This flag should only be used if the underlying solver
                considers wait time in its objective function.
                Default is False.

        """
        super().__init__(name, num_clusters, standardize)
        self.dist_matrix_func = dist_matrix_func
        self.use_overlap = use_overlap
        self.use_gap = use_gap
        self.normalize = normalize
        self.minimize_wait_time = minimize_wait_time


class KMeansDecomposer(BaseDecomposer):
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    @helpers.log_run_time
    def decompose(self, inst):
        fv = helpers.build_feature_vectors(inst, standardize=False)
        logger.info('')
        logger.info('Running k-means...')
        kmeans = KMeans(n_clusters=self.num_clusters, n_init=10)
        kmeans.fit(fv.data)
        labels = kmeans.labels_
        return self.get_clusters(labels)


class KMedoidsDecomposer(BaseDistanceMatrixBasedDecomposer):
    # https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.KMedoids.html
    @helpers.log_run_time
    def decompose(self, inst):
        fv = helpers.build_feature_vectors(inst)

        logger.info('')
        logger.info('Running k-medoids...')
        logger.info(f'using dist_matrix_func {self.dist_matrix_func.__name__}...')
        if self.use_overlap:
            logger.info('considering overlap...')
        if self.use_gap:
            logger.info('considering gap...')
        if self.minimize_wait_time:
            logger.info('minimizing wait time...')

        # for 'precomputed' must pass the fit() method a distance matrix
        # instead of a feature vector
        metric = 'precomputed'
        X = self.dist_matrix_func(fv, self)
        method = 'pam'
        # {‘random’, ‘heuristic’, ‘k-medoids++’, ‘build’}, default='build'
        init = 'k-medoids++'
        max_iter = 600 # default = 300

        logger.debug(f'Num clusters pass into KMedoids: {self.num_clusters}')
        kmedoids = KMedoids(
            n_clusters=self.num_clusters,
            metric=metric,
            method=method,
            init=init,
            max_iter=max_iter,
        )
        kmedoids.fit(X)
        labels = kmedoids.labels_
        inertia = kmedoids.inertia_
        return self.get_clusters(labels, inertia)


class APDecomposer(BaseDistanceMatrixBasedDecomposer):
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html
    @helpers.log_run_time
    def decompose(self, inst):
        fv = helpers.build_feature_vectors(inst)
        logger.info('')
        logger.info('Running Affinity Propogation...')
        logger.info(f'using dist_matrix_func {self.dist_matrix_func.__name__}...')
        affinity = 'precomputed'
        # affinity matrix is the negative of distance matrix
        X = -1 * self.dist_matrix_func(fv, self)

        ap = AP(affinity=affinity).fit(X)
        labels = ap.labels_
        # AP doesn't need num_clusters as an initial parameter,
        # it finds a certain number of clusters based on the algorithm.
        # So self.num_clusters from the constructor may not be correct,
        # thus assign self.num_clusters to the actual number of clusters
        # found by AP.
        # self.num_clusters = len(ap.cluster_centers_indices_)
        return self.get_clusters(labels)

