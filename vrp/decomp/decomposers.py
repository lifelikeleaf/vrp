import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation as AP
from sklearn.metrics import pairwise_distances
from sklearn_extra.cluster import KMedoids
from scipy.spatial.distance import euclidean

from .decomposition import AbstractDecomposer


class BaseDecomposer(AbstractDecomposer):
    """Abstract class that implements some common methods used by
    all decomposers implemented in this module.
    """
    def __init__(self, inst, num_clusters, include_tw=False) -> None:
        """

        Parameters
        ----------
        inst: `VRPInstance`
            A VRP problem instance.
        num_clusters: int
            Number of clusters to decompose the VRP problem instance into.
        include_tw: bool
            True if time windows should be included in features, else False.
            Default is False.

        """
        super().__init__(inst)
        self.num_clusters = num_clusters
        self.include_tw = include_tw
        # a list of feature vectors representing the customers
        # to be clustered, excluding the depot
        self.fv = self.build_feature_vectors()


    def build_feature_vectors(self):
        """Build feature vectors for clustering from VRP problem instance."""
        fv = []
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
            fv.append(row)

        # By CVRPLIB convention, index 0 is always depot;
        # depot should not be clustered
        return np.array(fv[1:])


    def get_clusters(self, labels):
        # array index in labels are customer IDs,
        # value at a given index is the cluster ID.
        clusters = [[] for i in range(self.num_clusters)]
        for i in range(len(labels)):
            # customer id is shifted by 1 bc index 0 is depot;
            # and depot is not clustered
            clusters[labels[i]].append(i + 1)

        return clusters


class KMeansDecomposer(BaseDecomposer):
    def decompose(self):
        # Run the k-means algorithm.
        # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

        print('Running k-means...')
        kmeans = KMeans(n_clusters=self.num_clusters, n_init=10).fit(self.fv)
        labels = kmeans.labels_
        return self.get_clusters(labels)

