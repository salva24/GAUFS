"""
base.py

Base class for clustering experiments in GAUFS.
Defines the common interface and functionality for all clustering
algorithm implementations.
Users can extend this class to implement custom clustering algorithms.
"""

import copy
import pandas as pd


class ClusteringExperiment:
    """
    Base class for clustering experiments.

    This class defines a common interface and shared functionality for
    unsupervised clustering experiments. Concrete subclasses are expected
    to configure the `algorithm` attribute with a clustering model that
    implements `fit()` and exposes `labels_` after fitting.

    Attributes
    ----------
    unlabeled_data : pandas.DataFrame or array-like
        Input data used for clustering.
    algorithm : object
        Clustering algorithm instance (e.g., KMeans, DBSCAN).
    assigned_clusters : pandas.Series or array-like
        Cluster labels assigned after running the experiment.
    """

    def __init__(self):
        self.unlabeled_data = None
        # Algorithm instance
        self.algorithm = None
        # Result: assigned clusters
        self.assigned_clusters = None

    def set_unlabeled_data(self, unlabeled_data):
        self.unlabeled_data = unlabeled_data.copy()

    def run(self):
        x_clust = self.unlabeled_data
        self.algorithm.fit(x_clust)
        self.assigned_clusters = self.algorithm.labels_

    def get_dist_samples(self):
        value_counts = pd.DataFrame(self.assigned_clusters.value_counts())
        value_counts["relative"] = (
            self.assigned_clusters.value_counts(normalize=True) * 100
        )
        return value_counts.sort_index()

    def get_data_with_clusters(self):
        data_with_clusters = self.unlabeled_data.copy()
        if self.assigned_clusters is None:
            raise ValueError(
                "Clustering has not been run yet. Please run the clustering algorithm first."
            )
        data_with_clusters["cluster"] = self.assigned_clusters
        return data_with_clusters

    def __copy__(self):
        """Shallow copy"""
        new_instance = type(self)()
        new_instance.__dict__.update(self.__dict__)
        return new_instance

    def __deepcopy__(self, memo):
        """Deep copy"""
        new_instance = type(self)()
        memo[id(self)] = new_instance
        for key, value in self.__dict__.items():
            setattr(new_instance, key, copy.deepcopy(value, memo))
        return new_instance


# To do: explore other clustering algorithms such as DBSCAN, MeanShift, Spectral Clustering, etc.
