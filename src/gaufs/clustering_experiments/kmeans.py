"""
kmeans.py

KMeans clustering experiment for GAUFS.
Implements the KmeansExperiment class, which
extends ClusteringExperiment using KMeans.
"""

from sklearn.cluster import KMeans

from .base import ClusteringExperiment


class KmeansExperiment(ClusteringExperiment):
    def __init__(self, unlabeled_data=None, seed=None):
        if unlabeled_data is not None:
            self.unlabeled_data = unlabeled_data.copy()
        self.n_clusters = None
        self.seed = seed

    def run(self):
        self.algorithm = (
            KMeans(n_clusters=self.n_clusters, random_state=self.seed)
            if self.seed != None
            else KMeans(n_clusters=self.n_clusters)
        )
        super().run()
