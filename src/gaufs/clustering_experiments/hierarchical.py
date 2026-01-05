"""
hierarchical.py

Hierarchical clustering experiment for GAUFS.
Implements the HierarchicalExperiment class, which
extends ClusteringExperiment using AgglomerativeClustering.
"""

from sklearn.cluster import AgglomerativeClustering

from .base import ClusteringExperiment


class HierarchicalExperiment(ClusteringExperiment):
    def __init__(self, unlabeled_data=None, linkage="ward"):
        if unlabeled_data is not None:
            self.unlabeled_data = unlabeled_data.copy()
        self.n_clusters = None
        # Linkage method: 'ward', 'complete', 'average', 'single'
        self.linkage = linkage

    def run(self):
        self.algorithm = AgglomerativeClustering(
            n_clusters=self.n_clusters, linkage=self.linkage
        )
        super().run()
