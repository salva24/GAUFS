# This file provides implementations of various clustering algorithms to be used in GAUFS.

import numpy as np
from scipy.stats import chi2_contingency
from sklearn.cluster import AgglomerativeClustering, KMeans
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import euclidean_distances
import copy

class ClusteringExperiment:
    def __init__(self):
        self.unlabeled_data = None
        self.n_clusters=0
        # Algorithm instance
        self.algorithm = None
        # Result: assigned clusters
        self.assigned_clusters = None

    def set_unlabeled_data(self, unlabeled_data):
        self.unlabeled_data = unlabeled_data.copy()

    def run(self):
        # clusters should not be in the unlabeled_data yet but we drop it anyway
        x_clust = self.unlabeled_data
        self.algorithm.fit(x_clust)
        self.assigned_clusters = self.algorithm.labels_

    def get_dist_samples(self):
        value_counts = pd.DataFrame(self.assigned_clusters.value_counts())
        value_counts['relative'] = self.assigned_clusters.value_counts(normalize=True) * 100
        return value_counts.sort_index()
    
    def get_data_with_clusters(self):
        data_with_clusters = self.unlabeled_data.copy()
        if self.assigned_clusters is None:
            raise ValueError("Clustering has not been run yet. Please run the clustering algorithm first.")
        data_with_clusters['cluster'] = self.assigned_clusters
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
    
# Hierarchical Clustering
class HierarchicalExperiment(ClusteringExperiment):
    def __init__(self, unlabeled_data=None, linkage='ward'):
        if unlabeled_data is not None:
            self.unlabeled_data = unlabeled_data.copy()
        self.n_clusters = None
        # Linkage method: 'ward', 'complete', 'average', 'single'
        self.linkage = linkage

    def run(self):
        self.algorithm = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage)
        super().run()

# KMeans Clustering
class KmeansExperiment(ClusteringExperiment):
    def __init__(self, unlabeled_data=None, seed=None):
        if unlabeled_data is not None:
            self.unlabeled_data = unlabeled_data.copy()
        self.n_clusters = None
        self.seed = seed

    def run(self):
        self.algorithm = KMeans(n_clusters=self.n_clusters, random_state=self.seed) if self.seed!= None else KMeans(n_clusters=self.n_clusters)
        super().run()



#To do: explore other clustering algorithms such as DBSCAN, MeanShift, Spectral Clustering, etc.
