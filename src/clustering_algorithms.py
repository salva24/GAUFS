# This file provides implementations of various clustering algorithms to be used in GAUFS.

import numpy as np
from scipy.stats import chi2_contingency
from sklearn.cluster import AgglomerativeClustering, KMeans
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import euclidean_distances

class ClusteringExperiment:
    def __init__(self):
        self.data = None
        self.n_clusters=0
        # Algorithm instance
        self.algorithm = None
        # Result: assigned clusters
        self.assigned_clusters = None

    def set_unlabeled_data(self, data):
        self.data = data.copy()

    def run(self):
        # clusters should not be in the data yet but we drop it anyway
        x_clust = self.data
        self.algorithm.fit(x_clust)
        self.assigned_clusters = self.algorithm.labels_

    def get_dist_samples(self):
        value_counts = pd.DataFrame(self.assigned_clusters.value_counts())
        value_counts['relative'] = self.assigned_clusters.value_counts(normalize=True) * 100
        return value_counts.sort_index()
    
    def get_data_with_clusters(self):
        data_with_clusters = self.data.copy()
        if self.assigned_clusters is None:
            raise ValueError("Clustering has not been run yet. Please run the clustering algorithm first.")
        data_with_clusters['cluster'] = self.assigned_clusters
        return data_with_clusters
    
# Hierarchical Clustering
class HierarchicalExperiment(ClusteringExperiment):
    def __init__(self, data, linkage='ward'):
        self.data = data.copy()
        self.n_clusters = None
        self.linkage = linkage
        self.selected_columns = self.data.columns.to_list()

    def run(self):
        self.algorithm = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage)
        super().run()

# KMeans Clustering
class KmeansExperiment(ClusteringExperiment):
    def __init__(self, seed=None):
        self.n_clusters = None
        self.seed = seed
        self.selected_columns = self.data.columns.to_list()  

    def run(self):
        self.algorithm = KMeans(n_clusters=self.n_clusters, random_state=self.seed) if self.seed!= None else KMeans(n_clusters=self.n_clusters)
        super().run()



#To do: explore other clustering algorithms such as DBSCAN, MeanShift, Spectral Clustering, etc.
