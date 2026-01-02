# This file contains various supervised and unsupervised evaluation metrics for clustering algorithms.
import numpy as np
from scipy.stats import chi2_contingency
from src.utils import get_cont_table
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import euclidean_distances


class EvaluationMetric:
    """
    Base class for clustering evaluation metrics.
    All metrics share the same signature: compute(assigned_clusters)
    """
    
    def __init__(self, unlabeled_data=None, true_labels=None):
        """
        Initialize the metric with data and labels that won't change.
        
        Args:
            unlabeled_data: The original data (DataFrame). Required for internal metrics.
            true_labels: Ground truth labels. Required only for external metrics. It can be left as None for internal metrics.
        """
        self.unlabeled_data = unlabeled_data
        self.true_labels = true_labels
    
    def compute(self, assigned_clusters):
        """
        Compute the metric given cluster assignments.
        Must be implemented by each metric subclass.
        
        Args:
            assigned_clusters: The cluster assignments to evaluate
            
        Returns:
            float: The metric score
        """
        raise NotImplementedError("Subclasses must implement compute()")




# Internal metrics (only need unlabeled_data)

class SilhouetteScore(EvaluationMetric):
    def compute(self, assigned_clusters):
        ## Score between -1 and 1. (very dense clusters)
        return metrics.silhouette_score(self.unlabeled_data, assigned_clusters)


class CalinskiHarabaszScore(EvaluationMetric):
    def compute(self, assigned_clusters):
        ## Bad for DBSCAN
        return metrics.calinski_harabasz_score(self.unlabeled_data, assigned_clusters)


class DaviesBouldinScore(EvaluationMetric):
    def compute(self, assigned_clusters):
        # Best values are the closest to 0. To maximize the score in GAUFS use davies_bouldin_score_for_maxamization
        return metrics.davies_bouldin_score(self.unlabeled_data, assigned_clusters)


class DaviesBouldinScoreForMaxamization(EvaluationMetric):
    def compute(self, assigned_clusters):
        # To maximize the score in GAUFS we do 1 - davies_bouldin_score
        return 1 - metrics.davies_bouldin_score(self.unlabeled_data, assigned_clusters)


class DunnScore(EvaluationMetric):
    def compute(self, assigned_clusters):
        ### Implementation of Dunn score, source: https://github.com/jqmviegas/jqm_cvi/blob/master/jqmcvi/base.py

        def delta_fast(ck, cl, distances):
            # First auxiliar funcion
            values = distances[np.where(ck)][:, np.where(cl)]
            values = values[np.nonzero(values)]          

            return np.min(values) if values.size != 0 else 0.
        
        def big_delta_fast(ci, distances):
            #Second auxiliar funcion
            values = distances[np.where(ci)][:, np.where(ci)]
            #values = values[np.nonzero(values)]
            epsilon=10.**-20
            res=np.max(values)
            return res if res > epsilon else epsilon # avoid dividing by 0
        
        points = self.unlabeled_data
        labels = assigned_clusters
        distances = euclidean_distances(points)
        ks = np.sort(np.unique(labels))
        
        deltas = np.ones([len(ks), len(ks)])*1000000
        big_deltas = np.zeros([len(ks), 1])
        
        l_range = list(range(0, len(ks)))
        
        for k in l_range:
            for l in (l_range[0:k]+l_range[k+1:]):
                deltas[k, l] = delta_fast((labels == ks[k]), (labels == ks[l]), distances)
            
            big_deltas[k] = big_delta_fast((labels == ks[k]), distances)

        di = np.min(deltas)/np.max(big_deltas)
        return di


class SSEScore(EvaluationMetric):
    def compute(self, assigned_clusters):
        # To maximize in GAUFS use sse_score_for_maximization
        # data with cluster assigned
        data_with_clusters_assigned = self.unlabeled_data.copy()
        data_with_clusters_assigned['cluster'] = assigned_clusters

        data = data_with_clusters_assigned.copy()
        variables = [col for col in data.columns if col != 'cluster']
        centroids = data.groupby('cluster')[variables].mean()
        sse = 0.0
        for cluster, centroid in centroids.iterrows():
            cluster_data = data[data['cluster'] == cluster][variables]
            centroid_array = centroid.values
            distances = np.square(cluster_data - centroid_array).sum(axis=1)
            sse += distances.sum()
        return sse


class SSEScoreForMaximization(EvaluationMetric):
    def compute(self, assigned_clusters):
        # To maximize the score in GAUFS -sse_score
        data_with_clusters_assigned = self.unlabeled_data.copy()
        data_with_clusters_assigned['cluster'] = assigned_clusters

        data = data_with_clusters_assigned.copy()
        variables = [col for col in data.columns if col != 'cluster']
        centroids = data.groupby('cluster')[variables].mean()
        sse = 0.0
        for cluster, centroid in centroids.iterrows():
            cluster_data = data[data['cluster'] == cluster][variables]
            centroid_array = centroid.values
            distances = np.square(cluster_data - centroid_array).sum(axis=1)
            sse += distances.sum()
        return -1.*sse


# External metrics (need true_labels)

class AdjustedMutualInformationScore(EvaluationMetric):
    def compute(self, assigned_clusters):
        ## AMI: Function that measures how well the optimal assignment corresponds to the one returned by the algorithm.
        # Based on entropy. Normalized and symmetric.
        # Bad assignments can take negative values, optimal assignment -> 1.
        return metrics.adjusted_mutual_info_score(self.true_labels, assigned_clusters)


class RandIndexScore(EvaluationMetric):
    def compute(self, assigned_clusters):
        ## Coefficient that assigns 1. to the optimal prediction and values close to 0 the more random the predictions are
        return metrics.adjusted_rand_score(self.true_labels, assigned_clusters)


class VMeasureScore(EvaluationMetric):
    def compute(self, assigned_clusters):
        ## Combination of two criteria: homogeneity and completeness.
        # Homogeneity: each cluster contains only values corresponding to one label.
        # Completeness: all values corresponding to the same label are assigned to the same cluster.
        # The v-measure is a combination of both criteria, weighted by a beta value (less than one for more homogeneity,
        # greater than one for more completeness).
        # It's normalized and symmetric. Random assignment does not get a score of 0. (!)
        return metrics.v_measure_score(self.true_labels, assigned_clusters)


class FowlkesMallowsScore(EvaluationMetric):
    def compute(self, assigned_clusters):
        ## Geometric mean of precision and recall.
        # Random assignments have a value close to 0.
        # Normalized.
        return metrics.fowlkes_mallows_score(self.true_labels, assigned_clusters)


class FScore(EvaluationMetric):
    def compute(self, assigned_clusters):
        return metrics.f1_score(self.true_labels, assigned_clusters, average='weighted')


class NMIScore(EvaluationMetric):
    def compute(self, assigned_clusters):
        return metrics.normalized_mutual_info_score(self.true_labels, assigned_clusters)


class HScore(EvaluationMetric):
    def compute(self, assigned_clusters):
        true_labels = self.true_labels
        predicted_cluster = assigned_clusters
        confusion = confusion_matrix(true_labels, predicted_cluster)
        # Number of clusters
        k = confusion.shape[0]
        # Number of elements in the dataset
        n = np.sum(confusion)
        # Calculate the criterion H
        ch = 1 - (1/n) * np.sum(np.max(confusion, axis=1))
        return ch


class Chi2(EvaluationMetric):
    def compute(self, assigned_clusters):
        table1 = get_cont_table(assigned_clusters, self.true_labels, round=False)
        table2 = get_cont_table(self.true_labels, assigned_clusters, round=False)
        chi2_1 = chi2_contingency(table1)[0]
        chi2_2 = chi2_contingency(table2)[0]
        return np.sum([chi2_1, chi2_2])


class DobPertScore(EvaluationMetric):
    def compute(self, assigned_clusters):
        cont_clust = get_cont_table(assigned_clusters, self.true_labels, round=False)
        cont_event = get_cont_table(self.true_labels, assigned_clusters, round=False)

        heuristic_table = pd.DataFrame(columns=['event_code', 'heuristic_value']).set_index('event_code')
        for event in cont_event.index:
            clust_table = pd.DataFrame(columns=['cluster', 'heuristic_value']).set_index('cluster')
            for clust in cont_clust.index:
               clust_table.loc[clust] = cont_clust.loc[clust, event] + cont_event.loc[event, clust]

            if (clust_table['heuristic_value'] > 48).any():
                heuristic_table.loc[event] = clust_table['heuristic_value'].max()
        return heuristic_table['heuristic_value'].sum()