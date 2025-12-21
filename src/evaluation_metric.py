# This file contains various supervised and unsupervised evaluation metrics for clustering algorithms.
import numpy as np
from scipy.stats import chi2_contingency
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import euclidean_distances

class EvaluationMetric:
    @staticmethod
    def get_cont_table_(c1, c2, round=True):
        res = pd.crosstab(c1, c2, normalize='index') * 100
        return res.round(1) if round else res
    
    @staticmethod
    def silhouette_score(unlabeled_data, assigned_clusters):
        ## Score between -1 and 1. (very dense clusters)
        return metrics.silhouette_score(unlabeled_data, assigned_clusters)

    @staticmethod
    def mutual_information_score(assigned_clusters, true_labels):
        ## AMI: Function that measures how well the optimal assignment corresponds to the one returned by the algorithm.
        # Based on entropy. Normalized and symmetric.
        # Bad assignments can take negative values, optimal assignment -> 1.
        return metrics.adjusted_mutual_info_score(true_labels.tolist(), assigned_clusters)

    @staticmethod
    def chi2(assigned_clusters, true_labels):
        table1 = EvaluationMetric.get_cont_table(assigned_clusters, true_labels, round=False)
        table2 = EvaluationMetric.get_cont_table(true_labels, assigned_clusters, round=False)
        chi2_1 = chi2_contingency(table1)[0]
        chi2_2 = chi2_contingency(table2)[0]
        return np.sum([chi2_1, chi2_2])

    @staticmethod
    def rand_index_score(assigned_clusters, true_labels):
        ## Coefficient that assigns 1. to the optimal prediction and values close to 0 the more random the predictions are
        return metrics.adjusted_rand_score(true_labels.tolist(), assigned_clusters)

    @staticmethod
    def v_measure_score(assigned_clusters, true_labels):
        ## Combination of two criteria: homogeneity and completeness.
        # Homogeneity: each cluster contains only values corresponding to one label.
        # Completeness: all values corresponding to the same label are assigned to the same cluster.
        # The v-measure is a combination of both criteria, weighted by a beta value (less than one for more homogeneity,
        # greater than one for more completeness).
        # It's normalized and symmetric. Random assignment does not get a score of 0. (!)
        return metrics.v_measure_score(true_labels.tolist(), assigned_clusters)

    @staticmethod
    def fowlkes_mallows_score(assigned_clusters, true_labels):
        ## Geometric mean of precision and recall.
        # Random assignments have a value close to 0.
        # Normalized.
        return metrics.fowlkes_mallows_score(true_labels.tolist(), assigned_clusters)

    @staticmethod
    def calinski_harabasz_score(unlabeled_data, assigned_clusters):
        ## Bad for DBSCAN
        return metrics.calinski_harabasz_score(unlabeled_data, assigned_clusters)

    @staticmethod
    def davies_bouldin_score(unlabeled_data, assigned_clusters):
        # Best values are the closest to 0. To maximize the score in GAUFS use davies_bouldin_score_for_maxamization
        return metrics.davies_bouldin_score(unlabeled_data, assigned_clusters)
    
    @staticmethod
    def davies_bouldin_score_for_maxamization(unlabeled_data, assigned_clusters):
        # To maximize the score in GAUFS we do 1 - davies_bouldin_score
        return 1- EvaluationMetric.davies_bouldin_score(unlabeled_data, assigned_clusters)

    @staticmethod
    def f_score(assigned_clusters, true_labels):
        return metrics.f1_score(list(true_labels),list( assigned_clusters),average='weighted')


    @staticmethod
    def dunn_score(unlabeled_data, assigned_clusters):
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
        
        points = unlabeled_data
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

    @staticmethod
    def dob_pert_score(assigned_clusters, true_labels):
        cont_clust = EvaluationMetric.get_cont_table(assigned_clusters, true_labels, round=False)
        cont_event = EvaluationMetric.get_cont_table(true_labels, assigned_clusters, round=False)

        heuristic_table = pd.DataFrame(columns=['event_code', 'heuristic_value']).set_index('event_code')
        for event in cont_event.index:
            clust_table = pd.DataFrame(columns=['cluster', 'heuristic_value']).set_index('cluster')
            for clust in cont_clust.index:
               clust_table.loc[clust] = cont_clust.loc[clust, event] + cont_event.loc[event, clust]

            if (clust_table['heuristic_value'] > 48).any():
                heuristic_table.loc[event] = clust_table['heuristic_value'].max()
        return heuristic_table['heuristic_value'].sum()

    @staticmethod
    def h_score(assigned_clusters, true_labels):
        true_labels = true_labels.tolist()
        predicted_cluster = assigned_clusters
        confusion = confusion_matrix(true_labels, predicted_cluster)
        # Number of clusters
        k = confusion.shape[0]
        # Number of elements in the dataset
        n = np.sum(confusion)
        # Calculate the criterion H
        ch = 1 - (1/n) * np.sum(np.max(confusion, axis=1))
        return ch
    
    @staticmethod
    def sse_score(unlabeled_data, assigned_clusters):
        # To maximize in GAUFS use sse_score_for_maximization
        # data with cluster assigned
        data_with_clusters_assigned = unlabeled_data.copy()
        data_with_clusters_assigned['cluster'] = assigned_clusters

        data = data.copy()
        variables = [col for col in data.columns if col != 'cluster']
        centroids = data.groupby('cluster')[variables].mean()
        sse = 0.0
        for cluster, centroid in centroids.iterrows():
            cluster_data = data[data['cluster'] == cluster][variables]
            centroid_array = centroid.values
            distances = np.square(cluster_data - centroid_array).sum(axis=1)
            sse += distances.sum()
        return sse

    @staticmethod
    def sse_score_for_maximization(unlabeled_data, assigned_clusters):
        # To maximize the score in GAUFS -sse_score
        return -1.*EvaluationMetric.sse_score(unlabeled_data, assigned_clusters)

    @staticmethod
    def nmi_score(assigned_clusters, true_labels):
        return metrics.normalized_mutual_info_score(true_labels,assigned_clusters)
