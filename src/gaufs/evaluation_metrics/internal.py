"""
Internal clustering evaluation metrics.

This module contains evaluation metrics that rely only on the
unlabeled input data and the cluster assignments.
These metrics do not require ground truth labels.
"""

import numpy as np
from sklearn import metrics
from sklearn.metrics.pairwise import euclidean_distances

from .base import EvaluationMetric


class SilhouetteScore(EvaluationMetric):
    def compute(self, assigned_clusters, unlabeled_data=None):
        ## Score between -1 and 1. (very dense clusters)
        return metrics.silhouette_score(unlabeled_data, assigned_clusters)


class CalinskiHarabaszScore(EvaluationMetric):
    def compute(self, assigned_clusters, unlabeled_data=None):
        ## Bad for DBSCAN
        return metrics.calinski_harabasz_score(unlabeled_data, assigned_clusters)


class DaviesBouldinScore(EvaluationMetric):
    def compute(self, assigned_clusters, unlabeled_data=None):
        # Best values are the closest to 0. To maximize the score in GAUFS use davies_bouldin_score_for_maxamization
        return metrics.davies_bouldin_score(unlabeled_data, assigned_clusters)


class DaviesBouldinScoreForMaximization(EvaluationMetric):
    def compute(self, assigned_clusters, unlabeled_data=None):
        # To maximize the score in GAUFS we do 1 - davies_bouldin_score
        return 1 - metrics.davies_bouldin_score(unlabeled_data, assigned_clusters)


class DunnScore(EvaluationMetric):
    def compute(self, assigned_clusters, unlabeled_data=None):
        ### Implementation of Dunn score, source: https://github.com/jqmviegas/jqm_cvi/blob/master/jqmcvi/base.py

        def delta_fast(ck, cl, distances):
            # First auxiliar funcion
            values = distances[np.where(ck)][:, np.where(cl)]
            values = values[np.nonzero(values)]

            return np.min(values) if values.size != 0 else 0.0

        def big_delta_fast(ci, distances):
            # Second auxiliar funcion
            values = distances[np.where(ci)][:, np.where(ci)]
            # values = values[np.nonzero(values)]
            epsilon = 10.0**-20
            res = np.max(values)
            return res if res > epsilon else epsilon  # avoid dividing by 0

        points = unlabeled_data
        labels = assigned_clusters
        distances = euclidean_distances(points)
        ks = np.sort(np.unique(labels))

        deltas = np.ones([len(ks), len(ks)]) * 1000000
        big_deltas = np.zeros([len(ks), 1])

        l_range = list(range(0, len(ks)))

        for k in l_range:
            for l in l_range[0:k] + l_range[k + 1 :]:
                deltas[k, l] = delta_fast(
                    (labels == ks[k]), (labels == ks[l]), distances
                )

            big_deltas[k] = big_delta_fast((labels == ks[k]), distances)

        di = np.min(deltas) / np.max(big_deltas)
        return di


class SSEScore(EvaluationMetric):
    def compute(self, assigned_clusters, unlabeled_data=None):
        # To maximize in GAUFS use sse_score_for_maximization
        # data with cluster assigned
        data_with_clusters_assigned = unlabeled_data.copy()
        data_with_clusters_assigned["cluster"] = assigned_clusters

        data = data_with_clusters_assigned.copy()
        variables = [col for col in data.columns if col != "cluster"]
        centroids = data.groupby("cluster")[variables].mean()
        sse = 0.0
        for cluster, centroid in centroids.iterrows():
            cluster_data = data[data["cluster"] == cluster][variables]
            centroid_array = centroid.values
            distances = np.square(cluster_data - centroid_array).sum(axis=1)
            sse += distances.sum()
        return sse


class SSEScoreForMaximization(EvaluationMetric):
    def compute(self, assigned_clusters, unlabeled_data=None):
        # To maximize the score in GAUFS -sse_score
        data_with_clusters_assigned = unlabeled_data.copy()
        data_with_clusters_assigned["cluster"] = assigned_clusters

        data = data_with_clusters_assigned.copy()
        variables = [col for col in data.columns if col != "cluster"]
        centroids = data.groupby("cluster")[variables].mean()
        sse = 0.0
        for cluster, centroid in centroids.iterrows():
            cluster_data = data[data["cluster"] == cluster][variables]
            centroid_array = centroid.values
            distances = np.square(cluster_data - centroid_array).sum(axis=1)
            sse += distances.sum()
        return -1.0 * sse
