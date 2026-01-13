# MINERVA AI-Lab
# Institute of Computer Engineering
# University of Seville, Spain
#
# Copyright 2026 Salvador de la Torre Gonzalez
# Antonio Bello Castro,
# José M. Núñez Portero
#
# Developed and currently maintained by:
#    Salvador de la Torre Gonzalez
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#     SPDX-License-Identifier: Apache-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
External clustering evaluation metrics.

This module contains evaluation metrics that require ground truth
labels in addition to the cluster assignments.
These metrics are typically used to compare clustering results
against known labels.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn import metrics
from sklearn.metrics import confusion_matrix

from .base import EvaluationMetric
from .utils import _get_cont_table


class AdjustedMutualInformationScore(EvaluationMetric):
    def compute(self, assigned_clusters, unlabeled_data=None):
        ## AMI: Function that measures how well the optimal assignment corresponds to the one returned by the algorithm.
        # Based on entropy. Normalized and symmetric.
        # Bad assignments can take negative values, optimal assignment -> 1.
        return metrics.adjusted_mutual_info_score(self.true_labels, assigned_clusters)


class AdjustedRandIndexScore(EvaluationMetric):
    def compute(self, assigned_clusters, unlabeled_data=None):
        ## Coefficient that assigns 1. to the optimal prediction and values close to 0 the more random the predictions are
        return metrics.adjusted_rand_score(self.true_labels, assigned_clusters)


class VMeasureScore(EvaluationMetric):
    def compute(self, assigned_clusters, unlabeled_data=None):
        ## Combination of two criteria: homogeneity and completeness.
        # Homogeneity: each cluster contains only values corresponding to one label.
        # Completeness: all values corresponding to the same label are assigned to the same cluster.
        # The v-measure is a combination of both criteria, weighted by a beta value (less than one for more homogeneity,
        # greater than one for more completeness).
        # It's normalized and symmetric. Random assignment does not get a score of 0. (!)
        return metrics.v_measure_score(self.true_labels, assigned_clusters)


class FowlkesMallowsScore(EvaluationMetric):
    def compute(self, assigned_clusters, unlabeled_data=None):
        ## Geometric mean of precision and recall.
        # Random assignments have a value close to 0.
        # Normalized.
        return metrics.fowlkes_mallows_score(self.true_labels, assigned_clusters)


class FScore(EvaluationMetric):
    def compute(self, assigned_clusters, unlabeled_data=None):
        return metrics.f1_score(self.true_labels, assigned_clusters, average="weighted")


class NMIScore(EvaluationMetric):
    def compute(self, assigned_clusters, unlabeled_data=None):
        return metrics.normalized_mutual_info_score(self.true_labels, assigned_clusters)


class HScore(EvaluationMetric):
    def compute(self, assigned_clusters, unlabeled_data=None):
        true_labels = self.true_labels
        predicted_cluster = assigned_clusters
        confusion = confusion_matrix(true_labels, predicted_cluster)
        # Number of clusters
        k = confusion.shape[0]
        # Number of elements in the dataset
        n = np.sum(confusion)
        # Calculate the criterion H
        ch = 1 - (1 / n) * np.sum(np.max(confusion, axis=1))
        return ch


class Chi2(EvaluationMetric):
    def compute(self, assigned_clusters, unlabeled_data=None):
        table1 = _get_cont_table(assigned_clusters, self.true_labels, round=False)
        table2 = _get_cont_table(self.true_labels, assigned_clusters, round=False)
        chi2_1 = chi2_contingency(table1)[0]
        chi2_2 = chi2_contingency(table2)[0]
        return np.sum([chi2_1, chi2_2])


class DobPertScore(EvaluationMetric):
    def compute(self, assigned_clusters, unlabeled_data=None):
        cont_clust = _get_cont_table(assigned_clusters, self.true_labels, round=False)
        cont_event = _get_cont_table(self.true_labels, assigned_clusters, round=False)

        heuristic_table = pd.DataFrame(
            columns=["event_code", "heuristic_value"]
        ).set_index("event_code")
        for event in cont_event.index:
            clust_table = pd.DataFrame(
                columns=["cluster", "heuristic_value"]
            ).set_index("cluster")
            for clust in cont_clust.index:
                clust_table.loc[clust] = (
                    cont_clust.loc[clust, event] + cont_event.loc[event, clust]
                )

            if (clust_table["heuristic_value"] > 48).any():
                heuristic_table.loc[event] = clust_table["heuristic_value"].max()
        return heuristic_table["heuristic_value"].sum()
