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
Base classes for clustering evaluation metrics in GAUFS.

This module defines the EvaluationMetric base class, which provides
a common interface for all clustering evaluation metrics.
Users can extend this class to implement custom evaluation metrics.
"""


class EvaluationMetric:
    """
    Base class for clustering evaluation metrics.
    All metrics share the same signature: compute(assigned_clusters)
    """

    def __init__(self, true_labels=None):
        """
        Initialize the metric with data and labels that won't change.

        Parameters
        ----------
        true_labels : array-like or None, optional
            Ground truth labels. Required only for external metrics.
            Can be left as None for internal metrics. Default is None.
        """
        self.true_labels = true_labels

    def compute(self, assigned_clusters, unlabeled_data=None):
        """
        Compute the metric given cluster assignments.

        Must be implemented by each metric subclass.

        Parameters
        ----------
        assigned_clusters : array-like
            The cluster assignments to evaluate.
        unlabeled_data : pd.DataFrame or None, optional
            The original data. Required for internal metrics.
            If the metric does not need it, it can be left as None.
            Default is None.

        Returns
        -------
        float
            The metric score.
        """
        raise NotImplementedError("Subclasses must implement compute()")
