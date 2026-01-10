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
kmeans.py

KMeans clustering experiment for GAUFS.
Implements the KmeansExperiment class, which
extends ClusteringExperiment using KMeans.
"""

from sklearn.cluster import KMeans

from .base import ClusteringExperiment


class KmeansExperiment(ClusteringExperiment):
    """
    K-means clustering experiment.

    Parameters
    ----------
    unlabeled_data : pandas.DataFrame or pandas.Series, optional
        The unlabeled dataset to be clustered. If provided, a copy is stored.
    seed : int, optional
        Random state seed for reproducibility. If None, results are non-deterministic.

    Attributes
    ----------
    n_clusters : int or None
        Number of clusters to form. Must be set before calling run().
    seed : int or None
        Random state seed used for the KMeans algorithm.

    """

    def __init__(self, unlabeled_data=None, seed=None):
        """
        Initialize a KmeansExperiment instance.

        Parameters
        ----------
        unlabeled_data : pandas.DataFrame or pandas.Series, optional
            The unlabeled dataset to be clustered.
        seed : int, optional
            Random state seed for reproducibility.
        """
        if unlabeled_data is not None:
            self.unlabeled_data = unlabeled_data.copy()
        self.n_clusters = None
        self.seed = seed

    def run(self):
        """
        Execute K-means clustering.

        Creates a KMeans instance with the specified number of clusters
        and optional random state, then runs the clustering.

        """
        self.algorithm = (
            KMeans(n_clusters=self.n_clusters, random_state=self.seed)
            if self.seed != None
            else KMeans(n_clusters=self.n_clusters)
        )
        super().run()
