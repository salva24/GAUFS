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
hierarchical.py

Hierarchical clustering experiment for GAUFS.
Implements the HierarchicalExperiment class, which
extends ClusteringExperiment using AgglomerativeClustering.
"""

from sklearn.cluster import AgglomerativeClustering

from .base import ClusteringExperiment


class HierarchicalExperiment(ClusteringExperiment):
    """
    Hierarchical clustering experiment using Agglomerative Clustering.

    Parameters
    ----------
    unlabeled_data : pandas.DataFrame or pandas.Series, optional
        The unlabeled dataset to be clustered. If provided, a copy is stored.
    linkage : str, default='ward'
        Linkage criterion for hierarchical clustering. Options: 'ward',
        'complete', 'average', 'single'.

    Attributes
    ----------
    n_clusters : int or None
        Number of clusters to form. Must be set before calling run().
    linkage : str
        The linkage method used for clustering.
    """

    def __init__(self, unlabeled_data=None, linkage="ward"):
        """
        Initialize a HierarchicalExperiment instance.

        Parameters
        ----------
        unlabeled_data : pandas.DataFrame or pandas.Series, optional
            The unlabeled dataset to be clustered.
        linkage : str, default='ward'
            Linkage criterion for hierarchical clustering.

        """
        if unlabeled_data is not None:
            self.unlabeled_data = unlabeled_data.copy()
        self.n_clusters = None
        # Linkage method: 'ward', 'complete', 'average', 'single'
        self.linkage = linkage

    def run(self):
        """
        Execute hierarchical clustering with AgglomerativeClustering.

        Creates an AgglomerativeClustering instance with the specified
        number of clusters and linkage method, then runs the clustering.

        """
        self.algorithm = AgglomerativeClustering(
            n_clusters=self.n_clusters, linkage=self.linkage
        )
        super().run()
