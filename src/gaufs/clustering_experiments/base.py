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
base.py

Base class for clustering experiments in GAUFS.
Defines the common interface and functionality for all clustering
algorithm implementations.
Users can extend this class to implement custom clustering algorithms.
"""

import copy
import pandas as pd


class ClusteringExperiment:
    """
    Base class for clustering experiments.

    This class defines a common interface and shared functionality for
    unsupervised clustering experiments. Concrete subclasses are expected
    to configure the `algorithm` attribute with a clustering model that
    implements `fit()` and exposes `labels_` after fitting.

    Attributes
    ----------
    unlabeled_data : pandas.DataFrame or array-like
        Input data used for clustering.
    algorithm : object
        Clustering algorithm instance (e.g., KMeans, DBSCAN).
    assigned_clusters : pandas.Series or array-like
        Cluster labels assigned after running the experiment.
    """

    def __init__(self):
        """
        Initialize a new ClusteringExperiment instance.

        Sets all attributes to None, including unlabeled_data, algorithm instance,
        and assigned_clusters results.

        """
        self.unlabeled_data = None
        # Algorithm instance
        self.algorithm = None
        # Result: assigned clusters
        self.assigned_clusters = None

    def set_unlabeled_data(self, unlabeled_data):
        """
        Set the unlabeled data for clustering.

        Parameters
        ----------
        unlabeled_data : pandas.DataFrame or pandas.Series
            The unlabeled dataset to be clustered. A copy is stored to avoid
            modifying the original data.

        """
        self.unlabeled_data = unlabeled_data.copy()

    def run(self):
        """
        Execute the clustering algorithm on the unlabeled data.

        Fits the clustering algorithm to the unlabeled data and stores the
        resulting cluster labels in the assigned_clusters attribute.

        Notes
        -----
        This method assumes that both unlabeled_data and algorithm have been
        properly set. The cluster labels are stored in :attr:`assigned_clusters`.

        See Also
        --------
        assigned_clusters : The attribute where cluster labels are stored
        """
        x_clust = self.unlabeled_data
        self.algorithm.fit(x_clust)
        self.assigned_clusters = self.algorithm.labels_

    def get_dist_samples(self):
        """
        Get the distribution of samples across clusters.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with cluster labels as index and two columns:
            - First column: absolute count of samples in each cluster
            - 'relative': percentage of samples in each cluster
            The DataFrame is sorted by cluster label index.

        Notes
        -----
        This method should be called after run() has been executed to ensure
        assigned_clusters contains valid cluster labels.

        """
        value_counts = pd.DataFrame(self.assigned_clusters.value_counts())
        value_counts["relative"] = (
            self.assigned_clusters.value_counts(normalize=True) * 100
        )
        return value_counts.sort_index()

    def get_data_with_clusters(self):
        """
        Get the unlabeled data with assigned cluster labels.

        Returns
        -------
        pandas.DataFrame
            Copy of unlabeled data with added 'cluster' column.

        Raises
        ------
        ValueError
            If clustering has not been run yet.

        """
        data_with_clusters = self.unlabeled_data.copy()
        if self.assigned_clusters is None:
            raise ValueError(
                "Clustering has not been run yet. Please run the clustering algorithm first."
            )
        data_with_clusters["cluster"] = self.assigned_clusters
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


# To do: explore other clustering algorithms such as DBSCAN, MeanShift, Spectral Clustering, etc.
