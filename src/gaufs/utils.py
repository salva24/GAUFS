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
Internal utilities for GAUFS.

This module contains helper functions used internally by the library.
These functions are not part of the public API, except for data loading.

In addition, it provides the following functions for users:
- `read_unlabeled_data_csv`
- `read_labeled_data_csv`
"""

import pandas as pd
import numpy as np
import copy
import os
import shutil


# ----------------------------
# Public functions for users
# ----------------------------
def read_unlabeled_data_csv(filepath):
    """
    Reads unlabeled data from a CSV file.

    Accepts CSVs with or without a header.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the unlabeled data of shape (n_samples, n_features).

    """
    # Try reading with header
    df = pd.read_csv(filepath)

    # If the first row is numeric, there was no real header
    if df.columns.to_list()[0].startswith("Unnamed") or all(
        c.replace(".", "", 1).isdigit() for c in df.columns
    ):
        df = pd.read_csv(filepath, header=None)

    # Force numeric (important for ML pipelines)
    df = df.apply(pd.to_numeric, errors="raise")
    return df


def read_labeled_data_csv(filepath):
    """
    Reads labeled data from a CSV file.

    Accepts CSVs with or without a header. The format of the labeled data CSV is:
    - First n-1 columns: features
    - Last column: true labels

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    unlabeled_data : pd.DataFrame
        DataFrame of shape (n_samples, n_features) containing the features.
    true_labels : np.ndarray
        Array of shape (n_samples,) containing the true labels.

    """
    df = pd.read_csv(filepath)

    # If the first row is numeric, there was no real header
    if df.columns.to_list()[0].startswith("Unnamed") or all(
        c.replace(".", "", 1).isdigit() for c in df.columns
    ):
        df = pd.read_csv(filepath, header=None)

    # Force numeric (important for ML pipelines)
    df = df.apply(pd.to_numeric, errors="raise")

    unlabeled_data = df.iloc[:, :-1]
    true_labels = df.iloc[:, -1].values

    return unlabeled_data, true_labels


# ----------------------------
# Internal helper functions (private)
# ----------------------------
def _evaluate_ind(
    unlabeled_data, cluster_number, variables, clustering_method, evaluation_metric
):
    """
    Evaluate an individual solution in the GAUFS algorithm.

    Evaluates a specific combination of cluster number and feature selection
    by performing clustering and computing the evaluation metric.

    Parameters
    ----------
    unlabeled_data : pd.DataFrame
        The unlabeled data to cluster.
    cluster_number : int
        The number of clusters to form.
    variables : list
        A binary list indicating selected variables (1 for selected, 0 for not).
    clustering_method : ClusteringExperiment
        An instance of a clustering method with a ``run()`` method.
    evaluation_metric : EvaluationMetric
        An instance of an evaluation metric with a ``compute()`` method.

    Returns
    -------
    float
        The evaluation score of the clustering result.
    """
    try:
        # if no variables are selected, return a very low fitness
        if np.all(np.array(variables) == 0):
            return -10000000000

        filtered_vars = [
            var for var, i in zip(unlabeled_data.columns, variables) if i == 1
        ]
        filtered_data = unlabeled_data[filtered_vars]

        # Create a copy of the provided clustering experiment
        # deepcopying should not be necessary because clustering_method.unlabeled_data
        # should be None if running the GA, but to be safe we do it in case this method is used elsewhere
        experiment = copy.deepcopy(clustering_method)
        # copy the filtered unlabeled data
        experiment.set_unlabeled_data(filtered_data)
        # Set the number of clusters
        experiment.n_clusters = cluster_number
        # Run the clustering algorithm
        experiment.run()
        # get the results
        assigned_clusters = experiment.assigned_clusters

        # Evaluate the clustering result with the provided metric and the filtered data
        ev = evaluation_metric.compute(
            assigned_clusters=assigned_clusters, unlabeled_data=filtered_data
        )

        return ev

    except Exception as e:
        print(
            f"Error evaluating the individual with {cluster_number} clusters and the selection {variables}; Exception: {type(e).__name__} - {e}"
        )


def _compute_variable_significance(
    num_variables, hof_counter, max_number_selections_for_ponderation
):
    """
    Compute variable significance from Hall of Fame individuals.

    Calculates variable significance values between 0 and 1 based on the
    evaluation scores and selection frequency of individuals in the Hall of Fame.

    Parameters
    ----------
    num_variables : int
        Number of variables in the dataset.
    hof_counter : dict
        Dictionary mapping variable selections to their statistics.
        Keys are binary tuples representing variable selections.
        Values are tuples of (score, selection_count) where:

        - score : Best fitness achieved for a chromosome with that variable selection
        - selection_count : Number of times a chromosome with that selection entered the Hall of Fame

    max_number_selections_for_ponderation : int
        Maximum number of top individuals to consider for weight computation.

    Returns
    -------
    np.ndarray
        Variable significance values between 0 and 1 for each variable.
    """
    selections = sorted(hof_counter.items(), key=lambda item: item[1][0], reverse=True)[
        :max_number_selections_for_ponderation
    ]
    scores = []
    total = 0
    for it in selections:
        # the score of the selection if the maximun fitness achieved by that selection multiplied by the number of times it was selected in the HoF
        score = it[1][0] * it[1][1]
        scores.append(score)
        total += score

    # normalize scores so that they sum to 1
    scores_normalized = [x / total for x in scores]
    res = [0] * num_variables
    for i in range(num_variables):
        for j, s in enumerate(scores_normalized):
            res[i] += s * selections[j][0][i]
    return res


def _clear_directory(directory_path):
    """
    Clears all contents of the specified directory.
    Works on Windows and Linux.
    """
    if os.path.exists(directory_path):
        try:
            shutil.rmtree(directory_path)
        except PermissionError:
            raise PermissionError(
                f"Could not delete {directory_path}. Check file permissions or open handles."
            )

    os.makedirs(directory_path, exist_ok=True)


# Helper function to plot with discontinuities
def _plot_discontinuous(ax, x_vals, y_vals, **kwargs):
    """Plot lines only between consecutive x values"""
    i = 0
    while i < len(x_vals):
        # Find the end of the consecutive sequence
        j = i
        while j < len(x_vals) - 1 and x_vals[j + 1] == x_vals[j] + 1:
            j += 1

        # Plot this consecutive segment
        ax.plot(x_vals[i : j + 1], y_vals[i : j + 1], **kwargs)

        # Move to next segment
        i = j + 1


def _convert_to_serializable(obj):
    """
    Function to convert numpy arrays to lists
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def _get_variables_over_threshold(variables_weights, threshold):
    """
    Given a list of variable weights and a threshold, returns a binary list indicating which variables have weights greater than or equal to the threshold.
    """
    return [1 if w >= threshold else 0 for w in variables_weights]


def _get_dictionary_num_clusters_fitness(
    unlabeled_data,
    variable_selection,
    cluster_number_search_band,
    clustering_method,
    evaluation_metric,
):
    """
    Computes a dictionary mapping the number of clusters whithin the cluster_number_search_band (min_inclusive, max_exclusive) to their corresponding fitness scores given the fixed variable selection binary_variable_selection.
    """
    dict_clusters_fitness = {}
    for k in range(cluster_number_search_band[0], cluster_number_search_band[1]):
        dict_clusters_fitness[k] = _evaluate_ind(
            unlabeled_data=unlabeled_data,
            cluster_number=k,
            variables=variable_selection,
            clustering_method=clustering_method,
            evaluation_metric=evaluation_metric,
        )
    return dict_clusters_fitness


# The overhead of not computing the maximum in _get_dictionary_num_clusters_fitness is negligible compared to the clustering computations
def _get_num_clusters_with_best_fitness(dict_clusters_fit):
    """
    Given a dictionary mapping number of clusters to fitness scores, returns the number of clusters with the best fitness.
    In case of tie, returns the highest number of clusters among those with the best fitness.
    """
    num_clusters_for_maximum_fitness = None
    max_fitness = None
    for key in dict_clusters_fit.keys():
        # In case of tie we select the highest number of clusters
        if (
            max_fitness == None
            or max_fitness < dict_clusters_fit[key]
            or (
                max_fitness == dict_clusters_fit[key]
                and num_clusters_for_maximum_fitness < key
            )
        ):
            max_fitness = dict_clusters_fit[key]
            num_clusters_for_maximum_fitness = key
    return num_clusters_for_maximum_fitness, max_fitness


def _min_max_normalize_dictionary(dictionary):
    """
    Returns a new dictionary with the values min-max normalized to the range [0, 1].
    """
    values = list(dictionary.values())
    max_value = max(values)
    min_value = min(values)
    # In case all values are the same, return a dictionary with all values set to 0.0
    if max_value == min_value:
        return {k: 0.0 for k in dictionary.keys()}

    return {k: (v - min_value) / (max_value - min_value) for k, v in dictionary.items()}
