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
GAUFS: Main class for Genetic Algorithm for Unsupervised Feature Selection for Clustering.

This module provides the main interface to run GAUFS, combining feature
selection via genetic algorithms with clustering experiments and evaluation
metrics.

It uses:
- Clustering experiments (HierarchicalExperiment, KMeansExperiment, etc.)
- Evaluation metrics (internal and external)
- Genetic search for feature optimization
- Utility functions for data handling and visualization
"""

import os
import random
import warnings
import json

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from .clustering_experiments.hierarchical import HierarchicalExperiment
from .evaluation_metrics.internal import SilhouetteScore
from .genetic_search import GeneticSearch
from .utils import (
    read_unlabeled_data_csv,
    _evaluate_ind,
    _compute_variable_significance,
    _clear_directory,
    _plot_discontinuous,
    _convert_to_serializable,
    _get_variables_over_threshold,
    _get_dictionary_num_clusters_fitness,
    _get_num_clusters_with_best_fitness,
    _min_max_normalize_dictionary,
)


class Gaufs:
    """
    Genetic Algorithm for Unsupervised Feature Selection for Clustering.

    GAUFS combines genetic algorithms with clustering experiments to perform unsupervised
    feature selection. It identifies the most relevant features for clustering by evolving
    feature subsets and evaluating their clustering quality across different numbers of clusters.

    The algorithm operates in two main phases:

    1. Genetic Search Phase (run_genetic_searches): Runs multiple independent genetic algorithm
    executions to discover high-quality feature subsets and compute variable significance scores.

    2. Variable Weight Analysis Phase (analyze_variable_weights): Analyzes results from all
    genetic searches using importance metrics with exponential decay to automatically select
    the optimal feature subset and number of clusters.

    The complete workflow can be executed with run(), which chains both phases and generates
    comprehensive output files including plots, CSV files, and JSON dictionaries.

    Examples
    --------
    Basic usage with default parameters::

        import pandas as pd
        from gaufs import Gaufs
        data = pd.read_csv('unlabeled_data.csv')
        gaufs = Gaufs(unlabeled_data=data, seed=42)
        optimal_solution, fitness = gaufs.run()
        print(f"Selected {sum(optimal_solution[0])} features")
        print(f"Optimal clusters: {optimal_solution[1]}")

    Custom configuration::

        gaufs = Gaufs(
            unlabeled_data=data,
            num_genetic_executions=5,
            ngen=200,
            npop=2000,
            cluster_number_search_band=(3, 15),
            output_directory='./my_results/'
        )
        optimal_solution, fitness = gaufs.run()

    Step-by-step execution::

        gaufs = Gaufs(unlabeled_data=data)
        variable_significance = gaufs.run_genetic_searches()
        optimal_solution, fitness = gaufs.analyze_variable_weights()
        gaufs.plot_dictionaries()

    See Also
    --------
    run : Execute the complete GAUFS workflow
    run_genetic_searches : Run only the genetic search phase
    analyze_variable_weights : Run only the analysis phase
    plot_dictionaries : Generate analysis plots
    get_plot_comparing_solution_with_another_metric : Compare results with external metrics

    Notes
    -----
    Configuration attributes (seed, ngen, npop, etc.) are public and can be modified
    directly after initialization.

    Data and results attributes are private and accessed via read-only properties
    (unlabeled_data, optimal_solution, etc.).

    Analysis dictionaries remain None or empty until the algorithm is executed.

    Parameters ngen, npop, hof_size, and max_number_selections_for_ponderation are
    automatically adjusted based on the number of features if not explicitly provided.

    """

    def __init__(
        self,
        seed=None,
        unlabeled_data=None,
        num_genetic_executions=1,
        ngen=150,
        npop=1500,
        cxpb=0.8,
        cxpb_rest_of_genes=0.5,
        mutpb=0.1,
        convergence_generations=50,
        hof_size=None,
        hof_alpha_beta=(0.1, 0.2),
        clustering_method=HierarchicalExperiment(linkage="ward"),
        evaluation_metric=SilhouetteScore(),
        cluster_number_search_band=(2, 26),
        fitness_weight_over_threshold=0.5,
        exponential_decay_factor=1.0,
        max_number_selections_for_ponderation=None,
        verbose=True,
        generate_genetics_log_files=True,
        graph_evolution=True,
        generate_files_with_results=True,
        output_directory=None,
    ):
        """
        Initialize the GAUFS (Genetic Algorithm for Unsupervised Feature Selection for Clustering) instance.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility. Default: None for a random integer between 0 and 10000.
        unlabeled_data : pd.DataFrame or None, optional
            The input dataset without labels. If None, creates empty DataFrame.
        num_genetic_executions : int, optional
            Number of times to run the genetic algorithm. Default: 1. Range: >= 1.
        ngen : int, optional
            Number of generations for the genetic algorithm.
            Default: 150 (auto: 150 if num_vars <= 100, else 300). Range: >= 1.
        npop : int, optional
            Population size for the genetic algorithm.
            Default: 1500 (auto: 1500 if num_vars <= 100, else 7000). Range: >= 1.
        cxpb : float, optional
            Crossover probability for genetic operations. Default: 0.8. Range: [0.0, 1.0].
        cxpb_rest_of_genes : float, optional
            Crossover probability for the rest of generations after initial ones.
            Default: 0.5. Range: [0.0, 1.0].
        mutpb : float, optional
            Mutation probability for genetic operations. Default: 0.1. Range: [0.0, 1.0].
        convergence_generations : int, optional
            Number of generations without improvement before stopping.
            Default: 50. Range: >= 1.
        hof_size : int or None, optional
            Hall of Fame size as an absolute number of best solutions to retain.
            If provided, overrides hof_alpha_beta automatic calculation.
            Default: None (uses hof_alpha_beta formula). Range: >= 1 or None.
        hof_alpha_beta : tuple, optional
            (alpha, beta) parameters for automatic Hall of Fame size calculation.
            Formula: hof_size = npop * (beta - (beta - alpha) * log(2) / log(num_vars + 1))

            - Alpha (first value): Minimum fraction of population for HOF (when num_vars is small)
            - Beta (second value): Maximum fraction of population for HOF (when num_vars is large)

            This adaptive formula increases HOF size as feature count increases.
            Default: (0.1, 0.2). Range: [0.0, 1.0] for each, beta >= alpha.
            Only used if hof_size is None.
        clustering_method : ClusteringExperiment, optional
            Clustering algorithm instance. Default: HierarchicalExperiment(linkage='ward').
            Must implement clustering interface ``ClusteringExperiment``.
        evaluation_metric : EvaluationMetric, optional
            Metric for evaluating clustering quality. Default: SilhouetteScore().
            Must implement evaluation interface ``EvaluationMetric``.
        cluster_number_search_band : tuple, optional
            Range of cluster numbers to explore as (min_inclusive, max_exclusive).
            Default: (2, 26). Range: (>= 2, <= num_samples).
        fitness_weight_over_threshold : float, optional
            Weight for fitness vs. threshold in variable weight analysis.
            Default: 0.5 i.e. both values are averaged. Range: [0.0, 1.0].
        exponential_decay_factor : float, optional
            Exponential decay factor for the automatic solution selector.
            Formula: δ_i / (1 + (N / exp(exponential_decay_factor * i))).
            If 0.0 there is no decay. Default: 1.0. Range: >= 0.0.
        max_number_selections_for_ponderation : int or None, optional
            Maximum selections from Hall of Fame for weight computation.
            Default: 2 * num_vars. Range: >= 1 or None.
        verbose : bool, optional
            Whether to print logs during execution. Default: True.
        generate_genetics_log_files : bool, optional
            Whether to generate a log file with Genetic Algorithm execution details.
            Default: True.
        graph_evolution : bool, optional
            Whether to generate a graph of the best and average fitness through
            the Genetic Algorithm's evolution. Default: True.
        generate_files_with_results : bool, optional
            Whether to generate files with results and plots. Default: True.
        output_directory : str or None, optional
            Path to store generated files including the plots.
            If None, "./out/" is used.

        Attributes
        ----------
        seed : int
            Random seed for reproducibility.
        num_genetic_executions : int
            Number of independent GA executions.
        ngen : int
            Maximum number of generations.
        npop : int
            Population size.
        cxpb : float
            Crossover probability.
        cxpb_rest_of_genes : float
            Crossover probability for rest of genes.
        mutpb : float
            Mutation probability.
        convergence_generations : int
            Generations without improvement before stopping.
        hof_size : int
            Hall of Fame size.
        hof_alpha_beta : tuple
            Parameters for automatic HOF size calculation.
        clustering_method : ClusteringExperiment
            Clustering algorithm instance.
        evaluation_metric : EvaluationMetric
            Evaluation metric instance.
        cluster_number_search_band : tuple
            Range of cluster numbers to test.
        fitness_weight_over_threshold : float
            Weight for fitness vs. threshold.
        exponential_decay_factor : float
            Decay factor for solution selection.
        max_number_selections_for_ponderation : int
            Max solutions for weight calculation.
        verbose : bool
            Whether to print execution logs.
        generate_genetics_log_files : bool
            Whether to generate GA log files.
        graph_evolution : bool
            Whether to generate evolution graphs.
        generate_files_with_results : bool
            Whether to generate output files.
        output_directory : str
            Directory for generated files.

        Notes
        -----
        **Configuration Attributes (Public - User Modifiable):**
        All parameters listed above are public attributes that can be modified
        directly after initialization.

        **Data Properties (Read-Only via @property):**

        * ``unlabeled_data``: Input dataset without labels (returns copy)
        * ``num_vars``: Number of features/variables in the dataset

        **Results Properties (Read-Only via @property, populated after running):**

        * ``variable_significances``: Variable significances for each GA execution
        * ``variable_significance``: Final averaged variable significance
        * ``best_chromosomes``: Best chromosome from each GA execution
        * ``ga_instances``: Each GA instance that has been run
        * ``optimal_solution``: Optimal variable selection and number of clusters, format: (variable_selection, num_clusters)
        * ``optimal_fitness``: Fitness value of the optimal solution

        **Analysis Dictionary Properties (Read-Only via @property):**

        * ``dict_selection_num_clusters``: Maps variable_selection -> optimal_num_clusters
        * ``dict_selection_fitness``: Maps variable_selection -> best_fitness
        * ``dict_num_var_selection_with_that_num_variables``: Maps num_variables -> variable_selection
        * ``dict_num_var_selected_num_clusters``: Maps num_variables -> optimal_num_clusters
        * ``dict_num_var_selected_fitness``: Maps num_variables -> best_fitness
        * ``dict_num_var_threshold``: Maps num_variables -> significance_threshold
        * ``dict_num_var_selected_fitness_min_max_normalized``: MinMax normalized fitness
        * ``dict_num_var_threshold_min_max_normalized``: MinMax normalized threshold
        * ``dict_num_var_selected_importance``: Weighted average importance
        * ``dictionary_deltas_importance_diferences``: Importance differences δ
        * ``dictionary_deltas_importance_diferences_with_exponential_decay``: δ with decay
        * ``dict_num_var_all_clusters_fitness``: All fitness values for 3D plotting

        Most analysis dictionaries are None or empty until ``run_genetic_searches()`` and
        ``analyze_variable_weights()`` are called, or ``run()`` which executes both in sequence.
        """
        # Random seed for reproducibility (randomly generated integer between 0 and 10000)
        self.seed = seed if seed is not None else random.randint(0, 10000)
        random.seed(self.seed)

        # Number of independent genetic algorithm executions to perform
        # Higher values provide more robust results but increase computation time
        self.num_genetic_executions = num_genetic_executions

        # Maximum number of generations for the genetic algorithm
        # Higher values allow more evolution but increase computation time
        self.ngen = ngen

        # Population size for the genetic algorithm
        # Larger populations explore solution space better but require more computation
        self.npop = npop

        # Crossover probability - likelihood of combining two parent solutions
        # Typical range: 0.6-0.9. Higher values promote exploration
        self.cxpb = cxpb

        # Crossover probability for the rest of genes after the cluster number gene
        self.cxpb_rest_of_genes = cxpb_rest_of_genes

        # Mutation probability - likelihood of randomly altering a solution
        # Typical range: 0.01-0.2. Higher values increase diversity
        self.mutpb = mutpb

        # Number of generations without improvement before early stopping
        # Helps prevent unnecessary computation when convergence is reached
        self.convergence_generations = convergence_generations

        # Alpha and Beta parameters for automatic Hall of Fame size calculation
        # hof_size = npop*(alpha-(alpha-beta)*log(2)/log(n_variables+1))
        self.hof_alpha_beta = hof_alpha_beta

        # Clustering algorithm instance used to evaluate feature subsets
        # Default: HierarchicalExperiment (hierarchical clustering)
        self.clustering_method = clustering_method

        # Evaluation metric for assessing clustering quality
        # Default: SilhouetteScore (measures cluster cohesion and separation)
        self.evaluation_metric = evaluation_metric = evaluation_metric

        # Range of cluster numbers to test: [min, max) - max is exclusive
        # Default: tests 2 to 25 clusters. Should be based on domain knowledge
        self.cluster_number_search_band = cluster_number_search_band

        # Weighting factor for fitness vs. threshold in variable weight analysis.
        # weight of fitness = fitness_weight_over_threshold
        # weight of threshold = 1 - fitness_weight_over_threshold
        self.fitness_weight_over_threshold = fitness_weight_over_threshold

        # Exponential decay factor for automatic solution selection
        self.exponential_decay_factor = exponential_decay_factor

        # Maximum number of top solutions from Hall of Fame used for feature weight calculation
        # Default: None (will be set to 2 * number of features)
        self.max_number_selections_for_ponderation = (
            max_number_selections_for_ponderation
        )

        # Wether generate a graph showing the evolution of the best fitness score in the Genetic Algorithm
        self.graph_evolution = graph_evolution

        # Verbosity flag
        self.verbose = verbose

        # Flag to generate a log file with Genetic Algorithm execution details
        self.generate_genetics_log_files = generate_genetics_log_files

        # Flag to generate files with plots and results
        self.generate_files_with_results = generate_files_with_results

        # Directory to store generated plots, logs and other files
        self.output_directory = (
            output_directory
            if output_directory is not None
            else os.path.join(".", "out")
        )
        if not self.output_directory.endswith(os.sep):
            self.output_directory += os.sep

        # Input dataset and derived properties
        if unlabeled_data is None:
            # Empty DataFrame when no data provided
            self._unlabeled_data = pd.DataFrame()
            # Number of features/variables in the dataset
            self._num_vars = None
        else:
            self.set_unlabeled_data(unlabeled_data, recompute_default_parameters=True)

        # If hof_size is provided it has preference over hof_alpha_beta
        if hof_size is not None:
            # Hall of Fame size as an absolute number
            # Stores the best solutions found during evolution
            self.hof_size = hof_size

        # Variable significances for each Genetic execuition
        self._variable_significances = []

        # Best chromosome for each Genetic execution
        self._best_chromosomes = []

        #  Each GA instance that has been run in case more specific data is wanted
        self._ga_instances = []

        # Final averaged variable significance.
        self._variable_significance = None

        # the solution must be chosen from the keys and values of this dictionary
        # dictionary={variable_selection:cluster_number_with_best_fitness}
        self._dict_selection_num_clusters = {}
        # dictionary={variable_selection:best_fitness}
        self._dict_selection_fitness = {}

        # Other dictionaries for analysis and plotting
        # dictionary={num_variables_selected:[variable_selection_with_that_num_variables]}
        self._dict_num_var_selection_with_that_num_variables = {}
        # \Psi_{num\_clusters}={num_variables_selected:cluster_number_with_best_fitness}
        self._dict_num_var_selected_num_clusters = {}
        # \tilde{\Psi}_{fitness}={num_variables_selected:best_fitness}
        self._dict_num_var_selected_fitness = {}
        # \tilde{\Psi}_{weight\_threshold}={num_variables_selected:threshold_for_selection}
        self._dict_num_var_threshold = {}
        # MinMax(\tilde{\Psi}_{fitness})
        self._dict_num_var_selected_fitness_min_max_normalized = None
        # MinMax(\tilde{\Psi}_{weight\_threshold})
        self._dict_num_var_threshold_min_max_normalized = None
        # \Phi_{average}={num_variables_selected:average_between_fitness_and_threshold}
        # the average is weighted by self.fitness_weight_over_threshold
        # \Phi_{average}= self.fitness_weight_over_threshold*MinMax(\tilde{\Psi}_{fitness}) + (1-self.fitness_weight_over_threshold)*MinMax(\tilde{\Psi}_{weight_threshold})
        self._dict_num_var_selected_importance = None
        # \delta={num_variables_selected:importance_difference_with_next_selection}
        self._dictionary_deltas_importance_diferences = {}
        # \tilde{\delta} = {num_variables_selected:importance_difference_with_next_selection_with_exponential_decay}
        # \tilde{\delta} = \frac{\delta_i}{1 + \left( \frac{N}{e^{self.exponential_decay_factor*i}} \right)}
        self._dictionary_deltas_importance_diferences_with_exponential_decay = None
        # this dictionary is only for the 3D plot {num_selected_variables:{num_clusters:fitness}}
        self._dict_num_var_all_clusters_fitness = {}

        # Optimal variable selection and number of clusters after analysis (variable selection as binary list, num_clusters)
        self._optimal_variable_selection_and_num_of_clusters = None
        # Fitness value associated to the optimal variable selection and number of clusters
        self._optimal_variable_selection_and_num_of_clusters = None

    @property
    def unlabeled_data(self):
        """
        Get the unlabeled data (read-only).

        Returns:
            pd.DataFrame: Copy of the input dataset without labels.
        """
        return self._unlabeled_data.copy()

    @property
    def num_vars(self):
        """
        Get number of variables in the dataset (read-only).

        Returns:
            int or None: Number of features/variables.
        """
        return self._num_vars

    @property
    def variable_significances(self):
        """
        Get variable significances for each GA execution (read-only).

        Returns:
            list: List of variable significance arrays, one per GA execution.
        """
        return self._variable_significances.copy()

    @property
    def variable_significance(self):
        """
        Get final averaged variable significance (read-only).

        Returns:
            np.ndarray or None: Averaged variable significance across all executions.
                None if run_genetic_searches() hasn't been called yet.
        """
        return self._variable_significance

    @property
    def best_chromosomes(self):
        """
        Get best chromosomes from each GA execution (read-only).

        Returns:
            list: List of best chromosomes, one per GA execution.
        """
        return self._best_chromosomes.copy()

    @property
    def ga_instances(self):
        """
        Get all GA instances that have been run (read-only).

        Returns:
            list: List of GeneticSearch instances for detailed analysis.
        """
        return self._ga_instances.copy()

    @property
    def optimal_solution(self):
        """
        Get optimal variable selection and number of clusters (read-only).

        Returns:
            tuple: (variable_selection, num_clusters) where variable_selection
                is a binary list (1=selected, 0=not selected).

        Raises:
            RuntimeError: If analysis hasn't been run yet.
        """
        if self._optimal_variable_selection_and_num_of_clusters is None:
            raise RuntimeError(
                "No optimal solution available. Run the algorithm first using "
                "run() or run analyze_variable_weights() after run_genetic_searches()."
            )
        return self._optimal_variable_selection_and_num_of_clusters

    @property
    def optimal_fitness(self):
        """
        Get fitness value of the optimal solution (read-only).

        Returns:
            float: Fitness value associated with the optimal solution.

        Raises:
            RuntimeError: If analysis hasn't been run yet.
        """
        if self._fitness_of_optimal_variable_selection_and_num_of_clusters is None:
            raise RuntimeError(
                "No optimal fitness available. Run the algorithm first using "
                "run() or run analyze_variable_weights() after run_genetic_searches()."
            )
        return self._fitness_of_optimal_variable_selection_and_num_of_clusters

    @property
    def dict_selection_num_clusters(self):
        """
        Maps variable selection to optimal number of clusters (read-only).

        Returns:
            dict: {variable_selection_tuple: optimal_num_clusters}
        """
        return self._dict_selection_num_clusters.copy()

    @property
    def dict_selection_fitness(self):
        """
        Maps variable selection to best fitness (read-only).

        Returns:
            dict: {variable_selection_tuple: best_fitness}
        """
        return self._dict_selection_fitness.copy()

    @property
    def dict_num_var_selection_with_that_num_variables(self):
        """
        Maps number of variables to variable selection (read-only).

        Returns:
            dict: {num_variables: variable_selection_list}
        """
        return self._dict_num_var_selection_with_that_num_variables.copy()

    @property
    def dict_num_var_selected_num_clusters(self):
        """
        Ψ_{num_clusters}: Maps number of variables to optimal clusters (read-only).

        Returns:
            dict: {num_variables: optimal_num_clusters}
        """
        return self._dict_num_var_selected_num_clusters.copy()

    @property
    def dict_num_var_selected_fitness(self):
        """
        Ψ̃_{fitness}: Maps number of variables to best fitness (read-only).

        Returns:
            dict: {num_variables: best_fitness}
        """
        return self._dict_num_var_selected_fitness.copy()

    @property
    def dict_num_var_threshold(self):
        """
        Ψ̃_{weight_threshold}: Maps number of variables to threshold (read-only).

        Returns:
            dict: {num_variables: significance_threshold}
        """
        return self._dict_num_var_threshold.copy()

    @property
    def dict_num_var_selected_fitness_min_max_normalized(self):
        """
        MinMax normalized fitness values (read-only).

        Returns:
            dict or None: MinMax(Ψ̃_{fitness})
        """
        if self._dict_num_var_selected_fitness_min_max_normalized is None:
            return None
        return self._dict_num_var_selected_fitness_min_max_normalized.copy()

    @property
    def dict_num_var_threshold_min_max_normalized(self):
        """
        MinMax normalized threshold values (read-only).

        Returns:
            dict or None: MinMax(Ψ̃_{weight_threshold})
        """
        if self._dict_num_var_threshold_min_max_normalized is None:
            return None
        return self._dict_num_var_threshold_min_max_normalized.copy()

    @property
    def dict_num_var_selected_importance(self):
        """
        Φ_{average}: Weighted average importance (read-only).

        Formula: fitness_weight * MinMax(fitness) + (1-fitness_weight) * MinMax(threshold)

        Returns:
            dict or None: {num_variables: importance_value}
        """
        if self._dict_num_var_selected_importance is None:
            return None
        return self._dict_num_var_selected_importance.copy()

    @property
    def dictionary_deltas_importance_diferences(self):
        """
        δ: Importance differences (read-only).

        Formula: δ_i = Φ_{average}(i) - Φ_{average}(i+1)

        Returns:
            dict: {num_variables: importance_difference}
        """
        return self._dictionary_deltas_importance_diferences.copy()

    @property
    def dictionary_deltas_importance_diferences_with_exponential_decay(self):
        """
        δ̃: Importance differences with exponential decay (read-only).

        Formula: δ̃_i = δ_i / (1 + (N / exp(exponential_decay_factor * i)))
        Used for optimal solution selection.

        Returns:
            dict or None: {num_variables: decayed_importance_difference}
        """
        if self._dictionary_deltas_importance_diferences_with_exponential_decay is None:
            return None
        return (
            self._dictionary_deltas_importance_diferences_with_exponential_decay.copy()
        )

    @property
    def dict_num_var_all_clusters_fitness(self):
        """
        All fitness values for 3D plotting (read-only).

        Returns:
            dict: {num_selected_variables: {num_clusters: fitness}}
        """
        return {k: v.copy() for k, v in self._dict_num_var_all_clusters_fitness.items()}

    def set_seed(self, seed):
        """
        Set the random seed for reproducibility.

        Parameters
        ----------
        seed : int
            The random seed to set.
        """
        self.seed = seed
        random.seed(self.seed)

    def set_unlabeled_data(self, _unlabeled_data, recompute_default_parameters=True):
        """
        Sets the unlabeled data to analyze with GAUFS.

        Parameters
        ----------
        _unlabeled_data : pd.DataFrame
            The input dataset without labels.
        recompute_default_parameters : bool, optional
            Whether to recompute default parameters based on the number of variables.
            Default is True.
        """
        self._unlabeled_data = _unlabeled_data.copy()
        self._num_vars = self._unlabeled_data.shape[1]

        # Set the genetic algorithm parameters based on the number of variables
        if recompute_default_parameters:
            # number of generations and population size
            if self._num_vars <= 100:
                self.ngen = 150
                self.npop = 1500
            else:
                self.ngen = 300
                self.npop = 7000
            # max number of selections for ponderation
            self.max_number_selections_for_ponderation = 2 * self._num_vars

            # hof_size = npop*(beta-(beta-alpha)*log(2)/log(n_variables+1))
            beta = self.hof_alpha_beta[1]
            alpha = self.hof_alpha_beta[0]
            tam = int(
                self.npop
                * (
                    beta
                    - (beta - alpha)
                    * np.log(2)
                    / np.log((self._unlabeled_data.shape[1] + 1))
                )
            )
            self.hof_size = tam if tam != 0 else 1

    def read_unlabeled_data_csv(self, filepath, recompute_default_parameters=True):
        """
        Reads unlabeled data from a CSV file.

        Accepts CSVs with or without header.
        Expected shape: (n_samples, n_features)

        Parameters
        ----------
        filepath : str
            Path to the CSV file containing the unlabeled data.
        recompute_default_parameters : bool, optional
            Whether to recompute default parameters based on the number of variables.
            Default is True.
        """

        df = read_unlabeled_data_csv(filepath)

        self.set_unlabeled_data(
            df, recompute_default_parameters=recompute_default_parameters
        )

    def run(self):
        """
        Run the complete GAUFS algorithm for unsupervised feature selection.

        Executes both genetic search and variable weight analysis phases,
        generating comprehensive output files including plots and results.

        Returns
        -------
        optimal_variable_selection_and_num_of_clusters : tuple
            (variable_selection, num_clusters) where variable_selection is a binary list.
        fitness_of_optimal_variable_selection_and_num_of_clusters : float
            Fitness value associated with the optimal solution.
        """
        if self._unlabeled_data.empty:
            raise ValueError(
                "Unlabeled data is not loaded. Please load the data before running the model."
            )

        # delete all the content of the output directory to avoid mixing files from different runs
        _clear_directory(self.output_directory)

        # get the variable significances by running the genetic searches
        self.run_genetic_searches()

        # analyze the variable weights to get the optimal variable selection and number of clusters
        self.analyze_variable_weights()

        if self.verbose:
            print(
                f"Optimal variable selection (1=selected, 0=not selected): {self._optimal_variable_selection_and_num_of_clusters[0]}\nwith the optimal number of clusters: {self._optimal_variable_selection_and_num_of_clusters[1]}"
            )
            print(
                "Fitness of optimal variable selection and number of clusters: ",
                self._fitness_of_optimal_variable_selection_and_num_of_clusters,
            )

        if self.generate_files_with_results:
            # Save optimal variable selection and number of clusters to a text file
            directory = os.path.join(self.output_directory, "results")
            os.makedirs(directory, exist_ok=True)
            output_path_txt = os.path.join(
                directory, "optimal_variable_selection_and_number_of_clusters.txt"
            )

            df_optimal_selection = pd.DataFrame(
                {
                    "Variable": self._unlabeled_data.columns,
                    "Selected": self._optimal_variable_selection_and_num_of_clusters[0],
                }
            )

            output_path_csv = os.path.join(directory, "optimal_variable_selection.csv")
            df_optimal_selection.to_csv(output_path_csv, index=False)

            # Also save summary to text file
            with open(output_path_txt, "w") as f:
                f.write(
                    f"Optimal number of clusters: {self._optimal_variable_selection_and_num_of_clusters[1]}\n"
                )
                f.write(
                    f"Fitness: {self._optimal_variable_selection_and_num_of_clusters}\n\n"
                )
                f.write("Selected Variables:\n")
                f.write(
                    df_optimal_selection[df_optimal_selection["Selected"] == 1][
                        "Variable"
                    ].to_string(index=False)
                )

            # Save plots of dictionaries
            self.plot_dictionaries()

            # 3D plot of number of variables, number of clusters and fitness
            self.plot_num_variables_and_clusters_3D()

            # Collect dictionaries
            data_dictionaries = {
                "dict_num_var_selection_with_that_num_variables": _convert_to_serializable(
                    self._dict_num_var_selection_with_that_num_variables
                ),
                "dict_num_var_selected_num_clusters": _convert_to_serializable(
                    self._dict_num_var_selected_num_clusters
                ),
                "dict_num_var_selected_fitness": _convert_to_serializable(
                    self._dict_num_var_selected_fitness
                ),
                "dict_num_var_threshold": _convert_to_serializable(
                    self._dict_num_var_threshold
                ),
                "dict_num_var_selected_importance": _convert_to_serializable(
                    self._dict_num_var_selected_importance
                ),
                "dictionary_deltas_importance_diferences_with_exponential_decay": _convert_to_serializable(
                    self._dictionary_deltas_importance_diferences_with_exponential_decay
                ),
            }

            # Save to JSON file
            out_path_dictionaries = os.path.join(
                directory, "dictionaries_variables_weight_analysis.json"
            )
            with open(out_path_dictionaries, "w") as f:
                json.dump(data_dictionaries, f, indent=4)

            if self.verbose:
                print(
                    f"Dictionaries from variable weight analysis saved to {out_path_dictionaries}"
                )
                print(
                    f"Optimal variable selection and number of clusters saved to {output_path_txt}"
                )

        return (
            self._optimal_variable_selection_and_num_of_clusters,
            self._optimal_variable_selection_and_num_of_clusters,
        )

    def run_genetic_searches(self):
        """ "
        Run multiple independent genetic algorithm executions.

        Computes variable significances from multiple GA runs.
        To run the full GAUFS algorithm use ``run()`` method instead.

        Returns
        -------
        np.ndarray
            Averaged variable significances across all genetic executions.
        """

        if self.num_genetic_executions < 1:
            raise ValueError("num_genetic_executions must be at least 1.")

        if self.hof_size >= self.npop:
            raise ValueError("hof_size must be less than population size (npop).")
        # Generate unique seeds for each genetic algorithm execution
        seeds_for_GA = random.sample(range(0, 10000), self.num_genetic_executions)

        for s in seeds_for_GA:
            ga_instance = GeneticSearch(
                seed=s,
                unlabeled_data=self._unlabeled_data,
                ngen=self.ngen,
                npop=self.npop,
                cxpb=self.cxpb,
                cxpb_rest_of_genes=self.cxpb_rest_of_genes,
                mutpb=self.mutpb,
                convergence_generations=self.convergence_generations,
                hof_size=self.hof_size,
                clustering_method=self.clustering_method,
                evaluation_metric=self.evaluation_metric,
                cluster_number_search_band=self.cluster_number_search_band,
                verbose=self.verbose,
                path_store_log=(
                    os.path.join(self.output_directory, f"GA_Seed_{s}")
                    if self.generate_genetics_log_files
                    else None
                ),
                path_store_plot=(
                    os.path.join(self.output_directory, f"GA_Seed_{s}")
                    if self.graph_evolution
                    else None
                ),
            )
            if self.verbose:
                print(f"Running Genetic Algorithm with seed {s}...")

            hof_counter, _ = ga_instance.run()
            # Compute variable significance from Hall of Fame
            variable_significance = _compute_variable_significance(
                num_variables=self._num_vars,
                hof_counter=hof_counter,
                max_number_selections_for_ponderation=self.max_number_selections_for_ponderation,
            )
            self._variable_significances.append(variable_significance)

            # Store Best chromosome and GA instance although it is not necessary for GAUFS operation
            self._best_chromosomes.append(ga_instance.hof[0])
            self._ga_instances.append(ga_instance)

        # Average variable significances across all genetic executions
        self._variable_significance = np.mean(self._variable_significances, axis=0)
        if self.verbose:
            print("Genetic Algorithm executions completed.")
            print(
                "The variable weights (significances) are: ",
                self._variable_significance,
            )

        if self.generate_files_with_results:
            # Save variable significances to a CSV file
            directory = os.path.join(self.output_directory, "results")
            os.makedirs(directory, exist_ok=True)
            df_var_significance = pd.DataFrame(
                {
                    "Variable": self._unlabeled_data.columns,
                    "Significance": self._variable_significance,
                }
            )
            output_path_csv = os.path.join(directory, "variable_significances.csv")
            df_var_significance.to_csv(output_path_csv, index=False)
            if self.verbose:
                print(f"Variable significances saved to {output_path_csv}")

        return self._variable_significance

    def analyze_variable_weights(self):
        """
        Analyze variable weights to determine optimal solution.

        Determines the optimal variable selection and number of clusters after
        running genetic searches with ``run_genetic_searches()``.

        To run the full GAUFS algorithm, use the ``run()`` method instead.

        Returns
        -------
        optimal_variable_selection_and_num_of_clusters : tuple
            (variable_selection, num_clusters) where variable_selection is a binary list.
        fitness_of_optimal_variable_selection_and_num_of_clusters : float
            Fitness value associated with the optimal solution.
        """

        if self._variable_significance is None:
            raise ValueError(
                "Variable significances not computed. Please run genetic searches with `run_genetic_searches` before analyzing variable weights."
            )

        thresholds = sorted(self._variable_significance, reverse=True)

        # list of the considered selections
        possible_variable_selections = []

        # Create the directory
        directory = os.path.join(self.output_directory, "results")
        os.makedirs(directory, exist_ok=True)

        # For each threshold get the selection and analyze it
        for threshold in thresholds:
            selection = _get_variables_over_threshold(
                self._variable_significance, threshold
            )
            number_of_selected_variables = sum(selection)

            possible_variable_selections.append(selection)
            dict_clusters_fit = _get_dictionary_num_clusters_fitness(
                unlabeled_data=self._unlabeled_data,
                variable_selection=selection,
                clustering_method=self.clustering_method,
                evaluation_metric=self.evaluation_metric,
                cluster_number_search_band=self.cluster_number_search_band,
            )

            num_clusters_for_maximum_fitness, max_fitness = (
                _get_num_clusters_with_best_fitness(dict_clusters_fit)
            )

            self._dict_selection_num_clusters[tuple(selection)] = (
                num_clusters_for_maximum_fitness
            )
            self._dict_num_var_selected_num_clusters[number_of_selected_variables] = (
                num_clusters_for_maximum_fitness
            )
            self._dict_num_var_threshold[number_of_selected_variables] = threshold

            self._dict_selection_fitness[tuple(selection)] = max_fitness
            self._dict_num_var_selected_fitness[number_of_selected_variables] = (
                max_fitness
            )

            self._dict_num_var_selection_with_that_num_variables[
                number_of_selected_variables
            ] = selection

            # This dictionary is only ofr the 3D plot
            self._dict_num_var_all_clusters_fitness[number_of_selected_variables] = (
                dict_clusters_fit
            )

        # there is only one selecction possible: selecting all variables
        if len(thresholds) == 1:
            warnings.warn(
                f"The only selection possible is selecting all variables "
                f"and num_clusters={num_clusters_for_maximum_fitness} "
                f"with fitness={max_fitness}",
                UserWarning,
            )
            # the only possible solution is the one in self._dict_selection_num_clusters
            self._optimal_variable_selection_and_num_of_clusters = (
                possible_variable_selections[0],
                list(self._dict_selection_num_clusters.values())[0],
            )
            self._fitness_of_optimal_variable_selection_and_num_of_clusters = list(
                self._dict_selection_fitness.values()
            )[0]
            return (
                self._optimal_variable_selection_and_num_of_clusters,
                self._fitness_of_optimal_variable_selection_and_num_of_clusters,
            )

        # This only affects the graphs. We make up solutions for the missing number of variables so that the graphs are continuous but these solutions
        # do not exist in reality as they do not correspond to any selection. They will not be selected as optimal solutions in any case.
        max_num_var_with_value = max(self._dict_num_var_selected_fitness.keys())
        # from the last one to the first one
        for i in range(max_num_var_with_value - 1, 0, -1):
            if i not in self._dict_num_var_selected_fitness.keys():
                # assign it the value of the one which is next to it
                self._dict_num_var_selected_fitness[i] = (
                    self._dict_num_var_selected_fitness[i + 1]
                )
                self._dict_num_var_threshold[i] = self._dict_num_var_threshold[i + 1]

        # MinMax normalization of fitness and thresholds
        self._dict_num_var_selected_fitness_min_max_normalized = (
            _min_max_normalize_dictionary(self._dict_num_var_selected_fitness)
        )
        self._dict_num_var_threshold_min_max_normalized = _min_max_normalize_dictionary(
            self._dict_num_var_threshold
        )

        # Get the considered number of variables sorted (including the fake ones created for continuity)
        keys_sorted = sorted(self._dict_num_var_selected_fitness.keys())

        # Compute the average importance
        self._dict_num_var_selected_importance = {
            num_var: self.fitness_weight_over_threshold
            * self._dict_num_var_selected_fitness_min_max_normalized[num_var]
            + (1 - self.fitness_weight_over_threshold)
            * self._dict_num_var_threshold_min_max_normalized[num_var]
            for num_var in keys_sorted
        }

        # Compute the differences (deltas)
        for i in range(len(keys_sorted) - 1):
            key_current = keys_sorted[i]
            key_next = keys_sorted[i + 1]
            self._dictionary_deltas_importance_diferences[key_current] = max(
                0,
                self._dict_num_var_selected_importance[key_current]
                - self._dict_num_var_selected_importance[key_next],
            )
        # For the last one, the next value is considered 0
        # The max is not necessary here because the importances are possitive but for consistency with the others
        self._dictionary_deltas_importance_diferences[keys_sorted[-1]] = max(
            0, self._dict_num_var_selected_importance[keys_sorted[-1]]
        )

        self._dictionary_deltas_importance_diferences_with_exponential_decay = (
            self._dictionary_deltas_importance_diferences.copy()
        )

        # Add the exponential decay to the deltas if exponential_decay_factor>0
        if self.exponential_decay_factor > 0:
            # It divides the ponderations by a factor of 1+((N-1) / (math.exp(exponential_decay_factor * num_var))) where N is the toltal number of variables and num_var is the number of variables selected.
            # The exponential_decay_factor * num_var < 700 avoids overflow
            self._dictionary_deltas_importance_diferences_with_exponential_decay = {
                num_var: (
                    delta
                    / (
                        1
                        + (
                            (len(self._dictionary_deltas_importance_diferences) - 1)
                            / (np.exp(self.exponential_decay_factor * num_var))
                        )
                    )
                    if self.exponential_decay_factor * num_var < 700
                    else delta
                )
                for (
                    num_var,
                    delta,
                ) in self._dictionary_deltas_importance_diferences.items()
            }

        # Use the deltas with exponential decay for selecting the optimal solution
        # Notice that the fake solutions' deltas are 0 and as long as there is a possitive delta they will not be selected
        optimal_num_variables = max(
            self._dictionary_deltas_importance_diferences_with_exponential_decay,
            key=self._dictionary_deltas_importance_diferences_with_exponential_decay.get,
        )

        # if a fake solution is selected this is because all the deltas are 0 =>
        # => all the variables have the same importance => all variables are significant
        if (
            optimal_num_variables
            not in self._dict_num_var_selection_with_that_num_variables.keys()
        ):
            self._optimal_variable_selection_and_num_of_clusters = [
                1
            ] * self._num_vars, self._dict_num_var_selected_num_clusters[self._num_vars]
            self._fitness_of_optimal_variable_selection_and_num_of_clusters = (
                self._dict_num_var_selected_fitness[self._num_vars]
            )
            return (
                self._optimal_variable_selection_and_num_of_clusters,
                self._fitness_of_optimal_variable_selection_and_num_of_clusters,
            )

        self._optimal_variable_selection_and_num_of_clusters = (
            self._dict_num_var_selection_with_that_num_variables[optimal_num_variables],
            self._dict_num_var_selected_num_clusters[optimal_num_variables],
        )
        self._fitness_of_optimal_variable_selection_and_num_of_clusters = (
            self._dict_num_var_selected_fitness[optimal_num_variables]
        )
        return (
            self._optimal_variable_selection_and_num_of_clusters,
            self._fitness_of_optimal_variable_selection_and_num_of_clusters,
        )

    def plot_dictionaries(self):
        """
        Generate comparison plots between GAUFS fitness and external metric.

        Creates two side-by-side plots:

        - Left: Variables vs Used Fitness in GAUFS execution
        - Right: Variables vs Provided Metric for each selection

        This allows comparison between the fitness used in GAUFS and an
        external metric of interest.

        Parameters
        ----------
        new_metric : EvaluationMetric
            Must implement evaluation interface.
        true_number_of_labels : int or None, optional
            True number of labels from the data. If specified, plots include
            a baseline comparing the score obtained with the true number of
            labels as number of clusters. Default is None.
        output_path : str or None, optional
            Path to save the generated plot. If None, saves to
            ``self.output_directory/comparison_fitness_vs_external_metric.png``.

        Returns
        -------
        str
            Path where the plot was saved.
        """
        if self._dict_num_var_selection_with_that_num_variables == {}:
            raise ValueError(
                "Variable weight analysis not performed. Please run `analyze_variable_weights` before plotting dictionaries."
            )

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Analysis by Number of Variables", fontsize=16)

        # Plot 1: Number of clusters
        ax1 = axes[0, 0]
        x1, y1 = zip(*sorted(self._dict_num_var_selected_num_clusters.items()))

        _plot_discontinuous(ax1, x1, y1, marker="o", color="black")
        ax1.set_xlabel("Number of Selected Variables")
        ax1.set_ylabel("Number of Clusters")
        ax1.set_title("Number of Clusters for Each Selection")
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # Plot 2: Fitness
        ax2 = axes[0, 1]
        x2, y2 = zip(*sorted(self._dict_num_var_selected_fitness.items()))
        ax2.plot(x2, y2, marker="o", color="tab:blue")
        ax2.set_xlabel("Number of Selected Variables")
        ax2.set_ylabel("Fitness")
        ax2.set_title("Fitness for Each Selection")
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # Plot 3: Threshold
        ax3 = axes[1, 0]
        x3, y3 = zip(*sorted(self._dict_num_var_threshold.items()))
        ax3.plot(x3, y3, marker="o", color="tab:blue")
        ax3.set_xlabel("Number of Selected Variables")
        ax3.set_ylabel("Threshold")
        ax3.set_title("Threshold for Each Selection")
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # Plot 4: Importance (continuous line) and Delta Importance (red crosses)
        ax4 = axes[1, 1]
        x4, y4 = zip(*sorted(self._dict_num_var_selected_importance.items()))
        ax4.plot(
            x4, y4, marker="o", label="Selected Variables' Importance", color="navy"
        )

        x5, y5 = zip(
            *sorted(
                self._dictionary_deltas_importance_diferences_with_exponential_decay.items()
            )
        )
        ax4.scatter(
            x5,
            y5,
            marker="x",
            s=50,
            color="red",
            label="Delta Importance with Exp Decay",
        )

        x_argmax = sum(self._optimal_variable_selection_and_num_of_clusters[0])
        ax4.axvline(
            x=x_argmax,
            color="black",
            linestyle="--",
            label=f"Automatic solution with {x_argmax} variables achieving a fitness of: {self._dict_num_var_selected_fitness[x_argmax]:.3f}",
        )

        ax4.set_xlabel("Number of Selected Variables")
        ax4.set_ylabel("Importance")
        ax4.set_title("Importance Metrics")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        plt.tight_layout()
        output_path = os.path.join(
            self.output_directory, "results", "analysis_by_number_of_variables.png"
        )
        plt.savefig(output_path)
        plt.close(fig)
        if self.verbose:
            print(f"Analysis by number of variables plot saved to {output_path}")

    def plot_num_variables_and_clusters_3D(self):
        """
        Creates a 3D plot showing the relationship between number of variables,
        number of clusters, and fitness values.
        """
        if self._dict_num_var_selection_with_that_num_variables == {}:
            raise ValueError(
                "Variable weight analysis not performed. Please run `analyze_variable_weights` before plotting."
            )

        try:
            # Prepare data points for 3D plot
            x = []
            y = []
            z = []

            for (
                _num_vars
            ) in self._dict_num_var_selection_with_that_num_variables.keys():
                dict_num_cluster_fit = self._dict_num_var_all_clusters_fitness[
                    _num_vars
                ]
                for num_clusters in dict_num_cluster_fit.keys():
                    fitness_value = dict_num_cluster_fit[num_clusters]
                    x.append(num_clusters)
                    y.append(_num_vars)
                    z.append(fitness_value)

            # Create 3D plot
            fig3D = plt.figure(figsize=(10, 8))
            ax3D = fig3D.add_subplot(111, projection="3d")
            ax3D.plot_trisurf(x, y, z, cmap="viridis", alpha=0.8)

            ax3D.set_title("3D Plot: Clusters vs Variables vs Fitness")
            ax3D.set_xlabel("Number of Clusters")
            ax3D.set_ylabel("Number of Variables")
            ax3D.set_zlabel("Fitness Value")

            plt.tight_layout()

            # Save the plot
            output_path_3D = os.path.join(
                self.output_directory, "results", "3D_plot_vars_clusters_fitness.png"
            )
            plt.savefig(output_path_3D, dpi=300, bbox_inches="tight")
            plt.close(fig3D)

            if self.verbose:
                print(f"3D plot saved to: {output_path_3D}")

        except Exception as e:
            warnings.warn(
                f"Couldn't create a 3D plot for Variables vs Clusters vs Fitness. Error: {str(e)}",
                UserWarning,
            )

    def get_plot_comparing_solution_with_another_metric(
        self, new_metric, true_number_of_labels=None, output_path=None
    ):
        """
        Generate comparison plots between GAUFS fitness and external metric.

        Creates two side-by-side plots:

        - Left: Variables vs Used Fitness in GAUFS execution
        - Right: Variables vs Provided Metric for each selection

        This allows comparison between the fitness used in GAUFS and an
        external metric of interest.

        Parameters
        ----------
        new_metric : EvaluationMetric
            Must implement evaluation interface.
        true_number_of_labels : int or None, optional
            True number of labels from the data. If specified, plots include
            a baseline comparing the score obtained with the true number of
            labels as number of clusters. Default is None.
        output_path : str or None, optional
            Path to save the generated plot. If None, saves to
            ``self.output_directory/comparison_fitness_vs_external_metric.png``.

        Returns
        -------
        str
            Path where the plot was saved.
        """
        if self._dict_num_var_selection_with_that_num_variables == {}:
            raise ValueError(
                "Variable weight analysis not performed. Please run `analyze_variable_weights` before plotting."
            )

        # Extract data
        num_vars = sorted(self._dict_num_var_selection_with_that_num_variables.keys())
        x_argmax = sum(self._optimal_variable_selection_and_num_of_clusters[0])

        # Calculate external metric for each selection
        fitness_values = []
        external_metrics = []

        fitnesses_with_true_labels = []
        external_metrics_with_true_labels = []

        # missing values of fake slections will not be calculated as they are not real selections
        for i in num_vars:
            selection = self._dict_num_var_selection_with_that_num_variables[i]
            n_clusters = self._dict_num_var_selected_num_clusters[i]

            fitness_values.append(self._dict_num_var_selected_fitness[i])
            metric_value = _evaluate_ind(
                unlabeled_data=self._unlabeled_data,
                cluster_number=n_clusters,
                variables=selection,
                clustering_method=self.clustering_method,
                evaluation_metric=new_metric,
            )
            external_metrics.append(metric_value)

            # If true number of labels is provided, calculate metrics for that as well
            if true_number_of_labels is not None:
                fitness_true_labels = _evaluate_ind(
                    unlabeled_data=self._unlabeled_data,
                    cluster_number=true_number_of_labels,
                    variables=selection,
                    clustering_method=self.clustering_method,
                    evaluation_metric=self.evaluation_metric,
                )
                fitnesses_with_true_labels.append(fitness_true_labels)

                metric_true_labels = _evaluate_ind(
                    unlabeled_data=self._unlabeled_data,
                    cluster_number=true_number_of_labels,
                    variables=selection,
                    clustering_method=self.clustering_method,
                    evaluation_metric=new_metric,
                )
                external_metrics_with_true_labels.append(metric_true_labels)

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left plot: Variables vs Fitness
        # plot with discontinuities
        _plot_discontinuous(
            ax1,
            num_vars,
            fitness_values,
            marker="o",
            linewidth=2,
            markersize=8,
            color="tab:blue",
            label="Fitness for each selection with its estimated number of clusters",
        )
        ax1.set_xlabel("Number of Variables", fontsize=12)
        ax1.set_ylabel("Fitness", fontsize=12)
        ax1.set_title("Used Fitness in GAUFS", fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.axvline(
            x=x_argmax,
            color="black",
            linestyle="--",
            label=f"Automatic solution with {x_argmax} variables achieving a fitness of: {self._dict_num_var_selected_fitness[x_argmax]:.3f}",
        )
        ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax1.legend()

        # Right plot: Variables vs External Metric
        # plot with discontinuities
        _plot_discontinuous(
            ax2,
            num_vars,
            external_metrics,
            marker="s",
            linewidth=2,
            markersize=8,
            color="tab:blue",
            label="Metric for each selection with its estimated number of clusters",
        )
        ax2.set_xlabel("Number of Variables", fontsize=12)
        ax2.set_ylabel("Metric", fontsize=12)
        ax2.set_title("New Given Metric for Comparison", fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.axvline(
            x=x_argmax,
            color="black",
            linestyle="--",
            label=f"Automatic solution with {x_argmax} variables achieving a metric of: {external_metrics[num_vars.index(x_argmax)]:.3f}",
        )
        ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax2.legend()

        # Add baseline scores for true number of labels
        if true_number_of_labels is not None:
            # Fitness for each selection with True Labels and Metric for each selection with True Labels
            # plot with discontinuities
            _plot_discontinuous(
                ax1,
                num_vars,
                fitnesses_with_true_labels,
                marker="o",
                linestyle="--",
                color="red",
                label="Fitness for each selection with True Number of Labels",
            )
            _plot_discontinuous(
                ax2,
                num_vars,
                external_metrics_with_true_labels,
                marker="s",
                linestyle="--",
                color="red",
                label="Metric for each selection with True Number of Labels",
            )
            ax1.legend()
            ax2.legend()

        plt.tight_layout()

        # Save the plot
        output_path = (
            os.path.join(
                self.output_directory, "comparison_fitness_vs_given_metric.png"
            )
            if output_path is None
            else output_path
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        if self.verbose:
            print(f"Comparison plot saved to: {output_path}")

        return output_path
