GAUFS Documentation
===================

Genetic Algorithm for Unsupervised Feature Selection for Clustering (GAUFS).

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   self
   modules

Overview
--------

GAUFS provides:

- **Gaufs**: the main class for running the algorithm
- **DataGenerator**: synthetic data generators for clustering experiments
- **read_unlabeled_data_csv** and **read_labeled_data_csv**: functions to load datasets from CSV files
- **clustering_experiments**: Subpackage providing base and concrete clustering algorithms. Users can extend the `ClusteringExperiment` to implement custom clustering algorithms.
- **evaluation_metrics**: Subpackage providing internal and external metrics for clustering evaluation, as well as the `EvaluationMetric` base class for custom metrics.

API Reference
-------------

.. currentmodule:: gaufs

- :class:`Gaufs` - Main GAUFS algorithm class
- :class:`DataGenerator` - Synthetic data generator
- :func:`read_unlabeled_data_csv` - Load unlabeled data from CSV
- :func:`read_labeled_data_csv` - Load labeled data from CSV
- :mod:`clustering_experiments` - Clustering experiment implementations
- :mod:`evaluation_metrics` - Evaluation metrics for clustering

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`