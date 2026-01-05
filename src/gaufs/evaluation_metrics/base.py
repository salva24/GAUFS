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

        Args:
            true_labels: Ground truth labels. Required only for external metrics. It can be left as None for internal metrics.
        """
        self.true_labels = true_labels

    def compute(self, assigned_clusters, unlabeled_data=None):
        """
        Compute the metric given cluster assignments.
        Must be implemented by each metric subclass.

        Args:
            assigned_clusters: The cluster assignments to evaluate
            unlabeled_data: The original data (DataFrame or None). Required for internal metrics. If the metric does not need it, it can be left as None.

        Returns:
            float: The metric score
        """
        raise NotImplementedError("Subclasses must implement compute()")
