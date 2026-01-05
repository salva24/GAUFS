"""
Clustering experiment implementations for GAUFS.

This subpackage provides a base clustering experiment class and concrete
implementations of different clustering algorithms.
Users can extend the base class to implement custom clustering strategies.
"""

from .base import ClusteringExperiment
from .hierarchical import HierarchicalExperiment
from .kmeans import KmeansExperiment

__all__ = [
    "ClusteringExperiment",
    "HierarchicalExperiment",
    "KmeansExperiment",
]
