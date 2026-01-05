"""
GAUFS: Genetic Algorithm for Unsupervised Feature Selection.

This package provides the main GAUFS interface and core public classes.

The API exposes:
- GAUFS: the main class for running the algorithm
- DataGenerator: two synthetic data generators for clustering experiments
- `read_unlabeled_data_csv` and `read_labeled_data_csv`: functions to load datasets from CSV files

Additional functionality, such as clustering experiments and evaluation
metrics, is available through dedicated subpackages.
"""

from .gaufs import Gaufs
from .data_generator import DataGenerator
from .utils import read_unlabeled_data_csv, read_labeled_data_csv

__all__ = ["Gaufs", "DataGenerator", "read_unlabeled_data_csv", "read_labeled_data_csv"]
