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
GAUFS: Genetic Algorithm for Unsupervised Feature Selection for Clustering.

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
