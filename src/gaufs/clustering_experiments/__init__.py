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
