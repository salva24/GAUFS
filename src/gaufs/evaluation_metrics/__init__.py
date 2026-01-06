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
Evaluation metrics for clustering algorithms in GAUFS.

This subpackage provides internal and external clustering evaluation metrics,
as well as the EvaluationMetric base class for implementing custom metrics.
"""

from .base import EvaluationMetric

from .internal import (
    SilhouetteScore,
    CalinskiHarabaszScore,
    DaviesBouldinScore,
    DaviesBouldinScoreForMaximization,
    DunnScore,
    SSEScore,
    SSEScoreForMaximization,
)

from .external import (
    AdjustedMutualInformationScore,
    AdjustedRandIndexScore,
    VMeasureScore,
    FowlkesMallowsScore,
    FScore,
    NMIScore,
    HScore,
    Chi2,
    DobPertScore,
)

__all__ = [
    "EvaluationMetric",
    "SilhouetteScore",
    "CalinskiHarabaszScore",
    "DaviesBouldinScore",
    "DaviesBouldinScoreForMaximization",
    "DunnScore",
    "SSEScore",
    "SSEScoreForMaximization",
    "AdjustedMutualInformationScore",
    "AdjustedRandIndexScore",
    "VMeasureScore",
    "FowlkesMallowsScore",
    "FScore",
    "NMIScore",
    "HScore",
    "Chi2",
    "DobPertScore",
]
