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
