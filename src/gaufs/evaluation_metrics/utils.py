"""
Internal utility functions for evaluation metrics.

This module contains helper functions used internally by the library.
"""

import pandas as pd


# Contingency table
@staticmethod
def _get_cont_table(c1, c2, round=True):
    """
    Generate a contingency table (cross-tabulation) between two categorical variables.
    """
    res = pd.crosstab(c1, c2, normalize="index") * 100
    return res.round(1) if round else res
