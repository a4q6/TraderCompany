import numpy as np
import pandas as pd
from typing import Any, List, Dict, Collection, Union, Callable


""" 
    Aggregation Functions:
        np.ndarray[float]-> float
"""


def simple_average(predictions: np.ndarray) -> float:
    return predictions.mean()


def score_positive_average(predictions: np.ndarray, scores: np.ndarray) -> float:
    return predictions[scores > 0].mean()


def top_average(predictions: np.ndarray, scores: np.ndarray, n_pct: float) -> float:
    theta = np.percentile(scores, n_pct)
    return predictions[scores > theta].mean()

