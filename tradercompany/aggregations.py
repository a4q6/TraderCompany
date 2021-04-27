import numpy as np
import pandas as pd
from typing import Any, List, Dict, Collection, Union, Callable


""" 
    Aggregation Functions:
        np.ndarray[float]-> float
"""


def simple_average(predictions: np.ndarray, **kwargs) -> float:
    return predictions.mean()


def score_positive_average(predictions: np.ndarray, _self, **kwargs) -> float:
    return predictions[_self.scores > 0].mean()


def top_average(predictions: np.ndarray, _self, n_pct, **kwargs) -> float:
    theta = np.percentile(_self.scores, n_pct)
    return predictions[_self.scores > theta].mean()

