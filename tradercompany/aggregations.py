import numpy as np
import pandas as pd
from typing import Any, List, Dict, Collection, Union, Callable


""" 
    Aggregation Functions:
        np.ndarray[float]-> float
"""


def simple_average(predictions: np.ndarray) -> float:
    return predictions.mean()

