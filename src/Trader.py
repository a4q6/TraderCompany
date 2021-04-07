import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, List, Dict, Collection, Union, Callable

from .activations import *
from .binaryops import *
from .Formula import Formula


class Trader:

    def __init__(self, weights, formulas):
        self.n_terms = len(formulas)
        self.weights = weights
        self.formulas = formulas


    def predict(self, feature_array: np.ndarray) -> float:
        """
        Args:
            feature_array (np.ndarray): array with shape of [time, #feature]
        Returns:
            float: predict value for targeting asset
        """
        return sum([self.weights[j] * self.formulas[j].predict(feature_array) for j in range(self.n_terms)])
