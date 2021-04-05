import numpy as np
import pandas as pd
from typing import Any, List, Dict, Collection, Union, Callable
from .Activations import *
from .BinaryOps import *

class Trader:

    def __init__(self, 
                 n_terms: int,
                 activation: Callable[[float], float],
                 weights: List[float],
                 binary_ops: List[Callable[[float], float]],
                 lag_term1: List[int],
                 lag_term2: List[int],
                 idx_term1: List[int],
                 idx_term2: List[int]) -> None:
        """ Trader object.
        Args:
            n_terms (int): M in article. j=1,2,3,..M
            activation (List[Callable[[float], float]]): A_j in article.
            weights (np.ndarray): w_j in article.
            binary_ops (List[Callable[float], float]): O_j in article.
            lag_term1 (np.ndarray): D_j in article.
            lag_term2 (np.ndarray): F_j in article.
        """
        self.n_terms = n_terms
        self.activation = activation
        self.weights = weights
        self.binary_ops = binary_ops
        self.lag_term1 = lag_term1 - 1
        self.lag_term2 = lag_term2 - 1
        self.idx_term1 = idx_term1
        self.idx_term2 = idx_term2


    def predict(self, feature_ts: np.ndarray) -> float:
        """
        Args:
            feature_ts (np.ndarray): array with shape of [time, #feature]
        Returns:
            float: predict value for targeting asset
        """
        res = 0
        for j in range(self.n_terms):
            res += self.weights[i] * self.activation[j](feature_ts[-self.lag_term1[j], self.idx_term1[j]], 
                                                        feature_ts[-self.lag_term2[j], self.idx_term2[j]] )
        return res