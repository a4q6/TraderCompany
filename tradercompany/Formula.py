import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, List, Dict, Collection, Union, Callable

from . import activations
from . import binaryops
from .activations import *
from .binaryops import *
from .activations import N_ACT
from .binaryops import N_BINOP

N_FORMULA_PARAM = 6  # activation, ... idx_term2


class Formula:

    def __init__(self,
                 activation: Callable[[float], float],
                 binary_op: Callable[[float, float], float],
                 lag_term1: int,
                 lag_term2: int,
                 idx_term1: int,
                 idx_term2: int) -> None:
        """
        Args:
            activation (Callable[[float]):
            binary_op (Callable[[float, float], float]):
            lag_term1 (int): 
            lag_term2 (int):
            idx_term1 (int): 
            idx_term2 (int): 
        """
        self.activation = activation
        self.binary_op = binary_op
        self.lag_term1 = lag_term1 + 1
        self.lag_term2 = lag_term2 + 1
        self.idx_term1 = idx_term1
        self.idx_term2 = idx_term2


    def predict(self, feature_array: np.ndarray) -> float:
        """
        Args:
            feature_array (np.ndarray): array with shape of [time, #feature]
        Returns:
            float: predict value for targeting asset
        """
        res =\
            self.activation(
            self.binary_op(
                feature_array[-self.lag_term1, self.idx_term1],
                feature_array[-self.lag_term2, self.idx_term2]
            )
        )
        return res


    def to_numerical_repr(self) -> Collection[float]:
        """ get numerical array representation of formula
        Returns:
            Collection[float]:
        """
        return [
            activations.func_to_int[ self.activation ],
            binaryops.func_to_int[ self.binary_op ],
            self.lag_term1,
            self.lag_term2,
            self.idx_term1,
            self.idx_term2
        ]


    @staticmethod
    def from_numerical_repr(numerical_repr: Collection[float]):
        """ restore Formula instance from numerical representation array.
        Args:
            numeric_repr (Collection[float]): 
        Returns:
            Formula: 
        """
        return Formula(
            activation = activations.int_to_func[ int(round(numerical_repr[0])) ],  # 0-origin
            binary_op  = binaryops.int_to_func[ int(round(numerical_repr[1])) ],  # 0-origin
            lag_term1 = numerical_repr[2],  # 0-origin
            lag_term2 = numerical_repr[3],  # 0-origin
            idx_term1 = numerical_repr[4],  # 0-origin
            idx_term2 = numerical_repr[5],  # 0-origin
        )

    
    def to_str(self, feature_names: Collection) -> str:
        str_expr = f"{self.activation.__name__}, {self.binary_op.__name__}, {feature_names[self.idx_term1]}[-{self.lag_term1}], {feature_names[self.idx_term2]}[{-self.lag_term2}]"
        return str_expr