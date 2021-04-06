import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, List, Dict, Collection, Union, Callable

from .Trader import Trader
from . import activations
from . import binaryops

N_PARAMTYPE = 7  # weights, activation, .... idx_term2


@dataclass
class Parameter:
    n_terms: int
    activation: Callable[[float], float]
    weights: List[float]
    binary_ops: List[Callable[[float], float]]
    lag_term1: List[int]
    lag_term2: List[int]
    idx_term1: List[int]
    idx_term2: List[int]

    def to_numeric_repr(self):
        """ get numerical represnentation of Trader parameter.
        """
        return\
            [activations.func_to_int(f) for f in self.activation]\
            + self.weights\
            + [binaryops.func_to_int(b) for b in self.binary_ops]\
            + self.lag_term1\
            + self.lag_term2\
            + self.idx_term1\
            + self.idx_term2

    @staticmethod
    def from_numeric_repr(numeric_repr):
        """
        Args:
            numeric_repr ([type]): [description]
        Returns:
            [type]: [description]
        """

        assert len(numeric_repr) % N_PARAMTYPE != 0,\
            "inconsistent length of parameter array. check numeric_repr."

        n_term = int(len(numeric_repr) / N_PARAMTYPE)
        return Parameter(
            n_term = n_term,
            activations = [activations.int_to_func(i) for i in numeric_repr[:n_term]],
            weights = numeric_repr[n_term : 2*n_term],
            binary_ops = [binaryops.int_to_func(i) for i in numeric_repr[2*n_term : 3*n_term]],
            lag_term1 = numeric_repr[3*n_term : 4*n_term],
            lag_term2 = numeric_repr[4*n_term : 5*n_term],
            idx_term1 = numeric_repr[5*n_term : 6*n_term],
            idx_term2 = numeric_repr[6*n_term : 7*n_term]
        )


def make_trader(param: Parameter) -> Trader:
    """ Make Trader instance from parameter.
    Args:
        param (Parameter):
    Returns:
        Trader: 
    """
    return Trader(
        param.n_terms,
        param.activation,
        param.weights,
        param.binary_ops,
        param.lag_term1,
        param.lag_term2,
        param.idx_term1,
        param.idx_term2
    )


def make_random_trader(n_terms: int, n_features: int) -> Trader:
    pass


def get_param(trader: Trader) -> Parameter:
    """ Get Parameter instance from Trader instance.
    Args:
        trader (Trader):
    Returns:
        Parameter:
    """
    return Parameter(
        trader.n_terms,
        trader.activation,
        trader.weights,
        trader.binary_ops,
        trader.lag_term1,
        trader.lag_term2,
        trader.idx_term1,
        trader.idx_term2
    )

