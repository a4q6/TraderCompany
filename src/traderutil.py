import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, List, Dict, Collection, Union, Callable

from .Trader import Trader
from .Formula import Formula
from . import activations
from . import binaryops
from .activations import N_ACT
from .binaryops import N_BINOP



""" 
    Parameter消して formulaにする.
    おそらくGMがfitする対象はFormula.
    Formulaが弱学習器になっている.
            activation (Callable[[float]):
            binary_op (Callable[[float, float], float]):
            lag_term1 (int): 
            lag_term2 (int):
            idx_term1 (int): 
            idx_term2 (int): 
"""

def make_random_trader(n_terms: int, n_features: int, maxlag: int) -> Trader:
    """ 
    Args:
        n_terms (int):
        n_features (int): 
    Returns:
        Trader: 
    """
    weights = np.random.uniform(-1, 1, size=n_terms)

    formulas = [
        Formula(
            activations.int_to_func[np.random.randint(1, N_ACT+1)],
            binaryops.int_to_func[np.random.randint(1, N_BINOP+1)],
            np.random.randint(n_features),
            np.random.randint(n_features),
            np.random.randint(maxlag),
            np.random.randint(maxlag)
        )
        for j in range(n_terms)
    ]

    return Trader(weights, formulas)
