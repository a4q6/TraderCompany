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


def make_random_trader(max_terms: int, n_features: int, maxlag: int) -> Trader:
    """ 
    Args:
        max_terms (int): max size of M in article
        n_features (int): S in article
    Returns:
        Trader: 
    """

    n_terms = np.random.randint(1, max_terms+1)
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
