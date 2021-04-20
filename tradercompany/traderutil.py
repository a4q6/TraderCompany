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


def make_random_trader(max_terms: int, n_features: int, max_lag: int) -> Trader:
    """ 
    Args:
        max_terms (int): max size of M in article. (* lag=0で与えられたfeatureのlatestの値のみ使う.)
        n_features (int): S in article
        max_lag (int): max lookback periods.
    Returns:
        Trader: 
    """
    assert max_lag > 0

    n_terms = np.random.randint(1, max_terms+1)
    weights = np.random.uniform(-1, 1, size=n_terms)

    formulas = [
        Formula(
            activations.int_to_func[np.random.randint(N_ACT)],
            binaryops.int_to_func[np.random.randint(N_BINOP)],
            np.random.randint(1, max_lag+1),  # 1 ~ max_lag+1 の値がrandomで出る.
            np.random.randint(1, max_lag+1),  # 1だけ大きい数になっているのは lag=0 <-> index=-1 に対応するため.(array[-1]で末尾)
            np.random.randint(n_features),
            np.random.randint(n_features)
        )
        for j in range(n_terms)
    ]

    return Trader(weights, formulas, max_lag)
