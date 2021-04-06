import numpy as np
import pandas as pd
from typing import Any, List, Dict, Collection, Union, Callable

N_ACT = 5

def ReLU(x: float) -> float:
    return np.max(0, x)

sign = np.sign

tanh = np.tanh

exp = np.exp

def linear(x: float) -> float:
    return x


func_to_int = {
    ReLU: 1,
    sign: 2,
    tanh: 3,
    exp: 4,
    linear: 5
}

int_to_func = {
    1: ReLU,
    2: sign,
    3: tanh,
    4: exp,
    5: linear
}