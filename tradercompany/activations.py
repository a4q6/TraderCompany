import numpy as np
import pandas as pd
from typing import Any, List, Dict, Collection, Union, Callable

N_ACT = 5

# funcs
def ReLU(x: float) -> float:
    return max(0, x)

sign = np.sign

tanh = np.tanh

exp = np.exp

def linear(x: float) -> float:
    return x


# hash table
func_to_int = {
    ReLU: 0,
    sign: 1,
    tanh: 2,
    exp: 3,
    linear: 4
}

int_to_func = {
    0: ReLU,
    1: sign,
    2: tanh,
    3: exp,
    4: linear
}