import numpy as np
import pandas as pd
from typing import Any, List, Dict, Collection, Union, Callable


def ReLU(x: float) -> float:
    return np.max(0, x)

sign = np.sign

tanh = np.tanh

exp = np.exp

def linear(x: float) -> float:
    return x