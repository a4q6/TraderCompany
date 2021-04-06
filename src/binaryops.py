import numpy as np
import pandas as pd
from typing import Any, List, Dict, Collection, Union, Callable

N_BINOP = 9

min = min

max = max

def add(x: float, y: float) -> float:
    return x + y

def sub(x: float, y: float) -> float:
    return x - y

def mul(x: float, y: float) -> float:
    return x * y

def get_left(x: float, y: float) -> float:
    return x

def get_right(x: float, y: float) -> float:
    return y

def left_upper(x: float, y: float) -> float:
    return (x > y) * 1.

def right_upper(x: float, y: float) -> float:
    return (x < y) * 1.