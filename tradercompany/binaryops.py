import numpy as np
import pandas as pd
from typing import Any, List, Dict, Collection, Union, Callable

N_BINOP = 9


# funcs
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


# hash table
func_to_int = {
    min: 0,
    max: 1,
    add: 2,
    sub: 3,
    mul: 4,
    get_left: 5,
    get_right: 6,
    left_upper: 7,
    right_upper: 8
}

int_to_func = {
    0: min,
    1: max,
    2: add,
    3: sub,
    4: mul,
    5: get_left,
    6: get_right,
    7: left_upper,
    8: right_upper
}
