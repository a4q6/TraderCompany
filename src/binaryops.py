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


func_to_int = {
    min: 1,
    max: 2,
    add: 3,
    sub: 4,
    mul: 5,
    get_left: 6,
    get_right: 7,
    left_upper: 8,
    right_upper: 9
}

int_to_func = {
    1: min,
    2: max,
    3: add,
    4: sub,
    5: mul,
    6: get_left,
    7: get_right,
    8: left_upper,
    9: right_upper
}
