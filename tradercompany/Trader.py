import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, List, Dict, Collection, Union, Callable
warnings.filterwarnings("ignore")
from statsmodels.api import OLS

from .activations import *
from .binaryops import *
from .Formula import Formula


class Trader:

    def __init__(self, weights, formulas, max_lag):
        self.n_terms = len(formulas)
        self.weights = weights
        self.formulas = formulas
        self.max_lag = max_lag
        self.score = 0
        self._pred_hist = np.zeros(max_lag)*np.nan  # length T
        self._pred_hist_formulas = np.zeros([max_lag, len(formulas)])*np.nan  # length [T, n_terms]
        self._time_index = 0


    def predict(self, feature_arr: np.ndarray) -> float:
        """ traderの予測値を返す
        Args:
            feature_array (np.ndarray): array with shape of [time, #feature]
        Returns:
            float: predict value for targeting asset
        """
        return sum([self.weights[j] * self.formulas[j].predict(feature_arr) for j in range(self.n_terms)])


    def _predict_with_formula(self, feature_arr: np.ndarray) -> List:
        """ トレーダの予測値と各formulaの予測値を計算して返す.
        Args:
            feature_arr (np.ndarray):
        Returns:
            List[float, np.ndarray[float]]: [pred(trader), [pred(formula_i)]]
        """
        pred_formulas = np.array([self.formulas[j].predict(feature_arr) for j in range(self.n_terms)])  # (1, #terms)
        pred_trader = np.sum(self.weights * pred_formulas)
        return pred_trader, pred_formulas.reshape(1, -1)


    def _recalc_predicts_hist(self, feature_arr_block: np.ndarray) -> None:
        """ 予測値の履歴をリセット. すべて計算し直す.
        Args:
            feature_arr_block (np.ndarary): array with shape of [time, maxlag, #feature]
        Effects:
            self._pred_hist
            self._pred_hist_formulas
        """
        T = feature_arr_block.shape[0]
        self._time_index = T
        self._pred_hist = np.zeros(T)
        self._pred_hist_formulas = np.zeros([T, self.n_terms])
        for t in range(T):
            pred, preds_f = self._predict_with_formula(feature_arr_block[t])  # maxlag=0の場合1行目を取り出す.
            self._pred_hist[t] = pred
            self._pred_hist_formulas[t] = preds_f


    def _append_predicts(self, feature_arr: np.ndarray) -> None:
        """ トレーダおよび保持しているFormulaの予測値履歴の末尾を追記.
        Args:
            feature_arr (np.ndarray):
        Effects:
            self._pred_hist
            self._pred_hist_formulas
        """
        self._time_index += 1
        pred, preds_f = self._predict_with_formula(feature_arr)
        self._pred_hist = np.append(self._pred_hist, pred)
        self._pred_hist_formulas = np.append(self._pred_hist_formulas, preds_f)


    def _update_score(self, return_arr: np.ndarray, eval_method="default") -> None:
        """ 
        Args:
            return_arr (np.ndarray): 
            eval_method (str, optional): 
                which method to use to evaluate traders.
                - "default":
                    cumulative return : sum sign(pred_t) * ret_t
        Effects:
            self.score
        """
        if eval_method == "default":
            self.score = np.sum( (np.sign(self._pred_hist) * return_arr) )

        else:
            raise NotImplementedError(f"unknown eval_method : {eval_method}")


    def _update_weights(self, return_arr: np.ndarray) -> None:
        """ weightsを更新
        Args:
            return_arr (np.ndarray):
        Effects:
            self.weights
        """
        y = return_arr
        X = self._pred_hist_formulas
        self.weights = OLS(y, X).fit().params


    def _to_numerial_repr(self) -> List:
        """
        Returns:
            List: {M, (formula_params)_j}
        """
        formula_arr = np.array([formula.to_numerical_repr() for formula in self.formulas])
        return self.n_temrs, formula_arr