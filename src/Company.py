import warnings
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.mixture import GaussianMixture
from typing import Any, List, Dict, Collection, Union, Callable
warnings.filterwarnings("ignore")
from statsmodels.api import OLS


from .traderutil import make_random_trader
from .Trader import Trader
from .Formula import Formula, N_FORMULA_PARAM
from .activations import N_ACT
from .binaryops import N_BINOP


N_GMCOMP_FORMULA = 10 # components
N_GMSTRD_FORMULA = 2  # stride
N_GMCOMP_TERMS = 10
N_GMSTRD_TERMS = 2


class Company:

    def __init__(self, 
                 n_traders: int,
                 n_features: int,
                 max_terms: int,
                 max_lag: int,
                 aggregate: Callable[[Collection], float],
                 educate_pct: float
                 ):
        self.n_traders = n_traders
        self.aggregate = aggregate
        self.max_lag = max_lag
        self.max_terms = max_terms
        self.traders = [make_random_trader(max_terms, n_features, max_lag) for n in range(n_traders)]
        self.educate_pct = educate_pct  # 0 ~ 100


    def predict(self, feature_arr: np.ndarray) -> float:
        """
        Args:
            feature_arr (np.ndarray): array with shape of [time, #feature]
        Returns:
            float: 
        """
        return self.aggregate([trader.predict(feature_arr) for trader in self.traders])


    def update_evaluation(self,
                          feature_arr: np.ndarray,
                          return_arr: np.ndarray,
                          eval_method="default") -> None:
        """tradersの予測履歴の末尾を更新. 評価値を更新.
        Args:
            feature_arr (np.ndarray):
            return_arr (np.ndarray):
            eval_method (str, optional): Defaults to "default".
        Effects:
            self.traders 
                .score 
                ._pred_hist 
                ._pred_hist_formulas
        """
        for trader in self.traders:
            trader._append_predicts(feature_arr)
            trader._update_score(return_arr, eval_method)


    def educate(self, feature_arr: np.ndarray, return_arr: np.ndarray, eval_method="default") -> None:
        """ update traders.weights and update each predict history and score
        Effects:
            self.traders
        """
        score_threshold = np.percentile([trader.score for trader in self.traders], q=self.educate_pct)
        for trader in self.traders:
            if trader.score < score_threshold:
                trader._update_weights(return_arr)
                trader._recalc_predicts_hist(feature_arr)
                trader._update_score(return_arr, eval_method)


    def prune_and_generate(self, feature_arr: np.ndarray, return_arr: np.ndarray, eval_method="default") -> None:
        """上位1-Q[%]に対して, GaussianMixtureをfit. これからサンプリングして下Q[%]を置き換える.
        """

        ## get reference to each groups
        score_threshold = np.percentile([trader.score for trader in self.traders], q=self.educate_pct)
        good_traders = [trader for trader in self.traders if trader.score > score_threshold]
        bad_traders = [trader for trader in self.traders if trader.score <= score_threshold]


        ## fit GM to good traders
        # prepare numerical representation
        n_row1 = sum([trader.n_terms for trader in good_traders])
        formula_arr = np.zeros([n_row1, N_FORMULA_PARAM])
        n_row2 = len(good_traders)
        terms_arr = np.zeros(n_row2)
        tmp_idx = 0
        for i,trader in enumerate(good_traders):
            M, formula_numerical = trader.to_numerical_repr()
            terms_arr[i] = M
            formula_arr[tmp_idx : tmp_idx+M] = formula_numerical
            tmp_idx += M
        # fit
        gmm_form = [
            GaussianMixture(n_components=n, random_state=1).fit(formula_arr)
            for n in range(1, N_GMCOMP_FORMULA, N_GMSTRD_FORMULA)
        ]
        min_idx = np.argmin([gmm.bic for gmm in gmm_form])
        gmm_form = gmm_form[min_idx]

        gmm_term = [
            GaussianMixture(n_components=n, random_state=1).fit(terms_arr.reshape(-1,1))
            for n in range(1, N_GMCOMP_TERMS, N_GMSTRD_TERMS)
        ]
        min_idx = np.argmin([gmm.bic for gmm in gmm_term])
        gmm_term = gmm_term[min_idx]


        ## replace bad traders to good traders
        n_new_trader = len(bad_traders)
        # generate n_terms
        Ms = gmm_term.sample(n_new_trader)[0].reshape(-1,)
        Ms = np.round(Ms).astype(int)
        Ms[Ms==0] = 1
        Ms[Ms > self.max_temrs] = self.max_terms
        # generate formula
        formulas = np.round(gmm_form.sample(sum(Ms))[0]).astype(int)
        formulas[formulas < 0] = 0
        formulas[formulas[:,[2,3]]>self.max_lag, [2,3]] = self.max_lag
        formulas[formulas[:,[4,5]]>self.n_features, [4,5]] = self.n_features
        # update bad traders
        tmp_idx = 0
        for i,trader in enumerate(bad_traders):
            M = Ms[i]
            formula_list = [Formula.from_numerical_repr(f) for f in formulas[tmp_idx : tmp_idx+M]]
            trader = Trader(M, formula_list, self.max_lag)
            tmp_idx += M
        