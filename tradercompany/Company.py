import warnings
import numpy as np
import pandas as pd
import multiprocessing
from concurrent import futures
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from typing import Any, List, Dict, Collection, Union, Callable, Union
warnings.filterwarnings("ignore")
from statsmodels.api import OLS

from .traderutil import make_random_trader
from .Trader import Trader
from .Formula import Formula, N_FORMULA_PARAM
from .activations import N_ACT
from .binaryops import N_BINOP
from .aggregations import simple_average


N_GMCOMPONENTS_FORMULA = 10 # components
N_GBSTRIDES_FORMULA = 2  # stride
N_GMCOMPONENTS_TERMS = 10
N_GMSTRIDES_TERMS = 2


class Company:

    def __init__(self, 
                 n_traders: int,
                 n_features: int,
                 max_terms: int,
                 max_lag: int,
                 educate_pct: float,
                 aggregate: Callable = simple_average,
                 eval_method: str = "default",
                 eval_lookback: int=100,
                 seed = 1
                 ):
        """
        Args:
            n_traders (int): 
            n_features (int): 
            max_terms (int): 
            max_lag (int): max lookback length. maxlag=0 means model use only latest data.
            educate_pct (float): educate & prune parameter. 0<.<100
            aggregate (Callable, optional): [description]. Defaults to aggregations.simple_average.
            eval_method (str): method to evaluate traders. defaullt is cumulative pnl in paepr definition. 
        """
        self.n_traders = n_traders
        self.n_features = n_features
        self.aggregate = aggregate
        self.max_lag = max_lag
        self.max_terms = max_terms
        self.traders: List[Trader] = [make_random_trader(max_terms, n_features, max_lag) for n in range(n_traders)]
        self.educate_pct = educate_pct  # 0 ~ 100
        self._max_job = multiprocessing.cpu_count() - 1
        self.eval_method = eval_method
        self.eval_lookback = eval_lookback
        np.random.seed(seed)


    def conv_feature(self, feature: Union[List, np.ndarray], label: Union[List, np.ndarray]) -> List:
        """ make feature timeseries block from feature timeseries.
            ** Each timeseries must be continuous, not discrete. (e.g. must not contains Sat,Sun.)
            if passed List, apply blocknize for each array and concatenate to one array.
        Args:
            feature (Union[List, np.ndarray]):
            label (Union[List, np.ndarray]): 
        Returns:
            X,y: np.ndarray[time - maxlag*n_list, maxlag+1, #feature], np.ndarray[time - maxlag*n_list, ]
        """
        if isinstance(feature, list) and isinstance(label, list):
            X = []
            y = []
            for i in range(len(feature)):
                _x, _y = _conv_feature(X[i], y[i])
                X.append(_x)
                y.append(_y)
            X = np.concatenate(X)
            y = np.concatenate(y)
        
        elif isinstance(feature, np.ndarray) and isinstance(label, np.ndarray):
            X, y = self._conv_feature(feature, label)
        
        else:
            raise ValueError("type of feature and label are different. check inputs.")

        return X, y


    def _conv_feature(self, feature_arr: np.ndarray, return_arr: np.ndarray) -> List:
        """ convert feature timeseries to feature timeseries block with max lag.
        Args:
            feature_arr (np.ndarray): array with shape of [time, #feature]
            return_arr (np.ndarray): array with shape of [time, ]
        Returns:
            np.ndarray: feature and label. array with shape of [time - maxlag, maxlag+1, #feature] and [time - maxlag,]
        """
        assert return_arr.shape[0] == feature_arr.shape[0]
        T = return_arr.shape[0]
        X = np.zeros([feature_arr.shape[0], self.max_lag+1, feature_arr.shape[1]]) * np.nan
        for t in range(self.max_lag+1, T):
            X[t] = feature_arr[t-self.max_lag-1:t]
        ind = np.isnan(X[:, 0, 0])
        y = return_arr[~ind]
        X = X[~ind]
        return X, y


    @property
    def scores(self):
        return pd.Series([trader.score for trader in self.traders])

    
    def get_trader_i(self, i):
        return deepcopy(self.traders[i])


    def dynamic_predict(self, 
                        feature_arr_block: np.ndarray, 
                        return_arr: np.ndarray, 
                        t_warm: int=100,
                        calib_stride: int=5,
                        **kwargs_predict
                        ) -> np.ndarray:
        """ 
        Args:
            feature_arr_block (np.ndarray): array with shape of [time, maxlag, #feature]
            return_arr (np.ndarray):
            t_warm (int):
        Returns:
            np.ndarrary:
        """
        pred = np.zeros_like(return_arr) * np.nan

        # warm up on first t_warm samples (0 <= i <= t_warm-1)
        self.recalc_evaluation(feature_arr_block[:t_warm], return_arr[:t_warm])
        self.educate(feature_arr_block[:t_warm], return_arr[:t_warm])
        self.prune_and_generate(feature_arr_block[:t_warm], return_arr[:t_warm])

        for t in tqdm(range(t_warm, return_arr.shape[0])):
            # predict & eval at i=t sample 
            # t_init = max(0, t - calib_lookback)
            t_init = 0
            pred[t] = self.predict(feature_arr_block[:t+1][-1], _self=self, **kwargs_predict)
            self.append_evaluation(feature_arr_block[:t+1][-1], return_arr[:t+1])

            if t % calib_stride == 0:
                # recalibarte on 0<=i<=t sample
                self.educate(feature_arr_block[:t+1], return_arr[:t+1])
                self.prune_and_generate(feature_arr_block[:t+1], return_arr[:t+1])

        return pd.Series(pred)


    def predict(self, feature_arr: np.ndarray, **kwargs) -> float:
        """ return company's prediction value(s).
        Args:
            feature_arr (np.ndarray): array with shape of [time, #feature] or [time, maxlag, #feature]
        Returns:
            float: 
        """
        if feature_arr.ndim == 2:
            # predict on one sample
            return self.aggregate(
                np.array([trader.predict(feature_arr) for trader in self.traders]), **kwargs
            )
        elif feature_arr.ndim == 3:
            # predict on multiple samples 
            return np.array(
                [self.aggregate(np.array([trader.predict(arr) for trader in self.traders]), **kwargs) 
                 for arr in feature_arr]
            )


    def append_evaluation(self,
                          feature_arr: np.ndarray,
                          return_arr: np.ndarray) -> None:
        """tradersの予測履歴の末尾を更新. 評価値を更新.
        Args:
            feature_arr (np.ndarray): array with shape of [time, #feature]
            return_arr (np.ndarray):
            eval_method (str, optional): Defaults to "default".
                "default": total return.
        Effects:
            self.traders 
                .score 
                ._pred_hist
                ._pred_hist_formulas
                ._time_index
        """
        for trader in self.traders:
            trader._append_predicts_hist(feature_arr)
            trader._update_score(return_arr, self.eval_lookback, self.eval_method)


    def recalc_evaluation(self, feature_arr_block: np.ndarray, 
                                return_arr: np.ndarray,
                                n_jobs=1) -> None:
        """ tradersの予測値履歴とその評価値を更新.
        Args:
            feature_arr_block (np.ndarray): array with shape of [time, lookback, n_feature]
            return_arr (np.ndarray): 
            eval_method (str, optional): [description]. Defaults to "default".
        """
        if n_jobs < 2:
            for trader in self.traders:
                trader.recalc(feature_arr_block[-self.eval_lookback:], return_arr[-self.eval_lookback:], self.eval_lookback, self.eval_method)

        elif n_jobs < self._max_job:
            future_list = []
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                for trader in self.traders:
                    future_list.append(
                        executor.submit(trader.recalc, feature_arr_block[-self.eval_lookback:], return_arr[-self.eval_lookback:], self.eval_lookback, self.eval_method))
                traders = futures.as_completed(fs=future_list)
                self.traders = traders
                
        else:
            raise ValueError(f"n_jobs={n_jobs} larger than limit. max_workers is {self._max_job}")


    def educate(self, feature_arr_block: np.ndarray, return_arr: np.ndarray) -> None:
        """ update traders.weights and update each predict history and score
        Effects:
            self.traders
        """
        score_threshold = np.percentile([trader.score for trader in self.traders], q=self.educate_pct)
        for trader in self.traders:
            if trader.score < score_threshold:
                trader._update_weights(return_arr, self.eval_lookback)
                trader._recalc_predicts_hist(feature_arr_block[-self.eval_lookback:])
                trader._update_score(return_arr, self.eval_lookback, self.eval_method)


    def prune_and_generate(self, feature_arr_block: np.ndarray, return_arr: np.ndarray) -> None:
        """上位1-Q[%]に対して, GaussianMixtureをfit. これからサンプリングして下Q[%]を置き換える.
        """

        ## get reference to each groups
        score_threshold = np.percentile([trader.score for trader in self.traders], q=self.educate_pct)
        good_traders = [trader for trader in self.traders if trader.score > score_threshold]
        bad_traders = [trader for trader in self.traders if trader.score <= score_threshold]


        ## fit GM to good traders
        # prepare numerical representation of traders. (each trader has Mi formulas)
        n_row1 = sum([trader.n_terms for trader in good_traders])
        formula_arr = np.zeros([n_row1, N_FORMULA_PARAM])
        # prepare n_terms(= M) array
        n_row2 = len(good_traders)
        nterms_arr = np.zeros(n_row2)
        tmp_idx = 0
        for i,trader in enumerate(good_traders):
            M, formula_numerical = trader._to_numerical_repr()
            nterms_arr[i] = M
            formula_arr[tmp_idx : tmp_idx+M] = formula_numerical
            tmp_idx += M
        # fit
        gmm_form = [
            GaussianMixture(n_components=n, random_state=1).fit(formula_arr)
            for n in range(1, N_GMCOMPONENTS_FORMULA, N_GBSTRIDES_FORMULA)
        ]
        min_idx = np.argmin([gmm.bic(formula_arr) for gmm in gmm_form])
        gmm_form = gmm_form[min_idx]

        gmm_nterm = [
            GaussianMixture(n_components=n, random_state=1).fit(nterms_arr.reshape(-1,1))
            for n in range(1, N_GMCOMPONENTS_TERMS, N_GMSTRIDES_TERMS)
        ]
        min_idx = np.argmin([gmm.bic(nterms_arr.reshape(-1,1)) for gmm in gmm_nterm])
        gmm_nterm = gmm_nterm[min_idx]


        ## replace bad traders to good traders
        n_new_trader = len(bad_traders)
        # generate n_terms
        Ms = gmm_nterm.sample(n_new_trader)[0].reshape(-1,)
        Ms = np.round(Ms).astype(int)
        Ms[Ms==0] = 1
        Ms[Ms > self.max_terms] = self.max_terms
        # generate formula
        formulas = np.round(gmm_form.sample(sum(Ms))[0]).astype(int)
        formulas[formulas < 0] = 0
        formulas[:, [0]] = np.where(formulas[:, [0]] >= N_ACT, N_ACT-1, formulas[:, [0]])
        formulas[:, [1]] = np.where(formulas[:, [1]] >= N_BINOP, N_BINOP-1, formulas[:, [1]])
        formulas[:, [2,3]] = np.where(formulas[:, [2,3]] > self.max_lag, self.max_lag, formulas[:, [2,3]])
        formulas[:, [4,5]] = np.where(formulas[:, [4,5]] > self.n_features-1, self.n_features-1, formulas[:, [4,5]])

        # update bad traders
        tmp_idx = 0
        for i,trader in enumerate(bad_traders):
            M = Ms[i]
            formula_list = [Formula.from_numerical_repr(f) for f in formulas[tmp_idx : tmp_idx+M]]
            trader = Trader(M, formula_list, self.max_lag)
            trader._recalc_predicts_hist(feature_arr_block[-self.eval_lookback:])
            tmp_idx += M
        

    def get_cumpnl(self, i, return_arr):
        return pd.Series(self.get_trader_i(i)._pred_hist * return_arr)