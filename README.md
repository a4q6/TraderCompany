# TraderCompany

Formulaが弱学習器になっている (sum M_i 本ある)
    activation (Callable[[float]):
    binary_op (Callable[[float, float], float]):
    lag_term1 (int): 
    lag_term2 (int):
    idx_term1 (int): 
    idx_term2 (int):

TraderはFormulaをまとめる中間学習器 (i=1,2,3...N人):
    + weight
    * educateおよび,pruneのために過去の成績を保持しておく必要がある.


Companyは最適化を兼ねた強学習器:
    * 連続したtimeseriesを与えてfitする必要がある.

    educate:
        OLS

    prune:
        GM
