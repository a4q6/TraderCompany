# TraderCompany

### summary
- Formula:  
    弱学習器. 過去のfeature値から適当に2項演算と活性化を噛ませて予測値を出力.  
    (feature1_with_lag1, feature2_with_lag2) - binOp -> - activation -> prediction.  
  
- Trader:  
    Formulaをまとめる中間学習器. M本のFormulaを持つ.
    各Formulaへのweightを保持しており, 重み和をとって予測値を出力する.  
    weightは後述のCompanyによる"educate"により, labelとの2乗誤差を最小化するように最適化される.
  
- Company:  
    最適化を兼ねた強学習器. N人のtraderを保持している.  
    Companyの予測はN人のtraderの予測値をアンサンブルして出力.    
    - educate():  
        下位TraderについてOLSでweightを調整し直す.
    - prune():  
        下位Traderをdiscardし, 上位TraderにfitさせてGaussianMixtureからのサンプルで置き換える.
