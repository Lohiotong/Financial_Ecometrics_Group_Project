#  From Price Prediction to Signal Extraction  
### A Comparative Study of Traditional and Machine Learning Models in Forecasting Stock Excess Returns

---

##  Project Overview  
This project proposes a robust excess return forecasting framework targeting 10-day forward stock performance in the U.S. tech sector.  
By comparing a suite of models—from traditional time series methods (ARIMA, GARCH), linear regressions (OLS, LASSO), to machine learning models (Random Forest, XGBoost, LSTM)—we evaluate not only prediction accuracy but also signal directionality and tradability.

Key contributions include:
- Reformulating price-level prediction into return signal extraction
- Investigating signal usability across firm types and market regimes
- Integrating backtesting to assess trading potential

---
##  Directory Structure

```
data/
├── RF_output/                         # Random Forest prediction results  
├── XGB_result/                        # XGBoost prediction results  
│
├── interim/                          
│   ├── cleaned/                       # Cleaned full data (long-form)  
│   │   └── cleaned_all_companies_long.csv  
│   └── transformed/                   # With lags, rolling metrics, EWM smoothed  
│       └── all_companies_long_sorted.csv  
│
├── result/                            # LSTM predictions and final strategy output  
│   ├── final_mixed_company_result/    # Individual company forecasts & plots (15 firms)  
│   ├── final_mixed_model/             # LSTM structure and saved output  
│   ├── portfolio_net_value.png        # Backtest cumulative net value  
│   ├── selected_factors.csv           # Top-20 features selected by RF  
├── company_comparison_vol_vs_price.xlsx 
├── final_combined_data_dictionary.xlsx  # Dictionary of all factor fields
│
├── ALL_Factor_Selection.py            # Master script for factor selection  
├── Arima.R                            # ARIMA model implementation  
├── Clean_Data.py                      # Data cleaning script  
├── Garch.R                            # GARCH volatility modeling  
├── LSTMCSV.py                         # LSTM training & prediction script  
├── OLS & Lasso.R                      # Linear regression models  
├── RF.ipynb                           # Random Forest implementation notebook  
├── XGB.ipynb                          # XGBoost implementation notebook  
└── stockbacktest.py                   # Portfolio backtesting logic  
```

---

##  Modeling Codebase

| Script | Description |
|--------|-------------|
| `ALL_Factor_Selection.py` | Random Forest-based feature ranking |
| `Clean_Data.py`           | Cleaning, NA handling, clipping outliers |
| `Arima.R`, `Garch.R`      | Traditional time series models |
| `OLS & Lasso.R`           | Factor-based linear return prediction |
| `RF.ipynb`, `XGB.ipynb`   | Tree-based models with tuning |
| `LSTMCSV.py`              | LSTM model training & inference |
| `stockbacktest.py`        | Rule-based signal-to-trade backtesting |

---

##  Target Variable and Evaluation Metrics

- **Target**: 10-day forward excess return, EWM smoothed, ±30% clipped
- **Metrics**:
  - MAE, R²
  - Direction Accuracy (DA): percentage of correct directional calls

---

##  Models Compared

- **Traditional**: ARIMA, GARCH
- **Linear**: OLS, LASSO
- **Tree-based ML**: Random Forest, XGBoost
- **Deep Learning**: LSTM (sequence model with BatchNorm + Dropout)

All machine learning models share the same unified top-20 features selected via Random Forest.

---

##  Firm Selection and Data Scope

- **Universe**: 15 S&P 500 tech companies, 2020–2024 (daily frequency)
- **Training Firms (n=12)**: Diverse subsectors (semiconductors, cloud, platforms)
- **Holdout Firms (n=3)**: Out-of-distribution validation for generalizability
- **Macroeconomic & Market Data**: VIX, interest rates, FX, commodities

---

##  Forecasting vs Tradability

- Directional accuracy >60% in most test cases
- LSTM provides smoother, more stable signals → fit for mid-horizon strategy
- Focus shifts from point prediction to signal utility and economic interpretability

---

##  Theoretical Implications

| Model Class     | Insight |
|-----------------|---------|
| ARIMA/GARCH     | Fails to capture adjusted returns → semi-strong EMH challenged |
| OLS/LASSO       | Captures priced factors → rigid to regime shifts |
| LSTM/XGBoost    | Learns nonlinear latent signals → challenges return efficiency |

---

##  Future Directions

### 1. **Adopt Advanced Architectures**
- Transformers / GNNs to model cross-firm relationships

### 2. **Feature Engineering + Data Fusion**
- Integrate high-frequency, or sentiment-based alternative data

### 3. **Portfolio-Aware Modeling**
- DRL-based dynamic allocation (DDPG)
- Sharpe-optimized loss functions

### 4. **XAI & Analyst Copilots**
- SHAP/LIME for factor interpretability
- Build interactive tools for institutional analyst feedback

---

##  Acknowledgments  
This project was developed as part of a university-level financial econometrics course.  
All stock-level and macroeconomic data used were publicly available or obtained via academic terminals.

---

> *For any questions, please contact the project team lead or refer to the included data dictionary.*