
**Implementation of a Market-Making Trading Strategy on the example of BTCUSDT**

A course project combining machine learning / deep learning **mid-price forecasting** with two classical **optimal market-making** models (Avellaneda–Stoikov and Cartea–Jaimungal–Ricci), integrated into a trading bot that collects data, trains models, backtests, and runs live on the Bybit testnet.

> **HSE University** · Faculty of Computer Science · Master's program *"Financial Technologies and Data Analysis"*
> **Author:** Tatiana D. Sorokina · **Supervisor:** Artem A. Yuliy (Sberbank, Head of Data Research) · Moscow, 2025

---

## Table of Contents

- [Overview](#overview)
- [Objectives](#objectives)
- [Pipeline](#pipeline)
- [Data](#data)
- [Feature Engineering](#feature-engineering)
- [Feature Selection](#feature-selection)
- [Mid-Price Forecasting: Machine Learning](#mid-price-forecasting-machine-learning)
- [Mid-Price Forecasting: Deep Learning (LSTM)](#mid-price-forecasting-deep-learning-lstm)
- [Market-Making Strategies](#market-making-strategies)
- [Risk Management](#risk-management)
- [Results](#results)
- [Key Findings](#key-findings)
- [Future Work](#future-work)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [References](#references)
- [Disclaimer](#disclaimer)

---

## Overview

Cryptocurrency markets are highly volatile, have a complex microstructure, and generate large volumes of real-time data. This makes accurate price forecasting and adaptive spread setting essential for effective market making. The project builds a **market-maker trading bot** that:

1. Predicts the short-term **mid-price** of the BTCUSDT perpetual using ML and DL models built on limit-order-book (LOB) and OHLCV features.
2. Uses those predictions to set **bid/ask quotes** via two optimal market-making frameworks.
3. Dynamically adjusts spreads and order sizes through a **risk-management** layer.
4. Is **backtested** in a simulation environment and **run live** on the Bybit testnet.

The headline result is deliberately honest: **LSTM forecasts mid-price far more accurately than the gradient-boosting baselines (MAE ≈ 42 vs ≈ 108), and the strategies are profitable in simulation — but in live testnet trading the bot produces a negative PnL.** The accuracy and latency of second-ahead predictions, together with a low order-fill rate, are not yet sufficient for profitable real trading. The work is a full end-to-end prototype and a foundation for further research, not a production-ready system.

---

## Objectives

- Build a system to collect and aggregate real-time data from a crypto exchange.
- Generate and select features from the order book and technical indicators.
- Train and compare ML models (LightGBM, CatBoost, Random Forest, XGBoost, NGBoost) for mid-price prediction.
- Build an LSTM deep-learning model to capture temporal dependencies in the financial series.
- Implement and backtest market-making algorithms based on the Avellaneda–Stoikov and Cartea–Jaimungal–Ricci models, with a risk-management component.
- Test the strategies on the Bybit demo (testnet) exchange and analyze their effectiveness.

---

## Pipeline

```
Bybit WebSocket API
        │
        ▼
Data collection ──► historical trades + L2 order book
        │
        ▼
Feature engineering ──► LOB features + 10+ technical indicators
        │
        ▼
Feature selection ──► Correlation Filter | Random Forest | Lasso | PCA
        │
        ▼
Mid-price forecasting ──► GBMs (LightGBM/CatBoost/RF/XGBoost/NGBoost) + LSTM
        │            (Nested TimeSeriesSplit CV, Optuna hyperparameter tuning)
        ▼
Quoting ──► Avellaneda–Stoikov | Cartea–Jaimungal–Ricci
        │
        ▼
Risk management ──► inventory / volatility / imbalance / liquidity adjustments
        │
        ▼
Backtest (simulation)  +  Live run (Bybit testnet)
```

---

## Data

| Parameter | Value |
|---|---|
| Source | Bybit exchange (WebSocket API v5) |
| Instrument | BTCUSDT perpetual futures |
| Streams | Historical trades + L2 order book state |
| Frequency | Second-level |
| Raw fields | `timestamp, open, high, low, close, volume, bid_price, bid_qty, ask_price, ask_qty` |
| Backtest window | 14,632 one-second observations (≈ 4 hours of trading) |

The combined dataset captures both **price dynamics** (OHLCV) and **market liquidity** (best bid/ask prices and quantities), forming the basis for feature engineering and model training.

---

## Feature Engineering

**From the L2 order book:**

| Feature | Definition |
|---|---|
| Mid-price | `(bid_price + ask_price) / 2` |
| Bid-ask spread | `best_bid − best_ask` |
| Order-book imbalance | `(bid_qty − ask_qty) / (bid_qty + ask_qty)` |
| Relative spread | `spread / mid_price` |
| Liquidity pressure | `bid_qty / (bid_qty + ask_qty)` |

**From OHLCV candles (10+ technical indicators):** Volatility (rolling std of closes), Moving Average (MA), RSI, Bollinger Bands, MACD (and MACD histogram), ADX, Ichimoku, ATR, and OBV.

The forecasting **target** is the mid-price one step (one second) ahead.

---

## Feature Selection

Four complementary methods were compared:

| Method | Type | Notes |
|---|---|---|
| **Correlation Filter** | Filter | Fast, model-agnostic; removes highly collinear features. Ignores the target and non-linear relations. Removed 8 features. |
| **Random Forest Importance** | Embedded | Target-aware, captures non-linear interactions; ranked all 10 remaining features. |
| **Lasso (L1)** | Embedded / linear | Zeros out irrelevant coefficients; kept 9 features. Requires scaling. |
| **PCA** | Dimensionality reduction | Orthogonal components; first 9 explain **98.34%** of variance (PC1 alone = 24.16%). Loses interpretability. |

**Top features (RF & Lasso agreement):** `moving_average, rsi, macd, obv, macd_hist, volatility, atr, adx, order_book_imbalance, bid_ask_spread`.

**PCA cumulative explained variance:** 24.16 → 45.24 → 57.86 → 67.86 → 77.62 → 86.99 → 94.01 → 96.44 → 98.34 (%).

---

## Mid-Price Forecasting: Machine Learning

Five ensemble/boosting models were trained: **LightGBM, CatBoost, Random Forest, XGBoost, NGBoost**. Each used:

- **Nested cross-validation** with `TimeSeriesSplit` (respecting temporal order).
- **Optuna** hyperparameter optimization (`OptunaSearchCV`).
- **StandardScaler** feature scaling.
- Metrics: **RMSE, MSE, MAE, R²**.

### Results by feature-selection method

**Lasso features**

| Model | Test RMSE | Test MAE | Test R² |
|---|---|---|---|
| LightGBM | 319.95 | 114.36 | 0.6951 |
| CatBoost | 321.11 | 118.93 | 0.6928 |
| Random Forest | 315.52 | 111.27 | 0.7034 |
| XGBoost | 321.96 | 119.34 | 0.6912 |
| **NGBoost** | **311.05** | **108.02** | **0.7118** |

**Random Forest features**

| Model | Test RMSE | Test MAE | Test R² |
|---|---|---|---|
| LightGBM | 319.75 | 114.23 | 0.6954 |
| CatBoost | 320.04 | 117.74 | 0.6949 |
| Random Forest | 316.70 | 112.68 | 0.7012 |
| XGBoost | 332.03 | 136.99 | 0.6716 |
| **NGBoost** | **311.84** | **109.14** | **0.7103** |

**PCA features**

| Model | Test RMSE | Test MAE | Test R² |
|---|---|---|---|
| LightGBM | 442.02 | 264.06 | 0.4180 |
| CatBoost | 466.14 | 274.37 | 0.3527 |
| Random Forest | 495.79 | 316.25 | 0.2678 |
| XGBoost | 434.46 | 260.43 | 0.4377 |
| NGBoost | 428.28 | 258.15 | 0.4536 |

**Takeaways:** Lasso and Random Forest feature selection perform similarly and clearly beat PCA (which roughly halves R² and doubles MAE — the loss of interpretable, target-relevant information hurts this task). **NGBoost** is the best classical model (MAE ≈ 108, R² ≈ 0.71), but all boosting models are limited in capturing temporal dependencies — which motivated the move to LSTM.

---

## Mid-Price Forecasting: Deep Learning (LSTM)

An LSTM recurrent network was used to capture long-term temporal dependencies that boosting models cannot model natively.

- **Lookback window:** 60 previous observations.
- **Loss:** MSE · **Scaling:** StandardScaler · **Early stopping** on validation loss.
- **Optuna** hyperparameter tuning (`OptunaSearchCV`).

**LSTM by feature-selection method (Optuna: 20 trials, 30 epochs)**

| Features | Test RMSE | Test MAE | Test R² |
|---|---|---|---|
| **Lasso** | **159.53** | **42.03** | **0.9245** |
| Random Forest | 327.72 | 120.65 | 0.6813 |
| PCA | 329.98 | 144.49 | 0.6770 |

**LSTM (Lasso) with more tuning (Optuna: 30 trials, 40 epochs)**

| Model | Test RMSE | Test MAE | Test R² |
|---|---|---|---|
| LSTM | 173.11 | 43.16 | 0.9111 |

**Takeaways:** LSTM with Lasso-selected features roughly **halves RMSE** and cuts **MAE from ≈ 108–120 down to ≈ 42** versus the best boosting models, lifting R² to ≈ 0.92. Additional tuning did not improve results further (MAE plateaued around 43), and LSTM is markedly more compute-intensive to train.

---

## Market-Making Strategies

Predicted mid-price feeds two quoting frameworks.

### 1. Avellaneda–Stoikov

Computes a **reservation price** that shifts away from the predicted mid-price as inventory grows, then places bid/ask quotes around it:

```
reservation_price = predicted_mid_price − inventory · γ · σ² · T
```

where `γ` = risk/liquidity coefficient, `σ²` = market variance, `T` = time to execution. Larger inventory pushes the reservation price away from mid to reduce risk, and quotes are set asymmetrically to encourage inventory-reducing trades.

### 2. Cartea–Jaimungal–Ricci

Sets the spread from a value function that penalizes deviation of inventory `q` from a target level `q_target`:

```
spread = max(0, log(λ / φ) + φ · (q − q_target) / κ)
```

where `λ` = market-order arrival intensity, `φ` = inventory-deviation penalty, `κ` = sensitivity to order-book depth. Here **`q_target = 0`**, so the bot aims to keep inventory at its initial level.

---

## Risk Management

Order sizes and spreads are adjusted in real time:

- **Inventory threshold** — base order size `0.001 BTC`, max inventory `0.1 BTC`. Above the limit, the spread multiplier increases by 50% (`×1.5`) to throttle new positions.
- **Volatility** — above the 90th percentile, spread doubles (`×2.0`); above the 95th percentile, trading is **paused**.
- **Order imbalance** — buyer-heavy (90th pct): bid spread −20%, ask spread +20%; seller-heavy (10th pct): the reverse.
- **Liquidity** — below the 10th percentile of liquidity pressure, order size is cut 50% and ask spread reduced 20% (`×0.8`).

Backtest fills use a **probabilistic execution model**: base fill probability `0.7`, adjusted down as the prediction error (MAE vs. historical percentiles) grows and depending on the sign of the error, to realistically model partial fills under forecast uncertainty.

---

## Results

### Backtest (simulation, ≈ 4 hours / 14,632 seconds)

| Strategy | BTC | USDT | Total PnL |
|---|---|---|---|
| **Stoikov** | −2.73 | 233,535.8 | **249.18** |
| Cartea | −2.76 | 236,154.4 | 217.73 |

Both strategies were **profitable in simulation**, with **Avellaneda–Stoikov performing best**.

### Live run on Bybit testnet (LightGBM, ≈ 1 hour)

| Spread optimization | Cumulative Realized + Unrealized PnL |
|---|---|
| Cartea | **−37.5 USDT** |
| Avellaneda–Stoikov | −59.2 USDT |

In live testnet trading the bot produced a **negative PnL**. On average it loses ~$15 for every 60 orders placed at `0.001 BTC`. With one-second-ahead predictions, fills are infrequent — roughly **2–3 orders per minute**. Runs were repeated across different times and days with consistent results.

---

## Key Findings

- **Deep learning wins on forecasting.** LSTM is substantially more accurate than gradient boosting for mid-price prediction (MAE ≈ 42 / R² ≈ 0.92 vs MAE ≈ 108 / R² ≈ 0.71), confirming the value of modeling temporal structure.
- **Feature selection matters.** Lasso and Random Forest importance give the best and near-identical results; **PCA degrades performance sharply** for this task.
- **Simulation ≠ live.** Both quoting strategies are profitable in backtest but lose money on the testnet. The two spread methods yield comparable outcomes — differences between them are small relative to the impact of forecast inaccuracy and low fill speed.
- **The bottleneck is prediction accuracy + latency.** Second-ahead forecasts are not accurate or fast enough, and order fill rates are low, so the current system is **not yet suitable for profitable real trading** without further improvement.
- **End-to-end contribution.** The project delivers a complete cycle — data collection, feature engineering/selection, ML+DL forecasting, two optimal market-making models, risk management, backtesting, and live testnet integration — as a reusable foundation.

---

## Future Work

- **More advanced architectures:** GRU (a lighter alternative to LSTM that retains long-term modeling) and hybrid **CNN-LSTM** (CNN for feature extraction, LSTM for temporal processing), especially for very short-horizon mid-price forecasting.
- **Additional market factors**, e.g. news/sentiment signals.
- **Lower-latency inference and quoting** to raise fill rates.

---

## Tech Stack

- **Language:** Python
- **ML:** scikit-learn, LightGBM, CatBoost, XGBoost, NGBoost, Random Forest
- **DL:** LSTM (Keras / TensorFlow)
- **Hyperparameter tuning:** Optuna (`OptunaSearchCV`)
- **Validation:** `TimeSeriesSplit` nested cross-validation
- **Preprocessing:** StandardScaler
- **Exchange integration:** Bybit WebSocket API (v5)

---

## References

1. Ho T., Stoll H. *On Dealer Markets Under Competition.* Journal of Finance, 1980.
2. Avellaneda M., Stoikov S. *High-Frequency Trading in a Limit Order Book.* Quantitative Finance, 2008.
3. Guéant O., Lehalle C.-A., Fernández-Tapia J. *Dealing with the Inventory Risk: A Solution to the Market-Making Problem.* 2012. arXiv:1205.5493
4. Cartea Á., Jaimungal S., Ricci J. *Buy Low, Sell High: A High-Frequency Trading Perspective.* SIAM J. Financial Mathematics 5(1), 2014, pp. 415–444.
5. Guéant O. *Optimal Market Making.* Wiley, 2017. arXiv:1605.01862
6. Aydoğan E., Uğur S., Aksoy M. *Optimal Limit Order Book Trading Strategies with Stochastic Volatility in the Underlying Asset.* Computational Economics 62:289–324, 2023.
7. Ntakaris A., Kanniainen J., Gabbouj M., Iosifidis A. *Mid-Price Prediction Based on Machine Learning Methods with Technical and Quantitative Indicators.*
8. Ntakaris A., Mirone G., Kanniainen J., Gabbouj M., Iosifidis A. *Feature Engineering for Mid-Price Prediction with Deep Learning.* arXiv:1904.05384
9. Bybit API documentation — https://bybit-exchange.github.io/docs/v5/intro

---

*Keywords: market-making, algorithmic trading, machine learning, deep learning, LSTM, limit order book, BTCUSDT, Avellaneda–Stoikov, Cartea–Jaimungal–Ricci.*
