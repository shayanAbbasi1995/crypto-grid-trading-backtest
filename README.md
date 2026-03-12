# Crypto Grid Trading Backtest

A Python framework for back-testing a grid trading strategy on crypto futures markets, with performance analytics and risk reporting.

## What it does

Given an exported trade history (e.g., from Binance), the framework:

- Reconstructs position holding periods from raw BUY/SELL transactions
- Computes per-trade features: gross price/quantity/amount differences, net returns after fees, and holding time
- Aggregates grid-level performance using the geometric mean of individual trade returns
- Reports cumulative return, daily/annualized return, win/loss/draw counts, and VaR (95 & 99)
- Plots PnL distributions with optional VaR overlays

## Structure

| Class | Role |
|---|---|
| `DataMaker` | Load and preprocess the exported trade history |
| `Portfolio` | Hold configuration (initial capital, leverage, trading pair) |
| `FeatureMaker` | Compute per-trade and per-grid features |
| `ReportMaker` | Generate performance statistics and PnL charts |

## Usage

Place your exported trade history at `./Export Trade History.xlsx`, then run `BT_perf.py`.
Results are saved to `./Features Dataframes/` and plots to `./Plots/PnL Distributions/`.

## Included documentation

- **Back test Work Flow.pdf** — methodology and workflow overview
- **Grid Strategy - Initialization and Implication.pdf** — strategy specification and parameter rationale
