# PricerLB — Soybean Meal (CBOT) Options Pricing & Risk Management Dashboard

A professional-grade Streamlit dashboard for managing Soybean Meal CBOT (ZM) futures and options positions. Built around the **Black-76 model**, it provides full Greeks computation, P&L attribution, volatility surface analysis, and a multi-leg strategy pricer.

> **Disclaimer:** All datasets included in this repository (positions, trades, account data) are **fictional and fully anonymized**. They do not represent any real trading activity, real accounts, or any real financial institution. They are provided solely for demonstration purposes.

---

## Features

### Open Positions & Greeks Dashboard
- Real-time Greeks computation: **Delta, Gamma, Vega, Theta, Rho, Vanna, Volga, Charm**
- Portfolio aggregation by expiry and instrument type
- Delta hedge advisor with optimal sizing and estimated cost
- Live P&L tracking with full attribution (Delta PnL, Gamma PnL, Vega PnL, Theta PnL)
- Market inputs for 8 expiries: F (Jan), H (Mar), K (May), N (Jul), Q (Aug), U (Sep), V (Oct), Z (Dec)

### P&L Analytics (Closed Positions)
- Realized P&L breakdown by trade, month, year, and expiry
- Daily equity curve and cumulative performance
- Key statistics: **Sharpe ratio, max drawdown, win rate**, trade count

### Volatility Tools
- Implied volatility smile visualization — by strike or by delta
- ATM vol term structure across expiries
- Skew metrics: **25Δ / 10Δ Risk Reversal (RR), Butterfly (BF), Put Skew**
- Smile shape analysis: slopes and curvature
- Interactive **3D volatility surface** (Plotly)
- Vol mispricing heatmap vs ATM baseline

### Strategy Pricer
- Multi-leg strategy builder interface
- Pre-configured strategies: spreads, straddles, strangles, butterflies, covered writes, protective puts
- Per-leg Greeks and cost breakdown

---

## Architecture

```
PricerLB/
├── Cockpit.py              # Main Streamlit app (4-page dashboard)
├── GreeksManagement.py     # Black-76 Greeks engine (~1,500 lines)
├── PnLComputation.py       # Realized P&L analytics
├── vol.py                  # Volatility surface & smile tools
├── SavingsManagement.py    # Excel I/O for position data
├── books/                  # Fictional position data (per account)
│   ├── {account_id}/
│   │   ├── book.xlsx       # Open positions
│   │   └── closed_book.xlsx# Closed/historical trades
└── vol/                    # Implied volatility data (by expiry)
    ├── VolFbarchart.xlsx   # January expiry (F)
    ├── VolHbarchart.xlsx   # March expiry (H)
    ├── VolKbarchart.xlsx   # May expiry (K)
    ├── VolNbarchart.xlsx   # July expiry (N)
    ├── VolQbarchart.xlsx   # August expiry (Q)
    ├── VolUbarchart.xlsx   # September expiry (U)
    ├── VolVbarchart.xlsx   # October expiry (V)
    └── VolZbarchart.xlsx   # December expiry (Z)
```

---

## Data Formats

**Open positions** (`book.xlsx`) — columns: `trade_id`, `date`, `underlying`, `type`, `expiry`, `lots`, `quantity`, `strike`, `price/premium`, `cost`, `units`

Instrument types: `fut` (futures), `c` (call), `p` (put)

**Closed positions** (`closed_book.xlsx`) — same schema with additional `open_date` and `end_price` columns.

**Implied vol files** (`vol/*.xlsx`) — pivot table by strike: `Delta_Call`, `ImplV_Call`, `Strike`, `ImplV_Put`, `Delta_Put`

---

## Contract Specifications

| Parameter | Value |
|-----------|-------|
| Underlying | Soybean Meal CBOT (ZM) |
| Pricing model | Black-76 |
| Contract multiplier | 100 USD / short ton |
| Tick size | 0.10 |
| Tick value | USD 10.00 |
| Currency | USD |
| Risk-free rate (default) | 2% |
| Day count (calendar) | ACT/365 |
| Day count (trading) | ACT/252 |

Expiry mapping: **F** = Jan · **H** = Mar · **K** = May · **N** = Jul · **Q** = Aug · **U** = Sep · **V** = Oct · **Z** = Dec
Termination: business day prior to the 15th day of the contract month

---

## Getting Started

### Prerequisites

```bash
pip install streamlit pandas numpy plotly matplotlib openpyxl
```

### Run the dashboard

```bash
streamlit run Cockpit.py
```

The app will open at `http://localhost:8501` in your browser.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI / App framework | [Streamlit](https://streamlit.io/) |
| Numerical computing | NumPy, pandas |
| Visualization | Plotly, Matplotlib |
| Options model | Black-76 |
| Data storage | Excel (.xlsx) via openpyxl |
| Language | Python 3.10+ |

---

## Disclaimer

The datasets included are **entirely fictional and anonymized** — no real client, account, or trade data is present in this repository. The tool is designed demonstration purpose.

---

## Author

Hugo Berthelier — EDHEC Business School x Centrale Lille 
[hugo.berthelier@edhec.com](mailto:hugo.berthelier@edhec.com)
