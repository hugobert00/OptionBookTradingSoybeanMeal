# System Architecture — PricerLB Soymeal
**Version:** 2.0 | **Asset Class:** Agricultural Commodities — Soybean Meal CBOT (ZM) | **Model:** Black-76

---

## 1. Executive Overview

PricerLB is a single-dealer risk management and pricing platform for CBOT Soybean Meal (ZM) options and futures. It provides real-time Greeks aggregation, P&L attribution, implied volatility surface construction, and a multi-leg strategy pricer — all within a unified, session-aware dashboard.

The system is organized around a **thin orchestration layer** (`Cockpit.py`) that delegates all quantitative logic to four specialized modules. Data persistence relies on Excel workbooks, enabling lightweight deployment with no database infrastructure.

---

## 2. High-Level System Architecture

```mermaid
graph TB
    User(["Trader / Risk Manager\n(Browser — localhost:8501)"])

    subgraph App["PricerLB Application Layer  ·  Streamlit Runtime"]
        direction TB
        Cockpit["Cockpit.py\nOrchestration & Session State\nPage routing · Market input management"]

        subgraph Pages["Presentation Layer — 4 Modules"]
            P1["Main Book\nOpen Positions & Greeks"]
            P2["Closed Positions\nP&L Analytics & Reporting"]
            P3["Volatility Tools\nSurface & Smile Analysis"]
            P4["Strategy Pricer\nMulti-leg Builder"]
        end

        subgraph Engine["Quantitative Engine"]
            GM["GreeksManagement.py\nBlack-76 Pricing Engine\n──────────────────────\nFull Greeks · PnL Attribution\nDelta Hedge Advisor\nPortfolio Aggregation"]
            PL["PnLComputation.py\nRisk & Performance Analytics\n──────────────────────\nEquity Curve · Sharpe Ratio\nMax Drawdown · Win Rate\nYTD / Inception Attribution"]
            VOL["vol.py\nVolatility Surface Engine\n──────────────────────\nSmile Construction · Term Structure\nRisk Reversal · Butterfly\n3D Surface · Mispricing Map"]
            SM["SavingsManagement.py\nData Access Layer\n──────────────────────\nPositions I/O\nExcel Read / Write"]
        end
    end

    subgraph DataLayer["Persistence Layer — Excel Workbooks"]
        direction LR
        subgraph Books["books/  ·  Position Store"]
            B1["95135/\nbook.xlsx · closed_book.xlsx"]
            B2["95136/\nbook.xlsx · closed_book.xlsx"]
            B3["95137/\nbook.xlsx · closed_book.xlsx"]
        end
        subgraph VolStore["vol/  ·  Implied Volatility Store"]
            V1["VolFbarchart.xlsx  — Jan (F)"]
            V2["VolHbarchart.xlsx  — Mar (H)"]
            V3["VolKbarchart.xlsx  — May (K)"]
            V4["VolNbarchart.xlsx  — Jul (N)"]
            V5["VolQbarchart.xlsx  — Aug (Q)"]
            V6["VolUbarchart.xlsx  — Sep (U)"]
            V7["VolVbarchart.xlsx  — Oct (V)"]
            V8["VolZbarchart.xlsx  — Dec (Z)"]
        end
    end

    User -->|"HTTP / WebSocket"| Cockpit
    Cockpit --> P1 & P2 & P3 & P4

    P1 --> GM & SM
    P2 --> PL & SM
    P3 --> VOL
    P4 --> GM

    SM -->|"pandas read_excel / to_excel"| Books
    VOL -->|"pandas read_excel"| VolStore
```

---

## 3. Data Pipeline

```mermaid
flowchart LR
    subgraph Sources["External Data Sources"]
        XLS_POS["book.xlsx\nclosed_book.xlsx\nPosition Store"]
        XLS_VOL["Vol*.xlsx\nBarchart Cmdty\nImplied Vol Feed"]
    end

    subgraph Ingestion["Ingestion & Parsing"]
        IO["SavingsManagement\nread_excel · dtype enforcement\ndate parsing · account routing"]
        PARSE_VOL["vol.py\nColumn remapping\nStrike / Delta extraction\nExpiry tagging"]
    end

    subgraph Compute["Quantitative Processing"]
        DF_POS[("DataFrame\nOpen / Closed\nPositions")]
        DF_VOL[("DataFrame\nImplied Vol\nSurface Panel")]

        TTM["TTM Computation\nACT/365 · Exchange Expiry Dates"]
        BLACK76["Black-76 Pricing Engine\nper-instrument vectorized pricing"]
        GREEKS["Greeks Computation\nΔ Γ ν Θ ρ · Vanna · Volga · Charm"]
        EXPLAIN["P&L Attribution Engine\nΔ PnL · Γ PnL · ν PnL · Θ PnL · Residual"]
        SMILE["Smile Analytics\nATM IV · RR · BF · Skew · Curvature"]
        SURFACE["3D Surface Interpolation\nStrike × Expiry × IV"]
        MISPRICE["Mispricing Map\nIV(K) − IV_ATM  per expiry"]
    end

    subgraph Presentation["Streamlit UI Layer"]
        UI_BOOK["Main Book\nGreeks tables · Delta/Gamma/Vega/Theta charts\nHedge advisor · Live PnL"]
        UI_PNL["P&L Reports\nEquity curve · Monthly bars\nPerformance KPIs"]
        UI_VOL["Vol Tools\nSmile panel · Term structure\nSkew metrics · 3D surface"]
        UI_PRICER["Strategy Pricer\nMulti-leg builder · Per-leg Greeks"]
    end

    XLS_POS -->|"read_excel"| IO --> DF_POS
    XLS_VOL -->|"read_excel"| PARSE_VOL --> DF_VOL

    DF_POS --> TTM --> BLACK76
    DF_VOL -->|"σ(K, T)"| BLACK76

    BLACK76 --> GREEKS --> UI_BOOK
    BLACK76 --> EXPLAIN --> UI_BOOK
    GREEKS --> UI_PRICER

    DF_POS --> EXPLAIN
    DF_POS --> UI_PNL

    DF_VOL --> SMILE --> UI_VOL
    DF_VOL --> SURFACE --> UI_VOL
    DF_VOL --> MISPRICE --> UI_VOL
```

---

## 4. Pricing Model — Black-76

```mermaid
flowchart TD
    subgraph Inputs["Model Inputs"]
        IN["F  — Futures price (market)\nK  — Strike price\nT  — Time to maturity  [ACT/365]\nr  — Risk-free rate (default 2%)\nσ  — Implied volatility  [smile interpolated]\ntype — call | put | fut"]
    end

    subgraph Intermediates["Intermediate Computations"]
        D1["d₁ = [ ln(F/K) + ½σ²T ] / σ√T"]
        D2["d₂ = d₁ − σ√T"]
        DISC["Discount factor = e^(−rT)"]
    end

    subgraph Pricing["Option Price"]
        CALL_PRICE["Call = e^(−rT) · [ F·N(d₁) − K·N(d₂) ]"]
        PUT_PRICE["Put  = e^(−rT) · [ K·N(−d₂) − F·N(−d₁) ]"]
    end

    subgraph Greeks["First & Second Order Greeks"]
        G1["Δ (Delta)   = e^(−rT) · N(d₁)"]
        G2["Γ (Gamma)   = e^(−rT) · N'(d₁) / (F·σ·√T)"]
        G3["ν (Vega)    = F · e^(−rT) · N'(d₁) · √T"]
        G4["Θ (Theta)   = −[F·σ·e^(−rT)·N'(d₁)] / (2√T) − r·K·e^(−rT)·N(d₂)"]
        G5["ρ (Rho)     = −T · option_price"]
        G6["Vanna        = −e^(−rT) · N'(d₁) · d₂/σ"]
        G7["Volga        = F · e^(−rT) · N'(d₁) · √T · d₁·d₂/σ"]
        G8["Charm        = −e^(−rT) · N'(d₁) · [2rT − d₂·σ·√T] / (2T·σ·√T)"]
    end

    subgraph PnLExplain["P&L Attribution (Taylor Expansion)"]
        PNL["PnL ≈  Δ·ΔF  +  ½Γ·ΔF²  +  ν·Δσ  +  Θ·Δt  +  residual"]
    end

    IN --> D1 & DISC
    D1 --> D2
    D1 & D2 & DISC --> CALL_PRICE & PUT_PRICE
    D1 & D2 & DISC --> G1 & G2 & G3 & G4 & G5 & G6 & G7 & G8
    G1 & G2 & G3 & G4 --> PNL
```

---

## 5. Session State & Reactivity Model

```mermaid
flowchart TD
    subgraph Init["Initialization (first load or account change)"]
        LOAD["load_open_positions(account_id)"]
        TTM_C["add_ttm_column(valuation_date)"]
        VOL_MAP["df['vol'] = expiry.map(DEFAULT_VOL)"]
        MULT["df['contract_multiplier'] = 100"]
        BUILD["build_greeks_dataframe(F_market, r=0.02)"]
    end

    subgraph State["st.session_state  ·  In-Memory Cache"]
        SS1["positions  — enriched DataFrame with Greeks"]
        SS2["F_market   — {expiry: futures_price}"]
        SS3["current_account  — active account id"]
        SS4["valuation_date   — pricing date"]
    end

    subgraph Reactivity["Reactivity Triggers"]
        T1["Account selector change"]
        T2["Valuation date change"]
        T3["Futures price slider update"]
    end

    subgraph Pages["Downstream Consumers"]
        P1["Main Book — Greeks, Hedge, PnL"]
        P4["Strategy Pricer — per-leg pricing"]
    end

    LOAD --> TTM_C --> VOL_MAP --> MULT --> BUILD
    BUILD --> SS1
    T1 & T2 --> LOAD
    T3 --> SS2

    SS1 & SS2 --> P1 & P4
    SS3 --> T1
    SS4 --> T2
```

---

## 6. Volatility Surface Construction

```mermaid
flowchart LR
    subgraph Raw["Raw Data — per expiry file"]
        FILE["Vol{X}barchart.xlsx\nDelta_Call · ImplV_Call · Strike\nImplV_Put · Delta_Put"]
    end

    subgraph Parse["Parsing & Normalization"]
        REMAP["Column remapping\n(Unnamed: x → semantic names)"]
        ATAG["Expiry tag extraction\nfrom filename → F/H/K/N/Q/U/V/Z"]
        FMAP["ATM forward injection\nF_market[expiry]"]
    end

    subgraph Panel["Unified Vol Panel"]
        DF["panel DataFrame\n[expiry · strike · delta_call · iv_call · iv_put · delta_put]"]
    end

    subgraph Analytics["Surface Analytics"]
        ATM["ATM IV — interpolated at F"]
        RR["Risk Reversal\nRR_25 = IV_25P − IV_25C\nRR_10 = IV_10P − IV_10C"]
        BF["Butterfly\nBF_25 = ½(IV_25C + IV_25P) − ATM\nBF_10 = ½(IV_10C + IV_10P) − ATM"]
        SLOPE["Smile Slopes\nslope_left · slope_right"]
        CURV["Curvature\nBF proxy at selected Δ"]
        SURF["3D Interpolation\nStrike × T axis — Plotly mesh"]
        HEAT["Mispricing Heatmap\nIV(K) − ATM_IV  per expiry"]
    end

    FILE --> REMAP --> ATAG --> FMAP --> DF
    DF --> ATM & RR & BF & SLOPE & CURV & SURF & HEAT
```

---

## 7. Contract & Expiry Reference

| Code | Month     | Termination Rule                               |
|------|-----------|------------------------------------------------|
| F    | January   | Business day prior to the 15th of the month   |
| H    | March     | Business day prior to the 15th of the month   |
| K    | May       | Business day prior to the 15th of the month   |
| N    | July      | Business day prior to the 15th of the month   |
| Q    | August    | Business day prior to the 15th of the month   |
| U    | September | Business day prior to the 15th of the month   |
| V    | October   | Business day prior to the 15th of the month   |
| Z    | December  | Business day prior to the 15th of the month   |

**Contract specs:** ZM · CBOT · 100 short tons · USD/short ton · Tick 0.10 · Tick value USD 10.00

---

## 8. Module Responsibility Matrix

| Module | Responsibility | Key Functions |
|---|---|---|
| `Cockpit.py` | Orchestration, routing, session state, UI layout | Page dispatch · `st.session_state` management · market input widgets |
| `GreeksManagement.py` | Black-76 pricing, Greeks, hedge advisor, PnL explain | `build_greeks_dataframe` · `portfolio_delta_by_expiry` · `compute_pnl_explain` · `delta_hedge_action_by_expiry` |
| `PnLComputation.py` | Realized P&L, performance metrics | `build_daily_pnl_series` · `compute_sharpe` · `compute_max_drawdown` · `compute_pnl_by_year` |
| `vol.py` | Vol surface construction, smile analytics | `build_smile_panel_from_excels` · `compute_skew_metrics` · `plot_vol_surface` · `compute_vol_mispricing_map` |
| `SavingsManagement.py` | Excel I/O, account-level data access | `load_open_positions` · `save_open_positions` · `load_closed_positions` |

---

## 9. Account & Portfolio Structure

```mermaid
graph LR
    subgraph Accounts["Managed Accounts"]
        A1["95135"]
        A2["95136"]
        A3["95137"]
    end

    subgraph Instruments["Instrument Types"]
        FUT["fut — Futures"]
        CALL["c — Call Options"]
        PUT["p — Put Options"]
    end

    subgraph Expiries["Active Expiry Cycle  ·  CBOT ZM"]
        E1["F  Jan"]
        E2["H  Mar"]
        E3["K  May"]
        E4["N  Jul"]
        E5["Q  Aug"]
        E6["U  Sep"]
        E7["V  Oct"]
        E8["Z  Dec"]
    end

    A1 & A2 & A3 --> FUT & CALL & PUT
    FUT & CALL & PUT --> E1 & E2 & E3 & E4 & E5 & E6 & E7 & E8
```

---

## 10. Known Constraints & Design Decisions

| Area | Decision | Rationale |
|---|---|---|
| Persistence | Excel workbooks (no database) | Zero-infrastructure deployment · audit-friendly flat files |
| Pricing model | Black-76 with flat vol per expiry (default) | Industry standard for commodity options · smile loaded from Barchart when available |
| Vol interpolation | Linear interpolation across strikes within each expiry | Sufficient for risk monitoring · no arbitrage constraints not enforced |
| Session state | `st.session_state` as in-memory cache | Avoids recomputing Greeks on every widget interaction |
| Deployment | `streamlit run Cockpit.py` — single process | Internal single-user tool · no concurrency requirements |
| Vol source | Barchart Cmdty manual export (`.xlsx`) | No live API — vol data requires manual refresh |
| FX exposure | USD-denominated · FX P&L not yet computed | Planned feature — flagged in UI |

---

*Source: `Cockpit.py` · `GreeksManagement.py` · `PnLComputation.py` · `vol.py` · `SavingsManagement.py`*
