#=====================
#Cockpit of the dashboard 
#Here is the main cockpit of the dashboard, here is the page in which you can manage the page's organisation and display 
#All the functions and dependencies are available in the joined file and precisely called in this part. 
#The book dash board is made the simpliest way as possible in order to keep over time
# Disclaimer : The chosen way to build the dashboard is using Streamlit, it seems to be consistent over time, 
#however it is possible that new updates on streamlit mess with this beutiful system 
#You can find all the information needed for maintenance on : https://streamlit.io
#For any questions you can email me at (try to answer the fast as I can): hugo.berthelier@edhec.com
#=====================

import streamlit as st
import pandas as pd
import numpy as np
import requests as r
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta, date 
import math
from pathlib import Path

#functions imports 
from SavingsManagement import load_open_positions, save_open_positions
from GreeksManagement import (add_ttm_column, 
                              build_greeks_dataframe, 
                              portfolio_delta_by_expiry, 
                              plot_delta_vs_future_subplots, 
                              hedge_cash_cost_by_expiry, 
                              delta_hedge_action_by_expiry, 
                              portfolio_gamma_by_expiry, 
                              plot_gamma_vs_future_subplots, 
                              portfolio_vega_by_expiry, 
                              plot_vega_vs_future_subplots, 
                              portfolio_theta_by_expiry, 
                              plot_theta_vs_future_subplots, 
                              compute_live_pnl, 
                              compute_pnl_explain)

from PnLComputation import (
    load_closed_positions,
    compute_line_pnl,
    build_daily_pnl_series,
    compute_pnl_by_month,
    compute_pnl_by_expiry,
    compute_closed_pnl,
    compute_pnl_by_year,
    build_daily_pnl_series_by_year,
    compute_returns,
    compute_max_drawdown,
    #compute_cash_burn,
    compute_sharpe,
)

from vol import (
    build_smile_panel_from_excels,
    plot_smiles_panel,
    plot_vol_surface,
    compute_atm_term_structure,
    plot_atm_term_structure,
    compute_skew_metrics,
    plot_skew_bars,
    compute_smile_shape_metrics,
    plot_shape_lines,
    compute_vol_mispricing_map,
    plot_vol_mispricing_heatmap,
)



soymeal_contract_info = {
    "Soybean Meal CBOT": {
        "product_code":"ZM",
        "contract_size": 100,
        "unit": "USD/short t",
        "currency": "USD",
        "tick_size": 0.10,
        "tick_value": 10.00,
        "expiration_months":["F","H","K","N","Q","U","V","Z"],
        "termination": "Business day prior to the 15th day of the contract month"
    }
}

accounts = {
    "95135":{
        "Client_id" : 95135,
        "Client_name": "nom1"
    },
    "95136":{
        "Client_id" : 95136,
        "Client_name": "nom2"
    },
    "95137":{
        "Client_id" : 95137,
        "Client_name": "nom2"
    }
}


DEFAULT_VOL = {
    "F27": 0.20,
    "H26": 0.20,
    "K26": 0.20,
    "N26": 0.21,
    "Q26": 0.21,
    "U26": 0.22,
    "V26": 0.22,
    "Z26": 0.22

}

BASE_PATH_3 = Path("/Users/hugoberthelier/Desktop/PricerLB_Soymeal/vol/")

FILENAMES = [
    "VolFbarchart.xlsx",
    "VolHbarchart.xlsx",
    "VolKbarchart.xlsx",
    "VolNbarchart.xlsx",
    "VolQbarchart.xlsx",
    "VolUbarchart.xlsx",
    "VolVbarchart.xlsx",
    "VolZbarchart.xlsx",
]

COLUMN_MAPPING = {
    "Unnamed: 0": "Delta_Call",
    "Unnamed: 1": "ImplV_Call",
    "Unnamed: 6": "Strike",
    "Unnamed: 11": "ImplV_Put",
    "Unnamed: 12": "Delta_Put",
}

builded_strategies = {
    "None (Manual Input)": [{"Type": "Call", "Quantity": 1, "Strike/fut price": 330}],
    "Call Spread": [{"Type": "Call", "Quantity": 1, "Strike/fut price": 330},
                     {"Type": "Call", "Quantity": -1, "Strike/fut price": 340}],
    "Put Spread": [{"Type": "Put", "Quantity": 1, "Strike/fut price": 330},
                    {"Type": "Put", "Quantity": -1, "Strike/fut price": 320}],
    "Short Call Spread":[{"Type": "Call", "Quantity": -1, "Strike/fut price": 330},
                     {"Type": "Call", "Quantity": 1, "Strike/fut price": 340}],
    "Short Put Spread":[{"Type": "Put", "Quantity": -1, "Strike/fut price": 330},
                    {"Type": "Put", "Quantity": 1, "Strike/fut price": 320}],
    "Covered Write": [{"Type": "fut", "Quantity": 1, "Strike/fut price": 330},
                       {"Type": "Call", "Quantity": -1, "Strike/fut price": 330}],
    "Protective Put": [{"Type": "fut", "Quantity": 1, "Strike/fut price": 330},
                        {"Type": "Put", "Quantity": 1, "Strike/fut price": 330}],
    "Long Straddle": [{"Type": "Call", "Quantity": 1, "Strike/fut price": 330},
                       {"Type": "Put", "Quantity": 1, "Strike/fut price": 330}],
    "Long Strangle": [{"Type": "Call", "Quantity": 1, "Strike/fut price": 340},
                       {"Type": "Put", "Quantity": 1, "Strike/fut price": 330}],
    "Short Straddle": [{"Type": "Call", "Quantity": -1, "Strike/fut price": 330},
                        {"Type": "Put", "Quantity": -1, "Strike/fut price": 330}],
    "Short Strangle": [{"Type": "Call", "Quantity": -1, "Strike/fut price": 340},
                       {"Type": "Put", "Quantity": -1, "Strike/fut price": 330}],
    "Long Call Butterfly": [{"Type": "Call", "Quantity": 1, "Strike/fut price": 320},
                             {"Type": "Call", "Quantity": -2, "Strike/fut price": 330},
                             {"Type": "Call", "Quantity": 1, "Strike/fut price": 340}],
    "Short Call Butterfly": [{"Type": "Call", "Quantity": -1, "Strike/fut price": 320},
                              {"Type": "Call", "Quantity": 2, "Strike/fut price": 330},
                              {"Type": "Call", "Quantity": -1, "Strike/fut price": 340}],
}




#--------------------------------------
#Dashboard construction 
#--------------------------------------
st.sidebar.image('logo_feedalliance_couleur1.png')
st.sidebar.title("Menu")

Menu =st.sidebar.selectbox("Which page would you access to ?", ('Main Book', 'Volatility tools', 'Pricer - Strategy', 'Closed Positions & PNL Reports'))
st.header(Menu)

#Cockpit page 
if Menu == 'Main Book':
    selected_account = st.sidebar.selectbox("Select an account", list(accounts.keys()))
    st.write(f'Welcome on your Soymeal Book Laurent!')
    st.subheader(f'Soymeal Book overview - Opened positions - Acc:{selected_account}')
    st.sidebar.subheader("Market inputs")
    valuation_date = st.sidebar.date_input("Valuation date",value=date.today())
    
    if "F_market" not in st.session_state:
        st.session_state["F_market"] = {
            "F27": 320.00,
            "H26": 325.00,
            "K26": 328.00,
            "N26": 330.00,
            "Q26": 325.00,
            "U26": 320.00,
            "V26": 320.00,
            "Z26": 325.00
        }

    for expiry in st.session_state["F_market"]:
        st.session_state["F_market"][expiry] = st.sidebar.number_input(
            f"Future price {expiry}",
            value=st.session_state["F_market"][expiry],
            step=1.0
        )

    if (
    "positions" not in st.session_state
    or st.session_state.get("current_account") != selected_account
    or st.session_state.get("valuation_date") != valuation_date
    ):

        df = load_open_positions(selected_account)
        
        
        df = add_ttm_column(df, today=valuation_date)
        df["vol"] = df["expiry"].map(DEFAULT_VOL)
        df["contract_multiplier"] = 100

        st.write(df)
        df = build_greeks_dataframe(
            df,
            F_market=st.session_state["F_market"],
            r=0.02
        )
        st.write("Greeks :")
        st.session_state["positions"] = df

        

        st.dataframe(
            df[
                [
                    "date", "type", "strike", "expiry", "quantity",
                    "delta", "gamma", "vega", "theta",
                    "rho", "vanna", "volga", "charm"
                ]
            ],
            width= 'stretch'
        )

    with st.expander("📊 Delta by maturity", expanded=True):

        df_positions = st.session_state["positions"]

        st.subheader("Delta by maturity")

        delta_by_expiry = portfolio_delta_by_expiry(df_positions)

        st.dataframe(
            delta_by_expiry,
            width='stretch'
        )

        # ----------------------------
        st.subheader("Delta vs future price")

        expiries = delta_by_expiry["expiry"].tolist()

        fig = plot_delta_vs_future_subplots(
            df=df_positions,
            expiries=expiries,
            F_market=st.session_state["F_market"],
            F_range_pct=0.10,
            price_steps=40
        )

        st.plotly_chart(fig, width='stretch')


        # ----------------------------
        st.subheader("Delta hedge actions")

        hedge_actions = delta_hedge_action_by_expiry(delta_by_expiry)

        st.dataframe(
            hedge_actions,
            width='stretch'
        )

        # ----------------------------
        st.subheader("Hedge cost")

        hedge_costs = hedge_cash_cost_by_expiry(
            hedge_actions_df=hedge_actions,
            F_market=st.session_state["F_market"],
            contract_multiplier=100
        )

        st.dataframe(
            hedge_costs,
            width='stretch'
        )

        # ----------------------------
        total_delta = delta_by_expiry["delta_expiry"].sum()
        total_notional = hedge_costs["hedge_notional"].abs().sum()

        col1, col2 = st.columns(2)
        col1.metric("Total Delta (contracts)", f"{total_delta:,.1f}")
        col2.metric("Total Hedge Notional (EUR)", f"{total_notional:,.0f}")

    with st.expander("📊 Gamma by maturity", expanded=False):

        df_positions = st.session_state["positions"]

        st.subheader("Gamma by maturity")

        gamma_by_expiry = portfolio_gamma_by_expiry(df_positions)

        st.dataframe(
            gamma_by_expiry,
            width='stretch'
        )

        # ----------------------------
        st.subheader("Gamma vs future price")

        expiries = gamma_by_expiry["expiry"].tolist()

        fig = plot_gamma_vs_future_subplots(
            df=df_positions,
            expiries=expiries,
            F_market=st.session_state["F_market"],
            F_range_pct=0.10,
            price_steps=40
        )

        st.plotly_chart(fig, width='stretch')

        # ----------------------------
        total_gamma = gamma_by_expiry["gamma_expiry"].sum()

        st.metric(
            "Total Gamma (contracts / price unit)",
            f"{total_gamma:,.6f}"
        )

    with st.expander("📊 Vega by maturity", expanded=False):

        df_positions = st.session_state["positions"]

        st.subheader("Vega by maturity")

        vega_by_expiry = portfolio_vega_by_expiry(df_positions)

        st.dataframe(
            vega_by_expiry,
            width='stretch'
        )

        # ----------------------------
        st.subheader("Vega vs future price")

        expiries = vega_by_expiry["expiry"].tolist()

        fig = plot_vega_vs_future_subplots(
            df=df_positions,
            expiries=expiries,
            F_market=st.session_state["F_market"],
            F_range_pct=0.10,
            price_steps=40
        )

        st.plotly_chart(fig, width='content')

        # ----------------------------
        total_vega = vega_by_expiry["vega_expiry"].sum()

        st.metric(
            "Total Vega (contracts / vol point)",
            f"{total_vega:,.1f}"
        )

    with st.expander("⏳ Theta by maturity", expanded=False):

            df_positions = st.session_state["positions"]

            st.subheader("Theta by maturity (EUR / day)")

            theta_by_expiry = portfolio_theta_by_expiry(df_positions)

            st.dataframe(
                theta_by_expiry,
                width='stretch'
            )

            # ----------------------------
            st.subheader("Theta vs future price")

            expiries = theta_by_expiry["expiry"].tolist()

            fig = plot_theta_vs_future_subplots(
                df=df_positions,
                expiries=expiries,
                F_market=st.session_state["F_market"],
                F_range_pct=0.10,
                price_steps=40
            )

            st.plotly_chart(fig, width='stretch')

            # ----------------------------
            total_theta = theta_by_expiry["theta_expiry"].sum()

            st.metric(
                "Total Theta (EUR / day)",
                f"{total_theta:,.0f}"
            )

    with st.expander("💰 Live PnL", expanded=True):

            df_positions = st.session_state["positions"]

            # --- Reference markets (yesterday / entry)
            F_ref = {k: v for k, v in st.session_state["F_market"].items()}
            vol_ref = {exp: DEFAULT_VOL[exp] for exp in F_ref}

            # --- Live PnL
            df_positions = compute_live_pnl(
                df_positions,
                F_market=st.session_state["F_market"]
            )

            # --- PnL Explain
            df_positions = compute_pnl_explain(
                df_positions,
                F_market=st.session_state["F_market"],
                F_ref=F_ref,
                vol_ref=vol_ref
            )

            # ----------------------------
            st.subheader("PnL by position")

            st.dataframe(
                df_positions[
                    [
                        "date", "type", "expiry", "quantity",
                        "live_pnl"
                    ]
                ],
                width='stretch'
            )

            # ----------------------------
            st.subheader("PnL Explain")

            pnl_explain_cols = [
                "pnl_delta",
                "pnl_gamma",
                "pnl_vega",
                "pnl_theta",
                "pnl_residual"
            ]

            totals = df_positions[pnl_explain_cols].sum()

            col1, col2, col3, col4, col5 = st.columns(5)

            col1.metric("Δ PnL", f"{totals['pnl_delta']:,.0f} €")
            col2.metric("Γ PnL", f"{totals['pnl_gamma']:,.0f} €")
            col3.metric("Vega PnL", f"{totals['pnl_vega']:,.0f} €")
            col4.metric("Theta PnL", f"{totals['pnl_theta']:,.0f} €")
            col5.metric("Residual", f"{totals['pnl_residual']:,.0f} €")

            # ----------------------------
            st.subheader("Total Live PnL")

            total_pnl = df_positions["live_pnl"].sum()

            st.metric(
                "Total Live PnL (EUR)",
                f"{total_pnl:,.0f}"
            )

    #Volume interest on each strike 
    #bar plot that shows the volumes of calls and puts present on each strike 



    st.markdown("### Contract specifications")
    st.markdown("""
    - **Product**: ZM – Soybean Meal
    - **Multiplier**: 100
    - **Currency**: USD / short ton
    - **Model**: Black-76
    - **TTM**: real exchange expiry dates
    """)

#Close Position & PNL Reports page
if Menu == 'Closed Positions & PNL Reports':
    selected_account = st.sidebar.selectbox("Select an account", list(accounts.keys()))
    st.write(f'Welcome on your Rapseed PnL report Laurent! You have selected {selected_account}.')
    if (
    "positions" not in st.session_state
    or st.session_state.get("current_account") != selected_account
    ):

        df = load_closed_positions(selected_account)
        df = compute_line_pnl(df)
        if df.empty : 
             st.warning("No closed positions available for this account.")
             st.stop()

        # ===============================
        # Daily / Monthly aggregates
        # ===============================
        daily_pnl = build_daily_pnl_series(df)
        monthly_pnl = compute_pnl_by_month(df)
        pnl_by_expiry = compute_pnl_by_expiry(df)
        summary = compute_closed_pnl(df)

        # ===============================
        # Yearly / Expiries aggregates
        # ===============================


        pnl_by_year = compute_pnl_by_year(df)
        daily_pnl_ytd = build_daily_pnl_series_by_year(df)



        # ===============================
        # Equity curve & risk metrics
        # ===============================
        equity_curve = daily_pnl.set_index("date")["cum_pnl"]

        returns = compute_returns(equity_curve)
        max_dd = compute_max_drawdown(equity_curve)
        #cash_burn = compute_cash_burn(monthly_pnl)
        sharpe = compute_sharpe(returns)




    # ===============================
    # Yearly PnL evolution
    # ===============================
    st.subheader("PnL overview")

    tab_global, tab_yearly = st.tabs(["Global (since inception)", "By year"])

    with tab_global:
        st.write("Cumulative realized PnL since inception (closed trades).")
        st.line_chart(daily_pnl.set_index("date")[["cum_pnl"]], height=350)

    with tab_yearly:
        st.write("PnL by calendar year and YTD equity curve (reset each Jan 1st).")

        st.bar_chart(pnl_by_year.set_index("year")[["pnl"]], height=250)

        years = [int(y) for y in pnl_by_year["year"].dropna().unique()]
        years = sorted(years)
        if years:
            selected_year = st.selectbox("Select a year", years, index=len(years)-1)

            ytd = daily_pnl_ytd[daily_pnl_ytd["year"] == selected_year].copy()
            if not ytd.empty:
                ytd_series = ytd.set_index("date")[["cum_pnl_ytd"]]
                st.line_chart(ytd_series, height=300)
            else:
                st.info("No daily data for selected year.")
        else:
            st.info("No yearly data available.")


    st.subheader("Monthly PnL : ")
    #here we put the bar chart of the PnL of each month 
    st.bar_chart(
        monthly_pnl.set_index("month")[["pnl"]],
        height=300
    )

    st.subheader("PnL by expiry")
    st.bar_chart(
        pnl_by_expiry.set_index("expiry")[["pnl"]],
        height=300
    )

    with st.expander("PnL by expiry - details", expanded=False):
        st.dataframe(pnl_by_expiry, width='stretch')


    


    with st.expander("Closed positions", expanded=False):
         st.subheader("Closed position details : ")
         st.write(df)


    # ===============================
    # Performance metrics
    # ===============================
    st.subheader("Performance metrics")

    # Last year available
    last_year = int(pnl_by_year["year"].dropna().max()) if not pnl_by_year.empty else None

    # ---------- Ligne 1 : Global ----------
    col1, col2, col3 = st.columns(3)

    if last_year is not None:
        pnl_last_year = float(
            pnl_by_year.loc[pnl_by_year["year"] == last_year, "pnl"].iloc[0]
        )
        n_last_year = int(
            pnl_by_year.loc[pnl_by_year["year"] == last_year, "n_trades"].iloc[0]
        )

        col1.metric(
            f"Total PnL {last_year} (USD)",
            f"{pnl_last_year:,.0f}"
        )
    else:
        col1.metric(
            "PnL (Year)",
            "n/a"
        )

    col2.metric(
        "Number of trades",
        f"{summary['n_trades']}"
    )

    col3.metric(
        "Win rate",
        f"{summary['win_rate'] * 100:.1f} %"
    )

    # ---------- Ligne 2 : Risk / performance ----------
    col4, col5, col6 = st.columns(3)

    col4.metric(
        "Max drawdown",
        f"{max_dd * 100:.2f} %"
    )

    col5.metric(
        "Sharpe ratio",
        f"{sharpe:.2f}"
    )

    col6.metric(
        "Total Cumulated PnL (USD)",
         f"{summary['total_pnl']:,.0f}"
     )

    # ===============================
    # Extra info
    # ===============================
    st.write("PnL on FX Exposure: ** EXPOSURE** (Soybean traded in USD implies an FX Exposure for you - avilable soon)")

#Volatility page tools 
if Menu == 'Volatility tools':
    valuation_date = st.sidebar.date_input("Valuation date", value=date.today())

    st.subheader("Volatility smile and skewness")
    st.write("Volatility smile for Soybean meal for all available expiries")

    # ----------------------------
    # Controls
    # ----------------------------
    x_mode_ui = st.selectbox(
        "Choose sticky strike or sticky delta",
        options=["Strike", "Delta (signé)", "Delta (absolu)"],
        index=0
    )
    x_mode = {
        "Strike": "strike",
        "Delta (signé)": "delta_signed",
        "Delta (absolu)": "delta_abs"
    }[x_mode_ui]

    available_expiries = sorted(st.session_state["F_market"].keys())
    selected_expiries = st.multiselect(
        "Expiries available",
        options=available_expiries,
        default=available_expiries
    )

    # ----------------------------
    # Load vols
    # ----------------------------
    panel = build_smile_panel_from_excels(
        base_path=BASE_PATH_3,
        filenames=FILENAMES,
        column_mapping=COLUMN_MAPPING,
        F_market=st.session_state["F_market"],
        year_suffix="26"
    )

    if panel.empty:
        st.warning("No implied vol data loaded (missing Excel files or expiry mapping).")
        st.stop()

    # Filtre panel sur expiries sélectionnées (important pour les métriques)
    panel_sel = panel[panel["expiry"].isin(selected_expiries)].copy()
    if panel_sel.empty:
        st.warning("No data for selected expiries.")
        st.stop()

    # ----------------------------
    # Smile plot
    # ----------------------------
    fig_smile = plot_smiles_panel(
        panel=panel_sel,
        x_mode=x_mode,
        expiries=selected_expiries,
        title="Implied vol smiles"
    )
    st.plotly_chart(fig_smile, width='stretch')

    with st.expander("Implied vol data", expanded=False):
        st.dataframe(panel_sel, width='stretch')

    st.divider()

    # ============================================================
    # (1) ATM term structure
    # ============================================================
    st.subheader("ATM Vol Term Structure")
    df_ts = compute_atm_term_structure(panel_sel, st.session_state["F_market"])
    st.plotly_chart(plot_atm_term_structure(df_ts, title="ATM IV term structure"), width='stretch')
    with st.expander("ATM term structure table", expanded=False):
        st.dataframe(df_ts, width='stretch')

    st.divider()

    # ============================================================
    # (2) Skew metrics: RR / BF / PutSkew for 25Δ and 10Δ
    # ============================================================
    st.subheader("Skew Metrics (10Δ / 25Δ)")

    df_skew = compute_skew_metrics(panel_sel, st.session_state["F_market"], deltas=(0.10, 0.25))
    with st.expander("Skew metrics table", expanded=False):
        st.dataframe(df_skew, width='stretch')

        c1, c2, c3 = st.columns(3)
        with c1:
            st.plotly_chart(plot_skew_bars(df_skew, metric="rr_25", title="25Δ Risk Reversal"), width='stretch')
        with c2:
            st.plotly_chart(plot_skew_bars(df_skew, metric="bf_25", title="25Δ Butterfly"), width='stretch')
        with c3:
            st.plotly_chart(plot_skew_bars(df_skew, metric="putskew_25", title="25Δ Put Skew (IV25P - ATM)"), width='stretch')

        st.divider()

    # ============================================================
    # (3) Smile slope & curvature (based on 25Δ by default)
    # ============================================================
    st.subheader("Smile Shape (Slopes & Curvature)")

    shape_delta = st.selectbox("Delta used for shape metrics", options=[0.10, 0.25], index=0)
    df_shape = compute_smile_shape_metrics(df_skew, d=float(shape_delta))

    k = int(round(float(shape_delta) * 100))
    cols_slopes = [f"slope_left_{k}", f"slope_right_{k}"]
    cols_curve = [f"curvature_{k}"]

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            plot_shape_lines(df_shape, cols=cols_slopes, title=f"Smile slopes ({k}Δ)"),
            width='stretch'
        )
    with c2:
        st.plotly_chart(
            plot_shape_lines(df_shape, cols=cols_curve, title=f"Smile curvature ({k}Δ)"),
            width='stretch'
        )

    with st.expander("Shape metrics table", expanded=False):
        st.dataframe(df_shape, width='stretch')

    st.divider()

    # ============================================================
    # 3D Surface
    # ============================================================
    st.subheader("3D Surface implied vol")
    fig_surface = plot_vol_surface(
        panel=panel_sel,
        valuation_date=valuation_date,
        expiries=selected_expiries,
        n_strikes=70,
        fill_across_expiries=True,
        title="Implied vol surface - Soybean Meal"
    )
    st.plotly_chart(fig_surface, width='stretch')

    st.divider()

    # ============================================================
    # (5) Vol Mispricing map vs ATM
    # ============================================================
    st.subheader("Vol Mispricing Map (IV - ATM)")
    df_map = compute_vol_mispricing_map(panel_sel, st.session_state["F_market"])
    fig_map = plot_vol_mispricing_heatmap(df_map, title="IV(K) - IV_ATM (vol points)")
    st.plotly_chart(fig_map, width='stretch')

    with st.expander("Mispricing map data", expanded=False):
        st.dataframe(df_map, width='stretch')

    st.divider()

    st.write("Disclaimer: all data are extracted from Barchart.com via Cmdty views.")
    st.write("Please manually verify implied vol data (source files in /PricerLB_Soymeal/vol).")


#--------------------------
#Pricer page building
#--------------------------


if Menu == 'Pricer - Strategy':

    st.subheader("Pricer - Build your strategy here - Soymeal")
    st.write("Welcome on your Soymeal pricer Laurent - Please enter your strategy below :")
    st.sidebar.subheader("Market inputs")
    valuation_date = st.sidebar.date_input("Valuation date",value=date.today())
    st.sidebar.write("Market prices for each expiry are taken from the Main book page.")
    st.sidebar.subheader("Pre-builded strategies : ")
    selected_strategy = st.sidebar.selectbox("Select a strategy", list(builded_strategies.keys()))
    st.sidebar.write("You can customize in the code your default strategies :)")
    st.sidebar.subheader("Pricing model used : ")
    st.sidebar.write("A Black-76 model is used with a Local Vol Model to capture skew dynamics.")


    


    st.write(" Add and delete components of your strategy here : ")
    #here the user can add a strategy 


    #here we display the payoff (Intrinsic value vs real value)


    #here we integrate a display of a data frame in a table form with all greeks 
    st.subheader("Details of the strategy : ")
    with st.expander("Greeks of the strategy :", expanded=True):
        st.write('Greeks data frame')

    

    #here we compute the cost of the full strategy added by the user 
    st.write("Cost of the full strategy : ")
    with st.expander("Cost of each legg :", expanded=False):
        st.write('cost of each component')

    




