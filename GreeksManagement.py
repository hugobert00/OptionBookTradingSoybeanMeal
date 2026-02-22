#=====================
#Greeks Computation: 
#Here is the page in which we make all the greeks computation and displays 
#We do compute them using basic BS and the PDE to make sure we do have consistent results over time 
#The greeks we do compute are: 
#- Delta 
#- Gamma 
#- Theta 
#- Vega 
#- Rhô 
#- Vanna 
#- Volga 
#- Charm 

#We also add the management and the display of Vega buckets and Gamma buckets and delta hedging advisory 
#=====================

#----------------------------------------
#               IMPORTS
#----------------------------------------
import numpy as np
import pandas as pd
from math import erf, sqrt
from datetime import date
import matplotlib.pyplot as plt 
import plotly.graph_objects as go
from plotly.subplots import make_subplots

FUTURES_MONTH_MAP = {
    "F": 1,   # Jan
    "H": 3,   # Mar
    "K": 5,   # May
    "N": 7,   # Jul
    "Q": 8,   # Aug
    "U": 9,   # Sep
    "V": 10,  # Oct
    "Z": 12   # Dec
}
EXPIRY_DATES = {

    # ===== CME OFFICIAL CALENDAR =====
    2026: {1: 14, 3: 13, 5: 14, 7: 14, 8: 14, 9: 14, 10: 14, 12: 14},
    2027: {1: 14, 3: 12, 5: 14, 7: 14, 8: 13, 9: 14, 10: 14, 12: 14},
    2028: {1: 14, 3: 14, 5: 12, 7: 14, 8: 14, 9: 14, 10: 13, 12: 14},
    2029: {1: 12, 3: 14, 5: 14, 7: 13, 8: 14, 9: 14, 10: 12, 12: 14},

    # ===== RULEBOOK-CONSISTENT (CME methodology) =====
    2030: {1: 14, 3: 14, 5: 14, 7: 12, 8: 14, 9: 13, 10: 14, 12: 13},
    2031: {1: 14, 3: 14, 5: 14, 7: 14, 8: 14, 9: 12, 10: 14, 12: 12},
    2032: {1: 14, 3: 12, 5: 14, 7: 14, 8: 13, 9: 14, 10: 14, 12: 14},
    2033: {1: 14, 3: 14, 5: 13, 7: 14, 8: 12, 9: 14, 10: 14, 12: 14},
    2034: {1: 13, 3: 14, 5: 12, 7: 14, 8: 14, 9: 14, 10: 13, 12: 14}
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

#----------------------------------------
#               HELPERS
#----------------------------------------
def N(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def n(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


#Future contract 
def fut0(S, r, u, y, T):
    """
    S = spot price in the market 
    r = risk free rate 
    u = storage cost, by default of 0.02 here 
    y = convenience yield, by default of 0 here 
    T = time left to expiry 
    """
    u, y = 0.02, 0
    return (S*np.exp((r+u-y)*(T)))

def bs76_d1(F, K, v, T):
    """
    BS formula for options on future contracts 
    """
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    v = np.asarray(v, dtype=float)
    T = np.asarray(T, dtype=float)

    if np.any((F <= 0) | (K <= 0) | (v <= 0) | (T <= 0)):
        return np.nan

    return (np.log(F / K) + 0.5 * v**2 * T) / (v * np.sqrt(T))


def sort_moneyness(instrument_type, S, K):
    moneyness = np.log(S/K)
    return moneyness


def bs76_price(option_type, F, K, v, T, r):
    """
    Black-76 price (discounted).
    """
    d1 = bs76_d1(F, K, v, T)
    d2 = d1 - v * np.sqrt(T)

    if option_type.lower() == "c":
        price = np.exp(-r * T) * (F * N(d1) - K * N(d2))
    elif option_type.lower() == "p":
        price = np.exp(-r * T) * (K * N(-d2) - F * N(-d1))
    else:
        raise ValueError("Option type must be 'c' or 'p'")

    return price

def get_expiry_date(expiry_code):
    """
    get the expiry date from a future code ex: 'H26'.
    """
    if not isinstance(expiry_code, str) or len(expiry_code) != 3:
        raise ValueError(f"Expiry code invalide : {expiry_code}")

    month_code = expiry_code[0].upper()
    year = 2000 + int(expiry_code[1:])

    if month_code not in FUTURES_MONTH_MAP:
        raise ValueError(f"Mois future non supporté : {month_code}")

    month = FUTURES_MONTH_MAP[month_code]

    if year not in EXPIRY_DATES:
        raise ValueError(f"Année non disponible dans le calendrier : {year}")

    if month not in EXPIRY_DATES[year]:
        raise ValueError(f"Mois {month} non disponible pour {year}")

    day = EXPIRY_DATES[year][month]

    return date(year, month, day)

def get_ttm_from_today(expiry_code, day_count=365):
    """
    Time to maturity (in years)
    """
    today = date.today()
    expiry_date = get_expiry_date(expiry_code)

    delta_days = (expiry_date - today).days

    if delta_days <= 0:
        return 0.0

    return delta_days / day_count

def add_ttm_column(df, today=None):
    if today is None:
        today = date.today()
    else:
        today = pd.to_datetime(today).date()

    df = df.copy()
    df["T"] = df["expiry"].apply(
        lambda exp: max((get_expiry_date(exp) - today).days, 0) / 365
    )
    return df


#----------------------------------------
#               MARKET INFO 
#----------------------------------------
#here you have to modify those price every time you actualize your book 
#to get the right greeks and have ints on contango and backwardation 
F_market = {
            "F27": 320.00,
            "H26": 325.00,
            "K26": 328.00,
            "N26": 330.00,
            "Q26": 325.00,
            "U26": 320.00,
            "V26": 320.00,
            "Z26": 325.00
        }



#----------------------------------------
#               DELTA
#----------------------------------------
# Here we do the delta computation for each lign and for the full portfolio
# Single Delta 

def single_delta(instrument_type, expiry, F_market, K, v, T, qty, contract_multiplier=1.0):
    """
    We do compute the delta according to the type of instrument which is written by the user 
    - instrument type available are : 'fut' | 'call'| 'put'
    - the sens i.e. 'long' or 'short' is registered in the qty 
    - contract_multiplier : the notional value of the contract which is of 100 on ZM 
    cf : https://www.cmegroup.com/markets/agriculture/oilseeds/soybean-meal.quotes.html
    """
    itype = str(instrument_type).strip().lower()
    F = F_market[expiry] #for the given expiry we take the right price

    if itype == "f":
        delta = 1.0

    elif itype in ('c', 'p'):
        d1 = bs76_d1(F, K, v, T)
        if itype =='c':
            delta = N(d1)
        
        else:
            delta = N(d1)- 1.0

    else: 
        raise ValueError (f"Instrument inconnu : {instrument_type}")
    
    return delta * float(qty)

#Delta VS Spot 
def delta_vs_spot (delta_vs_future, r, u, y, T):
    carry = np.exp((r + u - y) * T)
    delta_vs_spot = delta_vs_future * carry
    return delta_vs_spot


#Computation of the delta of each ligne of the portfolio
def compute_line_deltas(
        df, 
        F_market,
        col_type="type",
        col_expiry="expiry",
        col_K="strike",
        col_v="vol",
        col_T="T",
        col_qty="quantity",
        col_multiplier="contract_multiplier" ):
    
    df.copy()
    
    """
    Computation of the Delta of each lign of the portfolio (vs.future) and we add a column delta
    """
    def delta_row (row):
        multiplier = row[col_multiplier] if col_multiplier in df.columns else 1.0
        return single_delta(
            instrument_type=row[col_type],
            expiry=row[col_expiry],
            F_market=F_market,
            K=row[col_K],
            v=row[col_v],
            T=row[col_T],
            qty=row[col_qty],
            contract_multiplier = multiplier
        )
    df = df.copy()
    df["delta"]= df.apply(delta_row, axis = 1)

    return df

#Deltas of the portfolio arranged by expiry 
def portfolio_delta_by_expiry(df, col_expiry="expiry", col_delta="delta"):
    """
    Here we do aggregate the portfolio's deltas on each expiry 
    """
    return (
        df
        .groupby(col_expiry, as_index=False)[col_delta]
        .sum()
        .rename(columns={col_delta: "delta_expiry"})
    )


#Hedging instruction to cancel directional exposure of the portfolio 
def delta_hedge_action_by_expiry(delta_by_expiry_df):
    """
    Gives to the user the size and the nature of the order he has to 
    send in the market in order to neutralize its delta exposure 
    One of the most important thing here is that we do spread the delta exposure 
    on maturities. 

    it returns a dict of action : {action: 'buy/sell/hold', qty:...}
    we only give here delta hedging using future contracts because of the habits
    of the user but we can also easily add suggestions using option contracts. 

    we retrun the following DataFrame : 
    epiry | action | quantity 

    """
    rows = []
    for _, row in delta_by_expiry_df.iterrows():
        dp = float(row["delta_expiry"])

        if np.isfinite(dp) and abs(dp)>0:
            qty_to_trade = -dp
            action = "buy" if qty_to_trade >0 else "sell"
            qty = abs(qty_to_trade)
        else:
            action = "hold"
            qty = 0.0

        rows.append({
            "expiry": row["expiry"],
            "action": action,
            "qty": qty
        })
    return pd.DataFrame(rows)
    
#computation of the cash cost of the hedge 
def hedge_cash_cost_by_expiry(hedge_actions_df, F_market, contract_multiplier):
    """
    F_market : dict {expiry : future_price}
    """
    rows = []
    for _, row in hedge_actions_df.iterrows():
        expiry = row["expiry"]
        qty = float(row["qty"])
        F = float(F_market[expiry])

        notional = qty *F* contract_multiplier
        rows.append({
            "expiry": expiry,
            "hedge_action": row["action"],
            "hedge_qty": qty,
            "future_price": F,
            "hedge_notional": notional
        })

    return pd.DataFrame(rows)


#Delta plots
#Delta VS future prices, we display on the same graph both delta 
def plot_delta_vs_future_subplots(
    df,
    expiries,
    F_market,
    F_range_pct=0.10,
    price_steps=40
):
    """
    Subplots Delta vs Future price, one subplot per expiry.
    
    df         : trade book avec colonne 'delta'
    expiries   : liste des expiries à tracer (ex: ["H26", "X26"])
    F_market   : dict {expiry: future_price}
    """
    n = len(expiries)

    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=False,
        subplot_titles=[f"Delta vs Future – {exp}" for exp in expiries]
    )

    for i, expiry in enumerate(expiries, start=1):
        F0 = F_market[expiry]

        prices = np.linspace(
            F0 * (1 - F_range_pct),
            F0 * (1 + F_range_pct),
            price_steps
        )

        deltas = []

        for F in prices:
            F_tmp = F_market.copy()
            F_tmp[expiry] = F

            df_tmp = compute_line_deltas(df, F_tmp)

            delta_expiry = (
                df_tmp[df_tmp["expiry"] == expiry]["delta"]
                .sum()
            )

            deltas.append(delta_expiry)

        fig.add_trace(
            go.Scatter(
                x=prices,
                y=deltas,
                mode="lines",
                name=f"{expiry}"
            ),
            row=i,
            col=1
        )

        
        fig.add_vline(
            x=F0,
            line_dash="dash",
            line_color="gray",
            row=i,
            col=1
        )

    fig.update_layout(
        height=300 * n,
        title="Delta du portefeuille vs prix du future (par échéance)",
        showlegend=False,
        template="plotly_white"
    )

    fig.update_xaxes(title_text="Prix du future")
    fig.update_yaxes(title_text="Delta (contrats)")

    return fig


#Delta by expiry represented with a barchart 



#----------------------------------------
#               GAMMA
#----------------------------------------
def single_gamma(
        instrument_type,
        expiry,
        F_market,
        K, 
        v, 
        T, 
        qty
):
    """
    Gamma vs futures in contract units 
    """
    itype = str(instrument_type).strip().lower()
    F = F_market[expiry]

    if itype == "f":
        gamma = 0.0

    elif itype in ("c", "p"):
        d1 = bs76_d1(F, K, v, T)
        gamma = n(d1) / (F * v * np.sqrt(T))

    else:
        raise ValueError(f"Instrument inconnu : {instrument_type}")

    return gamma * float(qty)



def compute_line_gammas(
    df,
    F_market,
    col_type="type",
    col_expiry="expiry",
    col_K="strike",
    col_v="vol",
    col_T="T",
    col_qty="quantity"
):
    """
    Gamma computation for each lign of the portfolio 
    """
    def gamma_row(row):
        return single_gamma(
            instrument_type=row[col_type],
            expiry=row[col_expiry],
            F_market=F_market,
            K=row[col_K],
            v=row[col_v],
            T=row[col_T],
            qty=row[col_qty]
        )

    df = df.copy()
    df["gamma"] = df.apply(gamma_row, axis=1)
    return df

def portfolio_gamma_by_expiry(
    df,
    col_expiry="expiry",
    col_gamma="gamma"
):
    """
    Gamma agregation for each expiry.
    """
    return (
        df
        .groupby(col_expiry, as_index=False)[col_gamma]
        .sum()
        .rename(columns={col_gamma: "gamma_expiry"})
    )

#diplay by maturity Gamma vs future 
def plot_gamma_vs_future_subplots(
    df,
    expiries,
    F_market,
    F_range_pct=0.10,
    price_steps=40
):
    """
    Subplots Gamma vs Future price, one subplot per expiry.
    
    df         : trade book
    expiries   : liste des expiries à tracer (ex: ["H26", "X26"])
    F_market   : dict {expiry: future_price}
    """
    n = len(expiries)

    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=False,
        subplot_titles=[f"Gamma vs Future – {exp}" for exp in expiries]
    )

    for i, expiry in enumerate(expiries, start=1):
        F0 = F_market[expiry]

        prices = np.linspace(
            F0 * (1 - F_range_pct),
            F0 * (1 + F_range_pct),
            price_steps
        )

        gammas = []

        for F in prices:
            F_tmp = F_market.copy()
            F_tmp[expiry] = F

            df_tmp = compute_line_gammas(df, F_tmp)

            gamma_expiry = (
                df_tmp[df_tmp["expiry"] == expiry]["gamma"]
                .sum()
            )

            gammas.append(gamma_expiry)

        fig.add_trace(
            go.Scatter(
                x=prices,
                y=gammas,
                mode="lines",
                name=f"{expiry}"
            ),
            row=i,
            col=1
        )

        
        fig.add_vline(
            x=F0,
            line_dash="dash",
            line_color="gray",
            row=i,
            col=1
        )

    fig.update_layout(
        height=300 * n,
        title="Gamma du portefeuille vs prix du future (par échéance)",
        showlegend=False,
        template="plotly_white"
    )

    fig.update_xaxes(title_text="Prix du future")
    fig.update_yaxes(title_text="Gamma (contrats / unité de prix)")

    return fig

#add the display of gamma buckets by maturitues and plot with a bar chart

#rajouter les structures de convexité différence H26/X26


#----------------------------------------
#               VEGA
#----------------------------------------
def single_vega(
    instrument_type,
    expiry,
    F_market,
    K,
    v,
    T,
    qty
):
    """
    Vega vs volatility, in contract units
    """
    itype = str(instrument_type).strip().lower()
    F = F_market[expiry]

    if itype == "f":
        vega = 0.0 # Pas de vega sur un instrument linéaire tenu en cash & carry simple

    elif itype in ("c", "p"):
        d1 = bs76_d1(F, K, v, T)
        vega = F * n(d1) * np.sqrt(T)

    else:
        raise ValueError(f"Instrument inconnu : {instrument_type}")

    return vega * float(qty)

def compute_line_vegas(
    df,
    F_market,
    col_type="type",
    col_expiry="expiry",
    col_K="strike",
    col_v="vol",
    col_T="T",
    col_qty="quantity"
):
    """
    Vega computation for each line of the portfolio.
    """
    def vega_row(row):
        return single_vega(
            instrument_type=row[col_type],
            expiry=row[col_expiry],
            F_market=F_market,
            K=row[col_K],
            v=row[col_v],
            T=row[col_T],
            qty=row[col_qty]
        )

    df = df.copy()
    df["vega"] = df.apply(vega_row, axis=1)
    return df

def portfolio_vega_by_expiry(
    df,
    col_expiry="expiry",
    col_vega="vega"
):
    """
    Vega agregation for each expiry.
    """
    return (
        df
        .groupby(col_expiry, as_index=False)[col_vega]
        .sum()
        .rename(columns={col_vega: "vega_expiry"})
    )

#diplay Vega vs Fut price 
def plot_vega_vs_future_subplots(
    df,
    expiries,
    F_market,
    F_range_pct=0.10,
    price_steps=40
):
    """
    Subplots Vega vs Future price, one subplot per expiry.
    """
    n = len(expiries)

    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=False,
        subplot_titles=[f"Vega vs Future – {exp}" for exp in expiries]
    )

    for i, expiry in enumerate(expiries, start=1):
        F0 = F_market[expiry]

        prices = np.linspace(
            F0 * (1 - F_range_pct),
            F0 * (1 + F_range_pct),
            price_steps
        )

        vegas = []

        for F in prices:
            F_tmp = F_market.copy()
            F_tmp[expiry] = F

            df_tmp = compute_line_vegas(df, F_tmp)

            vega_expiry = (
                df_tmp[df_tmp["expiry"] == expiry]["vega"]
                .sum()
            )

            vegas.append(vega_expiry)

        fig.add_trace(
            go.Scatter(
                x=prices,
                y=vegas,
                mode="lines",
                name=f"{expiry}"
            ),
            row=i,
            col=1
        )

        
        fig.add_vline(
            x=F0,
            line_dash="dash",
            line_color="gray",
            row=i,
            col=1
        )

    fig.update_layout(
        height=300 * n,
        title="Vega du portefeuille vs prix du future (par échéance)",
        showlegend=False,
        template="plotly_white"
    )

    fig.update_xaxes(title_text="Prix du future")
    fig.update_yaxes(title_text="Vega (contrats / point de vol)")

    return fig


#display of  vega buckets 


#----------------------------------------
#               THETA
#----------------------------------------
# In order to be more user friendly, we do display the Theta in Theta/Day 
def single_theta(
    instrument_type,
    expiry,
    F_market,
    K,
    v,
    T,
    qty,
    contract_multiplier,
    day_count=252
):
    """
    Theta in EUR / day
    """
    itype = str(instrument_type).strip().lower()
    F = F_market[expiry]

    if itype == "f":
        theta = 0.0

    elif itype in ("c", "p"):
        d1 = bs76_d1(F, K, v, T)

        # Annual Theta (Black-76)
        theta_annual = - (F * n(d1) * v) / (2 * np.sqrt(T))

        # Conversion in EUR / day
        theta = theta_annual * float(qty) * float(contract_multiplier) / day_count

    else:
        raise ValueError(f"Instrument inconnu : {instrument_type}")

    return theta


def compute_line_thetas(
    df,
    F_market,
    col_type="type",
    col_expiry="expiry",
    col_K="strike",
    col_v="vol",
    col_T="T",
    col_qty="quantity",
    col_multiplier="contract_multiplier"
):
    """
    Theta computation for each line of the portfolio (EUR / day).
    """
    def theta_row(row):
        return single_theta(
            instrument_type=row[col_type],
            expiry=row[col_expiry],
            F_market=F_market,
            K=row[col_K],
            v=row[col_v],
            T=row[col_T],
            qty=row[col_qty],
            contract_multiplier=row[col_multiplier]
        )

    df = df.copy()
    df["theta"] = df.apply(theta_row, axis=1)
    return df


def portfolio_theta_by_expiry(
    df,
    col_expiry="expiry",
    col_theta="theta"
):
    """
    Theta agregation by expiry  (EUR / day).
    """
    return (
        df
        .groupby(col_expiry, as_index=False)[col_theta]
        .sum()
        .rename(columns={col_theta: "theta_expiry"})
    )

#Plot the Theta vs future value 

def plot_theta_vs_future_subplots(
    df,
    expiries,
    F_market,
    F_range_pct=0.10,
    price_steps=40
):
    """
    Subplots Theta vs Future price, one subplot per expiry.
    Theta  in EUR / day.
    """
    n = len(expiries)

    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=False,
        subplot_titles=[f"Theta vs Future – {exp}" for exp in expiries]
    )

    for i, expiry in enumerate(expiries, start=1):
        F0 = F_market[expiry]

        prices = np.linspace(
            F0 * (1 - F_range_pct),
            F0 * (1 + F_range_pct),
            price_steps
        )

        thetas = []

        for F in prices:
            F_tmp = F_market.copy()
            F_tmp[expiry] = F

            df_tmp = compute_line_thetas(df, F_tmp)

            theta_expiry = (
                df_tmp[df_tmp["expiry"] == expiry]["theta"]
                .sum()
            )

            thetas.append(theta_expiry)

        fig.add_trace(
            go.Scatter(
                x=prices,
                y=thetas,
                mode="lines",
                name=f"{expiry}"
            ),
            row=i,
            col=1
        )

        
        fig.add_vline(
            x=F0,
            line_dash="dash",
            line_color="gray",
            row=i,
            col=1
        )

    fig.update_layout(
        height=300 * n,
        title="Theta du portefeuille vs prix du future (USD / jour, par échéance)",
        showlegend=False,
        template="plotly_white"
    )

    fig.update_xaxes(title_text="Prix du future")
    fig.update_yaxes(title_text="Theta (USD / jour)")

    return fig

#----------------------------------------
#               RHÔ
#----------------------------------------
def single_rho(
    instrument_type,
    expiry,
    F_market,
    K,
    v,
    T,
    r,
    qty,
    contract_multiplier
):
    """
    Rhô Black-76 in USD for + or - 1%  in rates.
    """
    itype = str(instrument_type).strip().lower()
    F = F_market[expiry]

    if itype == "f":
        rho = 0.0

    elif itype in ("c", "p"):
        option_price = bs76_price(
            option_type=itype,
            F=F,
            K=K,
            v=v,
            T=T,
            r=r
        )

        # Rhô = dV/dr ≈ -T * V
        rho = -T * option_price * float(qty) * float(contract_multiplier) / 0.01

    else:
        raise ValueError(f"Instrument inconnu : {instrument_type}")

    return rho

def compute_line_rhos(
    df,
    F_market,
    r,
    col_type="type",
    col_expiry="expiry",
    col_K="strike",
    col_v="vol",
    col_T="T",
    col_qty="quantity",
    col_multiplier="contract_multiplier"
):
    """
    Rhô computation for each line (USD / 1%).
    """
    def rho_row(row):
        return single_rho(
            instrument_type=row[col_type],
            expiry=row[col_expiry],
            F_market=F_market,
            K=row[col_K],
            v=row[col_v],
            T=row[col_T],
            r=r,
            qty=row[col_qty],
            contract_multiplier=row[col_multiplier]
        )

    df = df.copy()
    df["rho"] = df.apply(rho_row, axis=1)
    return df

def portfolio_rho_by_expiry(
    df,
    col_expiry="expiry",
    col_rho="rho"
):
    """
    Rhô agregation by expiry (USD / 1%).
    """
    return (
        df
        .groupby(col_expiry, as_index=False)[col_rho]
        .sum()
        .rename(columns={col_rho: "rho_expiry"})
    )


#----------------------------------------
#               VANNA
#----------------------------------------
def single_vanna(
    instrument_type,
    expiry,
    F_market,
    K,
    v,
    T,
    qty,
    contract_multiplier
):
    itype = str(instrument_type).strip().lower()
    F = F_market[expiry]

    if itype == "f":
        vanna = 0.0

    elif itype in ("c", "p"):
        d1 = bs76_d1(F, K, v, T)

        vanna = (
            np.sqrt(T)
            * n(d1)
            * (1.0 - d1 / (v * np.sqrt(T)))
        )

        
        vanna *= float(qty) * float(contract_multiplier)

    else:
        raise ValueError(f"Instrument inconnu : {instrument_type}")

    return vanna

def compute_line_vannas(
    df,
    F_market,
    col_type="type",
    col_expiry="expiry",
    col_K="strike",
    col_v="vol",
    col_T="T",
    col_qty="quantity",
    col_multiplier="contract_multiplier"
):
    def vanna_row(row):
        return single_vanna(
            instrument_type=row[col_type],
            expiry=row[col_expiry],
            F_market=F_market,
            K=row[col_K],
            v=row[col_v],
            T=row[col_T],
            qty=row[col_qty],
            contract_multiplier=row[col_multiplier]
        )

    df = df.copy()
    df["vanna"] = df.apply(vanna_row, axis=1)
    return df

def portfolio_vanna_by_expiry(
    df,
    col_expiry="expiry",
    col_vanna="vanna"
):
    return (
        df
        .groupby(col_expiry, as_index=False)[col_vanna]
        .sum()
        .rename(columns={col_vanna: "vanna_expiry"})
    )



#----------------------------------------
#               VOLGA
#----------------------------------------
def single_volga(
    instrument_type,
    expiry,
    F_market,
    K,
    v,
    T,
    qty,
    contract_multiplier
):
    itype = str(instrument_type).strip().lower()
    F = F_market[expiry]

    if itype == "f":
        volga = 0.0

    elif itype in ("c", "p"):
        d1 = bs76_d1(F, K, v, T)
        d2 = d1 - v * np.sqrt(T)

        volga = (
            F
            * np.sqrt(T)
            * n(d1)
            * (d1 * d2 / v)
        )

        volga *= float(qty) * float(contract_multiplier)

    else:
        raise ValueError(f"Instrument inconnu : {instrument_type}")

    return volga


def compute_line_volgas(
    df,
    F_market,
    col_type="type",
    col_expiry="expiry",
    col_K="strike",
    col_v="vol",
    col_T="T",
    col_qty="quantity",
    col_multiplier="contract_multiplier"
):
    def volga_row(row):
        return single_volga(
            instrument_type=row[col_type],
            expiry=row[col_expiry],
            F_market=F_market,
            K=row[col_K],
            v=row[col_v],
            T=row[col_T],
            qty=row[col_qty],
            contract_multiplier=row[col_multiplier]
        )

    df = df.copy()
    df["volga"] = df.apply(volga_row, axis=1)
    return df


def portfolio_volga_by_expiry(
    df,
    col_expiry="expiry",
    col_volga="volga"
):
    return (
        df
        .groupby(col_expiry, as_index=False)[col_volga]
        .sum()
        .rename(columns={col_volga: "volga_expiry"})
    )

#----------------------------------------
#               CHARM
#----------------------------------------
def single_charm(
    instrument_type,
    expiry,
    F_market,
    K,
    v,
    T,
    qty,
    day_count=365
):
    itype = str(instrument_type).strip().lower()
    F = F_market[expiry]

    if itype == "f":
        charm = 0.0

    elif itype in ("c", "p"):
        d1 = bs76_d1(F, K, v, T)

        charm_annual = -n(d1) / (2.0 * T)
        charm = charm_annual * float(qty) / day_count

    else:
        raise ValueError(f"Instrument inconnu : {instrument_type}")

    return charm

def compute_line_charms(
    df,
    F_market,
    col_type="type",
    col_expiry="expiry",
    col_K="strike",
    col_v="vol",
    col_T="T",
    col_qty="quantity"
):
    def charm_row(row):
        return single_charm(
            instrument_type=row[col_type],
            expiry=row[col_expiry],
            F_market=F_market,
            K=row[col_K],
            v=row[col_v],
            T=row[col_T],
            qty=row[col_qty]
        )

    df = df.copy()
    df["charm"] = df.apply(charm_row, axis=1)
    return df

def portfolio_charm_by_expiry(
    df,
    col_expiry="expiry",
    col_charm="charm"
):
    return (
        df
        .groupby(col_expiry, as_index=False)[col_charm]
        .sum()
        .rename(columns={col_charm: "charm_expiry"})
    )



#---------------------
#       ALL GREEKS 
#---------------------
def build_greeks_dataframe(
    df,
    F_market,
    r=0.02,
    day_count_theta=252
):
    """
    Build a full greeks DataFrame line by line.
    This function is the single source of truth for all greeks.
    """

    df = df.copy()
    

    # --- Containers (crucial : pas de apply)
    deltas = []
    gammas = []
    vegas  = []
    thetas = []
    rhos   = []
    vannas = []
    volgas = []
    charms = []

    for _, row in df.iterrows():
        itype  = str(row["type"]).strip().lower()
        expiry = str(row["expiry"]).strip().upper()
        qty    = float(row["quantity"])

        K = row.get("strike", np.nan)
        v = row.get("vol", np.nan)
        T = row.get("T", np.nan)
        mult = float(row.get("contract_multiplier", 1.0))

        # ---- Delta
        deltas.append(
            single_delta(itype, expiry, F_market, K, v, T, qty)
        )

        # ---- Gamma
        gammas.append(
            single_gamma(itype, expiry, F_market, K, v, T, qty)
        )

        # ---- Vega
        vegas.append(
            single_vega(itype, expiry, F_market, K, v, T, qty)
        )

        # ---- Theta (EUR / day)
        thetas.append(
            single_theta(
                itype, expiry, F_market, K, v, T,
                qty, mult, day_count_theta
            )
        )

        # ---- Rho (EUR / 1%)
        rhos.append(
            single_rho(
                itype, expiry, F_market, K, v, T,
                r, qty, mult
            )
        )

        # ---- Vanna
        vannas.append(
            single_vanna(
                itype, expiry, F_market, K, v, T,
                qty, mult
            )
        )

        # ---- Volga
        volgas.append(
            single_volga(
                itype, expiry, F_market, K, v, T,
                qty, mult
            )
        )

        # ---- Charm
        charms.append(
            single_charm(
                itype, expiry, F_market, K, v, T, qty
            )
        )

    # ---- Attach columns (SAFE)
    df["delta"] = deltas
    df["gamma"] = gammas
    df["vega"]  = vegas
    df["theta"] = thetas
    df["rho"]   = rhos
    df["vanna"] = vannas
    df["volga"] = volgas
    df["charm"] = charms

    return df

#-----------------------------------
#           PNL 
#-----------------------------------
def compute_theoretical_price(row, F_market, r=0.02):
    itype = row["type"].lower()
    expiry = row["expiry"]
    F = F_market[expiry]

    if itype == "f":
        return F

    return bs76_price(
        option_type=itype,
        F=F,
        K=row["strike"],
        v=row["vol"],
        T=row["T"],
        r=r
    )

def compute_live_pnl(df, F_market, r=0.02):
    df = df.copy()

    theo_prices = []
    pnls = []

    for _, row in df.iterrows():
        theo_price = compute_theoretical_price(row, F_market, r)
        entry_price = row["price/premium"]

        pnl = (
            (theo_price - entry_price)
            * row["quantity"]
            * row["contract_multiplier"]
        )

        theo_prices.append(theo_price)
        pnls.append(pnl)

    df["theoretical_price"] = theo_prices
    df["live_pnl"] = pnls

    return df

#PnL explaination using Greeks 
def compute_pnl_explain(
    df,
    F_market,
    F_ref,
    vol_ref,
    day_count=252
):
    """
    PnL explain between ref market and live market
    """

    df = df.copy()

    delta_pnl = []
    gamma_pnl = []
    vega_pnl  = []
    theta_pnl = []

    for _, row in df.iterrows():
        expiry = row["expiry"]

        dF = F_market[expiry] - F_ref[expiry]
        dVol = row["vol"] - vol_ref[expiry]

        delta_pnl.append(
            row["delta"] * dF * row["contract_multiplier"]
        )

        gamma_pnl.append(
            0.5 * row["gamma"] * dF**2 * row["contract_multiplier"]
        )

        vega_pnl.append(
            row["vega"] * dVol * row["contract_multiplier"]
        )

        theta_pnl.append(
            row["theta"] * 1  # 1 day
        )

    df["pnl_delta"] = delta_pnl
    df["pnl_gamma"] = gamma_pnl
    df["pnl_vega"]  = vega_pnl
    df["pnl_theta"] = theta_pnl

    df["pnl_explained"] = (
        df["pnl_delta"]
        + df["pnl_gamma"]
        + df["pnl_vega"]
        + df["pnl_theta"]
    )

    df["pnl_residual"] = df["live_pnl"] - df["pnl_explained"]

    return df




# # #TESSTTTTT
#df["T"] = df["expiry"].apply(get_ttm_from_today)

# df_test = pd.DataFrame({
#     "trade_id": [1, 2, 3, 4, 5, 6],
#     "type": ["C", "P", "F", "C", "P", "F"],
#     "expiry": ["H26", "H26", "H26", "X26", "X26", "X26"],
#     "quantity": [20, -20, 10, 15, -10, -50],  # quantité signée
#     "strike": [480, 495, np.nan, 470, 485, np.nan],
#     "vol": [0.22, 0.22, np.nan, 0.25, 0.25, np.nan],
#     "T": [0.45, 0.45, 0.45, 0.75, 0.75, 0.75],
#     "contract_multiplier": [50, 50, 50, 50, 50, 50],
#     # prix de trade (INFORMATIF – PAS utilisé pour les greeks)
#     "trade_price_or_premium": [5.0, 4.0, 485.0, 4.5, 3.8, 490.0]
# })

# df_with_deltas = compute_line_deltas(
#     df_test,
#     F_market=F_market
# )

# print(df_with_deltas[[
#     "trade_id", "type", "expiry", "quantity", "delta"
# ]])

# delta_by_expiry = portfolio_delta_by_expiry(df_with_deltas)

# print(delta_by_expiry)

# hedge_actions = delta_hedge_action_by_expiry(delta_by_expiry)
# print(hedge_actions)

# hedge_costs = hedge_cash_cost_by_expiry(
#     hedge_actions,
#     F_market=F_market,
#     contract_multiplier=50
# )

# print(hedge_costs)

# # Gamma ligne par ligne
# df_with_gammas = compute_line_gammas(df_test, F_market)

# # Gamma par expiry
# gamma_by_expiry = portfolio_gamma_by_expiry(df_with_gammas)
# print(gamma_by_expiry)



# plot_delta_vs_future_subplots(
#     df_test,
#     expiries=["H26", "X26"],
#     F_market=F_market,
#     F_range_pct=0.10   # ±10%
# )

# plot_gamma_vs_future_subplots(
#     df_test,
#     expiries=["H26", "X26"],
#     F_market=F_market,
#     F_range_pct=0.10
# )


# # Vega ligne par ligne
# df_with_vegas = compute_line_vegas(df_test, F_market)

# # Vega par expiry
# vega_by_expiry = portfolio_vega_by_expiry(df_with_vegas)
# print(vega_by_expiry)

# # Plot
# plot_vega_vs_future_subplots(
#     df_test,
#     expiries=["H26", "X26"],
#     F_market=F_market,
#     F_range_pct=0.10
# )


# # Theta ligne par ligne
# df_with_thetas = compute_line_thetas(df_test, F_market)

# # Theta par expiry
# theta_by_expiry = portfolio_theta_by_expiry(df_with_thetas)
# print(theta_by_expiry)

# # Plot
# plot_theta_vs_future_subplots(
#     df_test,
#     expiries=["H26", "X26"],
#     F_market=F_market,
#     F_range_pct=0.10
# )


# df_vanna = compute_line_vannas(df_test, F_market)
# vanna_by_expiry = portfolio_vanna_by_expiry(df_vanna)

# df_volga = compute_line_volgas(df_test, F_market)
# volga_by_expiry = portfolio_volga_by_expiry(df_volga)

# df_charm = compute_line_charms(df_test, F_market)
# charm_by_expiry = portfolio_charm_by_expiry(df_charm)

# print(vanna_by_expiry)
# print(volga_by_expiry)
# print(charm_by_expiry)
