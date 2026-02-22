#=====================
#Here is the file in wich we compute the PnL of the position with 2 different components 
#The Latent PnL --> which is available on the "Main book" page 
#           File format = 
#           ex : 
#The CLosed PnL --> wich is available on the "Closed Positions" page 
#           File format = 
#           ex : 
#All the informations used to compute the PnL are information extracted for Euronext 
#For further information check : https://www.cmegroup.com/markets/agriculture/oilseeds/soybean-meal.quotes.html
#=====================

#=====================
#       IMPORTS

import pandas as pd 
import numpy as np 
from pathlib import Path
#=====================

#CONSTANT VARIABLES
BASE_DATA_PATH_2 = Path("/Users/hugoberthelier/Desktop/PricerLB_Soymeal/books") #<---- replace this by your desktop path

COLUMNS = [
    "trade_id",
    "date",
    "open_date",
    "underlying",
    "type",
    "expiry",
    "lots",
    "quantity",
    "strike",
    "price/premium",
    "end_price",
    "cost",
    "units",
]
CONTRACT_MULTIPLIER = 100

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


#--------------------------
def get_closedbook_path(account_id: str) -> Path:
    return BASE_DATA_PATH_2 / str(account_id) / "closed_book.xlsx"


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """
    clean data 
    """
    if series is None:
        return series
    s = series.astype(str).str.replace("\u00a0", "", regex=False)
    s = s.str.replace(" ", "", regex=False)
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

def load_closed_positions(account_id: str) -> pd.DataFrame:
    account_id = str(account_id)
    path = get_closedbook_path(account_id)

    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)

    df = pd.read_excel(path)

    # Assure toutes les colonnes
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[COLUMNS].copy()

    # Parsing dates
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["open_date"] = pd.to_datetime(df["open_date"], dayfirst=True, errors="coerce")

    # Numerics
    for col in ["lots", "quantity", "strike", "price/premium", "end_price"]:
        df[col] = _coerce_numeric(df[col])

    # Cost libre mais défaut = 0
    df["cost"] = _coerce_numeric(df.get("cost"))
    df["cost"] = df["cost"].fillna(0.0)

    df["type"] = df["type"].astype(str).str.lower().str.strip()
    df["expiry"] = df["expiry"].astype(str).str.strip()
    df["underlying"] = df["underlying"].astype(str).str.strip()

    return df



def compute_line_pnl(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule le PnL ligne par ligne (“realized PnL”),
    en tenant compte du contract multiplier.
    quantity signée → long/short.
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    out["quantity"] = out["quantity"].fillna(out["lots"])
    out["quantity"] = out["quantity"].fillna(0.0)

    # PnL total = (prix clôture - prix d'ouverture) × quantité × multiplicateur contrat
    out["pnl"] = (
        (out["end_price"] - out["price/premium"])
        * out["quantity"]
        * CONTRACT_MULTIPLIER
    )

    # Durée de détention (jours)
    out["holding_days"] = (out["date"] - out["open_date"]).dt.days

    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.to_period("M").astype(str)

    out["data_ok"] = (
        out["date"].notna()
        & out["open_date"].notna()
        & out["price/premium"].notna()
        & out["end_price"].notna()
        & out["quantity"].notna()
    )

    return out



def expiry_sort_key(expiry: str):
    """
    Convert an expiry type 'G26' in (year, month)
    to allow expiry sorting.
    """
    if expiry is None:
        return (9999, 99)

    s = str(expiry).strip().upper()
    if len(s) < 2:
        return (9999, 99)

    month_code = s[0]
    year_code = s[1:]

    month = FUTURES_MONTH_MAP.get(month_code, 99)

    try:
        year = int(year_code)
        year = 2000 + year if year < 70 else 1900 + year
    except ValueError:
        year = 9999

    return (year, month)




def compute_closed_pnl(df: pd.DataFrame) -> dict:
    """
    Résumé global du PnL des positions fermées
    """
    if df is None or df.empty:
        return {
            "total_pnl": 0.0,
            "n_trades": 0,
            "avg_pnl_per_trade": 0.0,
            "win_rate": np.nan,
        }

    d = df[df.get("data_ok", True)].copy()
    total = float(d["pnl"].sum())
    n = int(d.shape[0])
    avg = float(d["pnl"].mean()) if n > 0 else 0.0
    win_rate = float((d["pnl"] > 0).mean()) if n > 0 else np.nan

    return {
        "total_pnl": total,
        "n_trades": n,
        "avg_pnl_per_trade": avg,
        "win_rate": win_rate,
    }

def compute_pnl_by_expiry(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["expiry", "pnl", "n_trades"])

    d = df[df.get("data_ok", True)].copy()

    g = (
        d.groupby("expiry", dropna=False)
        .agg(
            pnl=("pnl", "sum"),
            n_trades=("trade_id", "count")
        )
        .reset_index()
    )

    g["sort_key"] = g["expiry"].apply(expiry_sort_key)
    g = (
        g.sort_values("sort_key")
         .drop(columns="sort_key")
         .reset_index(drop=True)
    )

    return g



def compute_pnl_by_month(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["month", "pnl", "n_trades"])

    d = df[df.get("data_ok", True)].copy()
    g = (
        d.groupby("month", dropna=False)
        .agg(pnl=("pnl", "sum"), n_trades=("trade_id", "count"))
        .reset_index()
        .sort_values("month")
    )
    return g

def compute_pnl_by_year(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["year", "pnl", "n_trades"])

    d = df[df.get("data_ok", True)].copy()
    if "year" not in d.columns:
        d["year"] = d["date"].dt.year

    g = (
        d.groupby("year", dropna=False)
        .agg(pnl=("pnl", "sum"), n_trades=("trade_id", "count"))
        .reset_index()
        .sort_values("year")
    )
    return g

def build_daily_pnl_series_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renvoie: date, year, daily_pnl, cum_pnl_ytd
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "year", "daily_pnl", "cum_pnl_ytd"])

    d = df[df.get("data_ok", True)].copy()
    d["year"] = d["date"].dt.year

    daily = (
        d.groupby(["year", "date"])
        .agg(daily_pnl=("pnl", "sum"))
        .reset_index()
        .sort_values(["year", "date"])
    )

    daily["cum_pnl_ytd"] = daily.groupby("year")["daily_pnl"].cumsum()
    return daily


def compute_returns(equity_curve: pd.Series) -> pd.Series:
    if equity_curve is None or equity_curve.empty:
        return pd.Series(dtype=float)

    eq = equity_curve.astype(float)
    rets = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    return rets

def compute_max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve is None or equity_curve.empty:
        return np.nan

    eq = equity_curve.astype(float)
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return float(dd.min())


def compute_sharpe(returns: pd.Series, periods_per_year: int = 252, rf: float = 0.0) -> float:
    if returns is None or returns.empty:
        return np.nan

    r = returns.astype(float).dropna()
    if r.std(ddof=1) == 0:
        return np.nan

    rf_period = (1 + rf) ** (1 / periods_per_year) - 1
    excess = r - rf_period
    sharpe = np.sqrt(periods_per_year) * excess.mean() / excess.std(ddof=1)
    return float(sharpe)

def build_daily_pnl_series(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "daily_pnl", "cum_pnl"])

    d = df[df.get("data_ok", True)].copy()
    daily = d.groupby("date").agg(daily_pnl=("pnl", "sum")).reset_index().sort_values("date")
    daily["cum_pnl"] = daily["daily_pnl"].cumsum()
    return daily