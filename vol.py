import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from pathlib import Path
import plotly.graph_objects as go
import re
from datetime import date


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



def load_raw_vol_data(base_path: Path, filenames: list[str],) -> dict[str, pd.DataFrame]:

    data = {}

    for filename in filenames:
        path = base_path / filename
        key = filename.replace(".xlsx", "")
        data[key] = pd.read_excel(path)

    return data

def clean_vol_data(df: pd.DataFrame, column_mapping: dict) -> pd.DataFrame:
    """
    clean the excel file 
    """

    # Rename
    df_clean = (
        df
        .rename(columns=column_mapping)
        .loc[:, column_mapping.values()]
        .copy()
    )

    df_clean = df_clean.drop(index=0).reset_index(drop=True)

    #Conversion Implied Vol (% -> float)
    for col in ["ImplV_Call", "ImplV_Put"]:
        df_clean[col] = (
            df_clean[col]
            .astype(str)
            .str.replace("%", "", regex=False)
            .astype(float)
            / 100
        )

    #Conversion of Strikes ('117.25s' -> 117.25)
    df_clean["Strike"] = (
        df_clean["Strike"]
        .astype(str)
        .str.extract(r"([-+]?\d*\.?\d+)")
        .astype(float)
    )

    #Conversion of deltas
    df_clean["Delta_Call"] = pd.to_numeric(df_clean["Delta_Call"], errors="coerce")
    df_clean["Delta_Put"] = pd.to_numeric(df_clean["Delta_Put"], errors="coerce")

    df_clean = (
        df_clean
        .dropna(subset=["Strike"])
        .sort_values("Strike")
        .reset_index(drop=True)
    )

    return df_clean


def expiry_from_filename(filename: str, year_suffix: str = "26") -> str:
    """
    Ex: 'VolKbarchart.xlsx' -> 'K26'
    """
    m = re.search(r"Vol([A-Z])", filename, flags=re.IGNORECASE)
    if not m:
        return filename.replace(".xlsx", "")
    letter = m.group(1).upper()
    return f"{letter}{year_suffix}"


def add_smile_metrics_black76(df: pd.DataFrame, F: float) -> pd.DataFrame:
    """
    Pour Black-76 (Futures):
      - IV_Mid en % (float) : moyenne call/put (fallback si un côté manque)
      - Delta_Signed : puts négatifs, calls positifs, choisi via K vs F
      - Delta_Abs : |Delta_Signed|
    """
    out = df.copy()

    # IV mid
    out["IV_Mid"] = np.nan
    has_call = out["ImplV_Call"].notna()
    has_put  = out["ImplV_Put"].notna()

    out.loc[has_call & has_put, "IV_Mid"] = 0.5 * (out.loc[has_call & has_put, "ImplV_Call"] +
                                                  out.loc[has_call & has_put, "ImplV_Put"])
    out.loc[has_call & ~has_put, "IV_Mid"] = out.loc[has_call & ~has_put, "ImplV_Call"]
    out.loc[~has_call & has_put, "IV_Mid"] = out.loc[~has_call & has_put, "ImplV_Put"]

    # Delta signé:
    # - pour K <= F : on prend le put (delta négatif)
    # - pour K >  F : on prend le call (delta positif)
    out["Delta_Signed"] = np.nan
    left = out["Strike"] <= F
    right = out["Strike"] > F

    out.loc[left, "Delta_Signed"]  = -out.loc[left, "Delta_Put"].abs()
    out.loc[right, "Delta_Signed"] =  out.loc[right, "Delta_Call"].abs()

    out["Delta_Abs"] = out["Delta_Signed"].abs()

    out = out.dropna(subset=["Strike", "IV_Mid"])
    out = out.sort_values("Strike").reset_index(drop=True)
    return out


def build_smile_panel_from_excels(
    base_path: Path,
    filenames: list[str],
    column_mapping: dict,
    F_market: dict[str, float],
    year_suffix: str = "26"
) -> pd.DataFrame:

    frames = []
    for fn in filenames:
        expiry = expiry_from_filename(fn, year_suffix=year_suffix)
        if expiry not in F_market:
            # si pas de future price dispo, on skip (ou raise selon ton choix)
            continue

        df_raw = pd.read_excel(base_path / fn)
        df_clean = clean_vol_data(df_raw, column_mapping)
        df_feat = add_smile_metrics_black76(df_clean, F=F_market[expiry])

        df_feat["expiry"] = expiry
        df_feat["F"] = F_market[expiry]
        frames.append(df_feat)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def plot_smiles_panel(
    panel: pd.DataFrame,
    x_mode: str = "strike",  # "strike" | "delta_signed" | "delta_abs"
    expiries: list[str] | None = None,
    title: str = "Smiles de volatilité",
) -> go.Figure:

    if panel.empty:
        return go.Figure()

    if expiries is None:
        expiries = sorted(panel["expiry"].unique())

    if x_mode == "strike":
        x_col, x_title = "Strike", "Prix d'exercice"
    elif x_mode == "delta_signed":
        x_col, x_title = "Delta_Signed", "Delta (signé)"
    elif x_mode == "delta_abs":
        x_col, x_title = "Delta_Abs", "Delta (absolu)"
    else:
        raise ValueError("x_mode must be: 'strike', 'delta_signed', 'delta_abs'")

    fig = go.Figure()
    for e in expiries:
        dfe = panel.loc[panel["expiry"] == e].dropna(subset=[x_col, "IV_Mid"]).copy()
        dfe = dfe.sort_values(x_col)

        fig.add_trace(go.Scatter(
            x=dfe[x_col],
            y=100.0 * dfe["IV_Mid"],
            mode="lines",
            name=e
        ))

    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title="Volatilité moyenne (%)",
        template="plotly_dark",
        legend_title_text="Echéance"
    )

    if x_mode == "delta_signed":
        fig.update_xaxes(range=[-1, 1])
    if x_mode == "delta_abs":
        fig.update_xaxes(range=[0, 1])

    return fig






#--------------------
# 3D Surface de vol 
#--------------------

def expiry_code_to_date(expiry_code: str) -> pd.Timestamp:
    """
    Converts 'K26' -> 2026-05-19 
    Also support 'K2026' .
    """
    e = str(expiry_code).strip().upper()
    if len(e) < 3:
        raise ValueError(f"Invalid expiry_code: {expiry_code}")

    letter = e[0]
    year_part = e[1:]

    if letter not in FUTURES_MONTH_MAP:
        raise ValueError(f"Unknown month code '{letter}' in {expiry_code}")

    # '26' => 2026, '2026' => 2026
    year = 2000 + int(year_part) if len(year_part) == 2 else int(year_part)
    month = FUTURES_MONTH_MAP[letter]

    if year not in EXPIRY_DATES or month not in EXPIRY_DATES[year]:
        raise ValueError(f"No expiry date configured for year={year}, month={month} (code={expiry_code})")

    day = EXPIRY_DATES[year][month]
    return pd.Timestamp(year=year, month=month, day=day).normalize()

def year_fraction_act365(valuation_date: date | pd.Timestamp, expiry_date: date | pd.Timestamp) -> float:
    v = pd.Timestamp(valuation_date).normalize()
    x = pd.Timestamp(expiry_date).normalize()
    return max((x - v).days, 0) / 252.0



def build_vol_surface_matrix(
    panel: pd.DataFrame,
    valuation_date: date | pd.Timestamp,
    expiries: list[str] | None = None,
    strike_grid: np.ndarray | None = None,
    n_strikes: int = 60,
    fill_across_expiries: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    need to have in the panel: ['expiry','Strike','IV_Mid'] with IV_Mid in decimals (0.22)

    Return:
      - K_grid : (n_strikes,)
      - T      : (n_expiries,) ACT/365
      - IV     : (n_expiries, n_strikes)
      - labels : ordonnated list of expiries 
    """
    if panel is None or panel.empty:
        raise ValueError("panel is empty")

    needed = {"expiry", "Strike", "IV_Mid"}
    missing = needed - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {missing}")

    df = panel.copy()
    if expiries is not None:
        df = df[df["expiry"].isin(expiries)].copy()

    if df.empty:
        raise ValueError("No data after expiry filtering")

    # Strike grid
    if strike_grid is None:
        kmin = float(np.nanmin(df["Strike"].values))
        kmax = float(np.nanmax(df["Strike"].values))
        if not np.isfinite(kmin) or not np.isfinite(kmax) or kmin >= kmax:
            raise ValueError("Cannot infer strike range from panel")
        strike_grid = np.linspace(kmin, kmax, n_strikes)
    else:
        strike_grid = np.asarray(strike_grid, dtype=float)
        if strike_grid.ndim != 1 or strike_grid.size < 5:
            raise ValueError("strike_grid must be 1D with >= 5 points")

    # Expiries order by actual expiry date
    labels = sorted(df["expiry"].unique().tolist(), key=lambda e: expiry_code_to_date(e))
    T = np.array([year_fraction_act365(valuation_date, expiry_code_to_date(e)) for e in labels], dtype=float)

    IV = np.full((len(labels), strike_grid.size), np.nan, dtype=float)

   
    for i, e in enumerate(labels):
        dfe = df[df["expiry"] == e].dropna(subset=["Strike", "IV_Mid"]).copy()
        if dfe.empty:
            continue

        dfe = dfe.sort_values("Strike")
        
        if dfe["Strike"].duplicated().any():
            dfe = dfe.groupby("Strike", as_index=False)["IV_Mid"].mean().sort_values("Strike")

        x = dfe["Strike"].to_numpy(dtype=float)
        y = dfe["IV_Mid"].to_numpy(dtype=float)

        y_interp = np.interp(strike_grid, x, y)
        # avoid artificial etrapolation 
        y_interp[(strike_grid < x.min()) | (strike_grid > x.max())] = np.nan

        IV[i, :] = y_interp

    
    if fill_across_expiries:
        iv_df = pd.DataFrame(IV, index=T, columns=strike_grid)
        iv_df = iv_df.interpolate(axis=1, method="linear", limit_direction="both")
        iv_df = iv_df.sort_index().interpolate(axis=0, method="linear", limit_direction="both")
        IV = iv_df.to_numpy(dtype=float)

    return strike_grid, T, IV, labels


def plot_vol_surface(
    panel: pd.DataFrame,
    valuation_date: date | pd.Timestamp,
    expiries: list[str] | None = None,
    strike_grid: np.ndarray | None = None,
    n_strikes: int = 60,
    fill_across_expiries: bool = True,
    title: str = "3D Surface implied vol"
) -> go.Figure:

    K, T, IV, labels = build_vol_surface_matrix(
        panel=panel,
        valuation_date=valuation_date,
        expiries=expiries,
        strike_grid=strike_grid,
        n_strikes=n_strikes,
        fill_across_expiries=fill_across_expiries
    )

    Z = 100.0 * IV  # in %

    fig = go.Figure(data=[
        go.Surface(x=K, y=T, z=Z, showscale=True)
    ])

    fig.update_layout(
        title=title,
        template="plotly_dark",
        scene=dict(
            xaxis_title="Strike",
            yaxis_title="TTM (years, ACT/252)",
            zaxis_title="IV (%)"
        ),
        margin=dict(l=10, r=10, t=50, b=10)
    )

    return fig


def _interp_1d(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """
    Interpolation linéaire 1D avec NaN hors bornes (pas d'extrapolation).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xq = np.asarray(xq, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2:
        return np.full_like(xq, np.nan, dtype=float)

    order = np.argsort(x)
    x, y = x[order], y[order]

    if np.any(np.diff(x) == 0):
        df = pd.DataFrame({"x": x, "y": y}).groupby("x", as_index=False)["y"].mean()
        x, y = df["x"].to_numpy(), df["y"].to_numpy()

    yq = np.interp(xq, x, y)
    yq[(xq < x.min()) | (xq > x.max())] = np.nan
    return yq


def _get_smile_by_expiry(panel: pd.DataFrame, expiry: str) -> pd.DataFrame:
    dfe = panel.loc[panel["expiry"] == expiry].copy()
    dfe = dfe.dropna(subset=["Strike"]).sort_values("Strike").reset_index(drop=True)
    return dfe


def _iv_at_strike(dfe: pd.DataFrame, K: float) -> float:
    x = dfe["Strike"].to_numpy(dtype=float)
    y = dfe["IV_Mid"].to_numpy(dtype=float)
    return float(_interp_1d(x, y, np.array([K]))[0])


def _iv_at_delta_signed(dfe: pd.DataFrame, delta_signed: float, F: float) -> float:
    """
    Build IV(Delta_signed) :
      - put side for K<=F (negative delta )
      - call side for K>F (positive delta )
    Puis interpole sur delta_signed.
    """
    puts = dfe.loc[
        (dfe["Strike"] <= F) &
        dfe["Delta_Put"].notna() &
        dfe["ImplV_Put"].notna()
    ].copy()
    puts["d"] = -puts["Delta_Put"].abs()
    puts["iv"] = puts["ImplV_Put"]

    calls = dfe.loc[
        (dfe["Strike"] > F) &
        dfe["Delta_Call"].notna() &
        dfe["ImplV_Call"].notna()
    ].copy()
    calls["d"] = calls["Delta_Call"].abs()
    calls["iv"] = calls["ImplV_Call"]

    curve = pd.concat([puts[["d", "iv"]], calls[["d", "iv"]]], ignore_index=True)
    curve = curve.dropna().sort_values("d")

    x = curve["d"].to_numpy(dtype=float)
    y = curve["iv"].to_numpy(dtype=float)

    return float(_interp_1d(x, y, np.array([delta_signed]))[0])


# =========
# (1) ATM Term Structure
# =========

def compute_atm_term_structure(panel: pd.DataFrame, F_market: dict[str, float]) -> pd.DataFrame:
    rows = []
    for e in sorted(panel["expiry"].unique()):
        if e not in F_market:
            continue
        dfe = _get_smile_by_expiry(panel, e)
        F = float(F_market[e])
        rows.append({
            "expiry": e,
            "F": F,
            "iv_atm": _iv_at_strike(dfe, F),
        })
    return pd.DataFrame(rows)


def plot_atm_term_structure(df_ts: pd.DataFrame, title: str = "ATM Vol Term Structure") -> go.Figure:
    fig = go.Figure()
    if df_ts is None or df_ts.empty:
        return fig

    fig.add_trace(go.Scatter(
        x=df_ts["expiry"],
        y=100.0 * df_ts["iv_atm"],
        mode="lines+markers",
        name="ATM IV"
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Expiry",
        yaxis_title="ATM IV (%)",
        template="plotly_dark"
    )
    return fig


# =========
# (2) Skew Metrics (RR/BF/PutSkew) pour 10Δ et 25Δ
# =========

def compute_skew_metrics(panel: pd.DataFrame, F_market: dict[str, float], deltas=(0.10, 0.25)) -> pd.DataFrame:
    rows = []
    for e in sorted(panel["expiry"].unique()):
        if e not in F_market:
            continue
        F = float(F_market[e])
        dfe = _get_smile_by_expiry(panel, e)

        iv_atm = _iv_at_strike(dfe, F)
        row = {"expiry": e, "F": F, "iv_atm": iv_atm}

        for d in deltas:
            d = float(d)
            k = int(round(d * 100))

            iv_put = _iv_at_delta_signed(dfe, delta_signed=-d, F=F)
            iv_call = _iv_at_delta_signed(dfe, delta_signed=+d, F=F)

            row[f"iv_{k}p"] = iv_put
            row[f"iv_{k}c"] = iv_call
            row[f"rr_{k}"] = iv_call - iv_put
            row[f"bf_{k}"] = 0.5 * (iv_call + iv_put) - iv_atm
            row[f"putskew_{k}"] = iv_put - iv_atm

        rows.append(row)

    return pd.DataFrame(rows)


def plot_skew_bars(df_skew: pd.DataFrame, metric: str = "rr_25", title: str | None = None) -> go.Figure:
    fig = go.Figure()
    if df_skew is None or df_skew.empty:
        return fig

    title = title or f"Skew metric: {metric}"

    fig.add_trace(go.Bar(
        x=df_skew["expiry"],
        y=100.0 * df_skew[metric],  # vol points
        name=metric
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Expiry",
        yaxis_title="Vol points (%)",
        template="plotly_dark"
    )
    return fig


# =========
# (3) Smile slope / curvature autour de l'ATM (via 25Δ ou 10Δ)
# =========

def compute_smile_shape_metrics(df_skew: pd.DataFrame, d: float = 0.25) -> pd.DataFrame:
    if df_skew is None or df_skew.empty:
        return df_skew

    out = df_skew.copy()
    k = int(round(d * 100))

    iv_atm = out["iv_atm"]
    iv_p = out[f"iv_{k}p"]
    iv_c = out[f"iv_{k}c"]

    out[f"slope_left_{k}"] = (iv_atm - iv_p) / d
    out[f"slope_right_{k}"] = (iv_c - iv_atm) / d
    out[f"curvature_{k}"] = (iv_c + iv_p - 2.0 * iv_atm)

    return out


def plot_shape_lines(df_shape: pd.DataFrame, cols: list[str], title: str) -> go.Figure:
    fig = go.Figure()
    if df_shape is None or df_shape.empty:
        return fig

    for c in cols:
        fig.add_trace(go.Scatter(
            x=df_shape["expiry"],
            y=100.0 * df_shape[c],
            mode="lines+markers",
            name=c
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Expiry",
        yaxis_title="(x100) units",
        template="plotly_dark"
    )
    return fig


# =========
# (5) Vol Mispricing map vs ATM (IV(K) - IV_ATM)
# =========

def compute_vol_mispricing_map(panel: pd.DataFrame, F_market: dict[str, float]) -> pd.DataFrame:
    rows = []
    for e in sorted(panel["expiry"].unique()):
        if e not in F_market:
            continue
        F = float(F_market[e])
        dfe = _get_smile_by_expiry(panel, e).dropna(subset=["Strike", "IV_Mid"]).copy()

        iv_atm = _iv_at_strike(dfe, F)
        dfe["iv_atm"] = iv_atm
        dfe["dv"] = dfe["IV_Mid"] - iv_atm
        dfe["F"] = F
        dfe["expiry"] = e

        rows.append(dfe[["expiry", "Strike", "F", "IV_Mid", "iv_atm", "dv"]])

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_vol_mispricing_heatmap(df_map: pd.DataFrame, title: str = "Vol Mispricing Map (IV - ATM)") -> go.Figure:
    fig = go.Figure()
    if df_map is None or df_map.empty:
        return fig

    piv = df_map.pivot_table(index="expiry", columns="Strike", values="dv", aggfunc="mean").sort_index()
    x = piv.columns.to_numpy(dtype=float)
    y = piv.index.to_list()
    z = 100.0 * piv.to_numpy(dtype=float)

    fig = go.Figure(data=go.Heatmap(
        x=x, y=y, z=z,
        colorbar=dict(title="Vol pts"),
        hovertemplate="Expiry=%{y}<br>Strike=%{x}<br>ΔIV=%{z:.2f} vol pts<extra></extra>"
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Strike",
        yaxis_title="Expiry",
        template="plotly_dark"
    )
    return fig
