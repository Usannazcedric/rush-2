# Retry writing the forecasting script to disk
from pathlib import Path

script_path = Path("/mnt/data/forecast_5y.py")
code = r''''''

import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def parse_months(arg: str | None):
    if not arg:
        return None
    if "-" in arg:
        a, b = arg.split("-", 1)
        return set(range(int(a), int(b)+1))
    else:
        return set(int(x.strip()) for x in arg.split(",") if x.strip())

def load_monthly_series(excel: Path, sheet: str, price_map: Path | None, months_keep: set[int] | None) -> pd.Series:
    df = pd.read_excel(excel, sheet_name=sheet)
    needed = {"timestamp", "category", "qty"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Colonnes requises manquantes dans '{sheet}'. Trouvé: {df.columns.tolist()}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "qty"])
    df["month"] = df["timestamp"].dt.month

    if months_keep is not None:
        df = df[df["month"].isin(months_keep)]

    if price_map is not None:
        pm = pd.read_csv(price_map)
        if not {"category","price"}.issubset(pm.columns):
            raise ValueError("price-map doit contenir les colonnes: category, price")
        df = df.merge(pm[["category","price"]], on="category", how="left")
        df["price"] = df["price"].fillna(0.0)
        df["amount"] = df["qty"].astype(float) * df["price"].astype(float)
        value_col = "amount"
    else:
        value_col = "qty"

    s = (
        df.groupby(pd.Grouper(key="timestamp", freq="MS"))[value_col]
          .sum()
          .sort_index()
    )
    s = s[s.notna()]
    full_idx = pd.period_range(s.index.min().to_period("M").to_timestamp(),
                               s.index.max().to_period("M").to_timestamp(),
                               freq="MS").to_timestamp()
    s = s.reindex(full_idx).fillna(0.0)
    s.name = value_col
    return s

def adf_pvalue(series: pd.Series) -> float:
    if series.isna().any():
        series = series.dropna()
    if len(series) < 12:
        return np.nan
    try:
        return adfuller(series, autolag="AIC")[1]
    except Exception:
        return np.nan

def try_sarima(y: pd.Series, seasonal_periods: int = 12):
    grid = [(p,d,q,P,D,Q) for p in (0,1,2) for d in (0,1) for q in (0,1,2)
                          for P in (0,1)  for D in (0,1) for Q in (0,1)]
    best_aic = np.inf
    best_res = None
    best_order = None
    for (p,d,q,P,D,Q) in grid:
        try:
            model = SARIMAX(y, order=(p,d,q), seasonal_order=(P,D,Q,seasonal_periods),
                            enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False)
            if res.aic < best_aic:
                best_aic = res.aic
                best_res = res
                best_order = ((p,d,q),(P,D,Q,seasonal_periods))
        except Exception:
            continue
    return best_res, best_order, best_aic

def holt_winters(y: pd.Series, seasonal_periods: int = 12):
    if seasonal_periods and len(y) >= 2*seasonal_periods:
        model = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=seasonal_periods, damped_trend=True)
    else:
        model = ExponentialSmoothing(y, trend="add", seasonal=None, damped_trend=True)
    res = model.fit(optimized=True)
    return res

def forecast_next_60_months(y: pd.Series, outdir: Path):
    report_lines = []

    pv = adf_pvalue(y)
    report_lines.append(f"ADF p-value (stationnarity): {pv:.4f}" if not np.isnan(pv) else "ADF p-value: N/A")

    use_sarima = (len(y) >= 24)
    res = None
    method = None

    if use_sarima:
        res, order, aic = try_sarima(y, seasonal_periods=12)
        if res is not None:
            method = f"SARIMA order={order[0]}, seasonal_order={order[1]}, AIC={aic:.2f}"
            report_lines.append(f"Chosen model: {method}")
        else:
            use_sarima = False

    if not use_sarima:
        hw = holt_winters(y, seasonal_periods=12)
        res = hw
        method = "Holt-Winters (additif, damped)"
        report_lines.append(f"Chosen model: {method}")

    steps = 60
    if hasattr(res, "get_forecast"):
        fc = res.get_forecast(steps=steps)
        mean = fc.predicted_mean
        conf = fc.conf_int(alpha=0.05)
        lower = conf.iloc[:,0]
        upper = conf.iloc[:,1]
    else:
        mean = res.forecast(steps)
        lower = pd.Series([np.nan]*steps, index=mean.index)
        upper = pd.Series([np.nan]*steps, index=mean.index)

    hist = y.rename("history")
    df_fc = pd.DataFrame({
        "forecast": mean,
        "lower_95": lower,
        "upper_95": upper
    })
    monthly_path = outdir / "monthly_forecast.csv"
    df_fc.to_csv(monthly_path, index_label="timestamp", encoding="utf-8-sig")

    future_years = mean.index.to_period("M").strftime("%Y")
    tmp = pd.DataFrame({"year": future_years, "value": mean.values})
    yearly = tmp.groupby("year")["value"].sum().head(5)
    yearly_path = outdir / "yearly_forecast.csv"
    yearly.to_csv(yearly_path, header=["forecast_sum"], encoding="utf-8-sig")

    plt.figure(figsize=(10,5))
    hist.plot()
    mean.plot()
    plt.title("Historique vs Prévision (60 mois)")
    plt.xlabel("Temps")
    plt.ylabel("Valeur mensuelle (qty ou amount)")
    plt.tight_layout()
    plot_path = outdir / "forecast_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    report_path = outdir / "model_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    return str(monthly_path), str(yearly_path), str(plot_path), str(report_path), "\n".join(report_lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True, type=Path, help="Chemin du fichier Excel")
    ap.add_argument("--sheet", default="monthly", help="Nom de la feuille (par défaut: monthly)")
    ap.add_argument("--outdir", default="out", help="Dossier de sortie")
    ap.add_argument("--price-map", type=Path, default=None, help="CSV prix (category,price)")
    ap.add_argument("--months", type=str, default=None, help='Restreindre aux mois, ex: "1-9" ou "1,2,3"')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    months_keep = parse_months(args.months)

    y = load_monthly_series(args.excel, args.sheet, args.price_map, months_keep)
    m_path, y_path, p_path, r_path, report = forecast_next_60_months(y, outdir)

    print("[OK] Monthly forecast ->", m_path)
    print("[OK] Yearly forecast  ->", y_path)
    print("[OK] Plot            ->", p_path)
    print("[OK] Report          ->", r_path)
    print(report)

if __name__ == "__main__":
    main()

