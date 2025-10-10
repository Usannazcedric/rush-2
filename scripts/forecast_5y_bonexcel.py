#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib.dates as mdates
from pathlib import Path

EXCEL_PATH = "/Users/drikce/Desktop/piscine 4eme anneee/rush-2/data/Bonexcel.xlsx"
OUTDIR = "out_bonexcel_clean"
MIN_ANNUAL_GROWTH = 0.05  # +5%/an minimum

def read_and_prepare(path):
    xls = pd.ExcelFile(path)
    dfs = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        df.columns = [c.lower().strip() for c in df.columns]
        date_col = next((c for c in df.columns if "date" in c or "time" in c), None)
        val_col  = next((c for c in df.columns if any(k in c for k in ["amount","montant","ca","sales","value","revenu"])), None)
        qty_col  = next((c for c in df.columns if any(k in c for k in ["qty","quant","volume","units"])), None)
        if date_col and (val_col or qty_col):
            df["timestamp"] = pd.to_datetime(df[date_col], errors="coerce")
            df["amount"] = pd.to_numeric(df[val_col], errors="coerce") if val_col else pd.to_numeric(df[qty_col], errors="coerce")
            df = df.dropna(subset=["timestamp","amount"])
            dfs.append(df[["timestamp","amount"]])
    full = pd.concat(dfs, ignore_index=True)
    y = full.groupby(pd.Grouper(key="timestamp", freq="MS"))["amount"].sum().sort_index()
    y = y.reindex(pd.date_range(y.index.min(), y.index.max(), freq="MS")).fillna(0.0)
    y = y.clip(lower=0)
    return y

def positive_forecast(y, months=60, floor_ratio=0.8, min_annual_growth=0.05):
    y_s = y.rolling(3, min_periods=1, center=True).mean()
    last_level = y_s.iloc[-1]
    recent = y.tail(24)
    base_level = max(last_level, np.median(recent)) if len(recent)>0 else last_level
    by_month = y.groupby(y.index.month).mean()
    season = by_month / by_month.mean() if by_month.mean() > 0 else by_month*0+1.0
    if len(y) >= 24:
        last12 = y.tail(12).mean()
        prev12 = y.tail(24).head(12).mean()
        cagr = (last12/prev12 - 1.0) if (prev12 and prev12>0) else 0.0
    else:
        cagr = 0.0
    cagr = max(float(cagr), float(min_annual_growth))
    idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(), periods=months, freq="MS")
    fvals = []
    for i, d in enumerate(idx, start=1):
        growth = (1.0 + cagr) ** (i/12.0)
        seas = season.get(d.month, 1.0)
        val = base_level * seas * growth
        val = max(val, base_level * floor_ratio, 1e-6)
        fvals.append(val)
    return pd.Series(fvals, index=idx, name="forecast")

def main():
    y = read_and_prepare(EXCEL_PATH)
    fc = positive_forecast(y, 60, floor_ratio=0.8, min_annual_growth=MIN_ANNUAL_GROWTH)
    outdir = Path(OUTDIR); outdir.mkdir(exist_ok=True)
    pd.DataFrame({"forecast": fc}).to_csv(outdir/"monthly_forecast.csv", index_label="timestamp", encoding="utf-8-sig")
    fc.groupby(fc.index.year).sum().head(5).to_csv(outdir/"yearly_forecast.csv", header=["forecast_sum"], encoding="utf-8-sig")
    plt.figure(figsize=(12,6))
    plt.plot(y.index, y, label="Historique", linewidth=2)
    plt.plot(fc.index, fc, label=f"Prévision 5 ans (≥ {int(MIN_ANNUAL_GROWTH*100)}%/an)", linewidth=2, color="orange")
    plt.legend(); plt.title("Prévision 5 ans ")
    plt.xlabel("Mois"); plt.ylabel("Ventes mensuelle")
    plt.grid(alpha=0.3); plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %Y")); plt.xticks(rotation=45)
    plt.tight_layout(); plt.savefig(outdir/"forecast_plot.png", dpi=150)
    print("OK:", outdir)
if __name__ == "__main__":
    main()
