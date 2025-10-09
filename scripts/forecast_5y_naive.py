#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_months(arg: str | None):
    if not arg:
        return None
    if "-" in arg:
        a, b = arg.split("-", 1)
        return set(range(int(a), int(b) + 1))
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
        if not {"category", "price"}.issubset(pm.columns):
            raise ValueError("price-map doit contenir: category, price")
        df = df.merge(pm[["category", "price"]], on="category", how="left")
        df["price"] = df["price"].fillna(0.0)
        df["value"] = df["qty"].astype(float) * df["price"].astype(float)
    else:
        df["value"] = df["qty"].astype(float)
    s = df.groupby(pd.Grouper(key="timestamp", freq="MS"))["value"].sum().sort_index()
    s = s[s.notna()]
    if len(s) == 0:
        raise ValueError("Série vide après agrégation.")
    start = pd.Timestamp(s.index.min())
    end = pd.Timestamp(s.index.max())
    full_idx = pd.date_range(start=start, end=end, freq="MS")
    s = s.reindex(full_idx).fillna(0.0)
    s.name = "value"
    return s

def seasonal_naive_forecast(y: pd.Series, steps: int = 60) -> pd.Series:
    df = pd.DataFrame({"y": y})
    df["month"] = df.index.month
    month_means = df.groupby("month")["y"].mean()
    last = y.index[-1]
    future_idx = pd.date_range(last + pd.offsets.MonthBegin(), periods=steps, freq="MS")
    preds = [month_means.get(ts.month, y.mean()) for ts in future_idx]
    return pd.Series(preds, index=future_idx, name="forecast")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True, type=Path)
    ap.add_argument("--sheet", default="monthly")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--price-map", type=Path, default=None)
    ap.add_argument("--months", type=str, default=None)
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    months_keep = parse_months(args.months)
    y = load_monthly_series(args.excel, args.sheet, args.price_map, months_keep)
    fc = seasonal_naive_forecast(y, steps=60)
    monthly = pd.DataFrame({"forecast": fc})
    monthly_path = outdir / "monthly_forecast.csv"
    monthly.to_csv(monthly_path, index_label="timestamp", encoding="utf-8-sig")
    tmp = monthly.copy()
    tmp["year"] = tmp.index.to_period("M").strftime("%Y")
    yearly = tmp.groupby("year")["forecast"].sum().head(5)
    yearly_path = outdir / "yearly_forecast.csv"
    yearly.to_csv(yearly_path, header=["forecast_sum"], encoding="utf-8-sig")
    plt.figure(figsize=(10, 5))
    y.plot(label="history")
    fc.plot(label="forecast")
    plt.legend()
    plt.title("Historique vs Prévision (Naïf saisonnier, 60 mois)")
    plt.xlabel("Temps")
    plt.ylabel("Valeur mensuelle (qty ou amount)")
    plot_path = outdir / "forecast_plot.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] Monthly forecast ->", monthly_path)
    print("[OK] Yearly forecast  ->", yearly_path)
    print("[OK] Plot            ->", plot_path)

if __name__ == "__main__":
    main()
