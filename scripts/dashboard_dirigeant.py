import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


TS_CANDIDATES = ["timestamp", "datetime", "date", "datum"]
CAT_CANDIDATES = ["category", "categorie", "code_atc", "atc", "famille", "family"]
QTY_CANDIDATES = ["qty", "quantite", "quantity", "qte", "qtty", "quantité"]

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    raise ValueError(
        f"Impossible de trouver une colonne parmi {candidates}. "
        f"Colonnes disponibles: {df.columns.tolist()}"
    )

def _read_any(input_path: Path, sheet_name: str | None) -> pd.DataFrame:
    suffix = input_path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        try:
            return pd.read_csv(input_path)
        except UnicodeDecodeError:
            return pd.read_csv(input_path, encoding="latin-1", sep=";")
    else:
  
        return pd.read_excel(input_path, sheet_name=sheet_name or 0)

def load_jan_sep(input_path: Path, sheet_name: str | None = None) -> pd.DataFrame:
    """
    Charge le fichier (CSV ou Excel), détecte les colonnes datetime/category/qty,
    nettoie, et filtre JAN→SEP (mois 1..9) pour TOUTES les années.
    """
    df = _read_any(input_path, sheet_name)

    ts_col = _find_col(df, TS_CANDIDATES)
    cat_col = _find_col(df, CAT_CANDIDATES)
    qty_col = _find_col(df, QTY_CANDIDATES)

    df = df.rename(columns={ts_col: "timestamp", cat_col: "category", qty_col: "qty"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=False)
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")

    df = df.dropna(subset=["timestamp", "qty"])
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month

    df = df[df["month"].between(1, 9)]

    df["category"] = df["category"].astype(str)

    return df[["timestamp", "year", "month", "category", "qty"]].copy()

def apply_price_map(df: pd.DataFrame, price_map_path: Path | None) -> pd.DataFrame:
    """
    Si price_map fourni (colonnes: category, price), calcule 'amount' = qty * price.
    Sinon, 'amount' = qty (proxy de CA).
    """
    df = df.copy()
    if price_map_path is None:
        df["amount"] = df["qty"].astype(float)
        return df

    prices = _read_any(price_map_path, None)
    if "category" not in prices.columns:
        for alt in ["Code_ATC", "code_atc", "categorie", "Category"]:
            if alt in prices.columns:
                prices = prices.rename(columns={alt: "category"})
                break

    req_cols = {"category", "price"}
    if not req_cols.issubset(prices.columns):
        raise ValueError(
            f"Le fichier prix doit contenir les colonnes {req_cols}. "
            f"Colonnes présentes: {prices.columns.tolist()}"
        )

    prices["category"] = prices["category"].astype(str)
    df["category"] = df["category"].astype(str)

    merged = df.merge(prices[["category", "price"]], on="category", how="left")
    if merged["price"].isna().any():
        missing_cats = merged.loc[merged["price"].isna(), "category"].unique()[:20]
        print(f"[AVERTISSEMENT] Prix manquants pour certaines catégories (extrait): {missing_cats}")
        merged["price"] = merged["price"].fillna(0.0)

    merged["amount"] = merged["qty"].astype(float) * merged["price"].astype(float)
    return merged

def compute_annual_kpis(df: pd.DataFrame) -> pd.DataFrame:
    kpi = (
        df.groupby("year")
          .agg(CA=("amount", "sum"),
               Volume=("qty", "sum"))
          .reset_index()
          .sort_values("year")
    )
    kpi["Growth_%"] = kpi["CA"].pct_change() * 100
    kpi["Avg_Basket_proxy"] = kpi["CA"] / kpi["Volume"].replace(0, np.nan)
    return kpi

def compute_category_shares(df: pd.DataFrame) -> pd.DataFrame:
    by_cat = df.groupby("category")["amount"].sum().sort_values(ascending=False)
    share = (by_cat / by_cat.sum() * 100).reset_index()
    share.columns = ["category", "Share_%"]
    return share

def compute_top_categories(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    by_cat_qty = df.groupby("category")["qty"].sum().sort_values(ascending=False).head(n)
    topn = by_cat_qty.reset_index()
    topn.columns = ["category", "Total_qty"]
    return topn

def plot_pie_category_amount(df: pd.DataFrame, outpath: Path) -> Path:
    pie_data = df.groupby("category")["amount"].sum().sort_values(ascending=False)
    plt.figure(figsize=(7,7))
    pie_data.plot.pie(autopct="%1.1f%%", startangle=90)
    plt.title("Répartition des ventes (jan–sep) par grande famille thérapeutique")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    return outpath

def plot_annual_trend(kpi: pd.DataFrame, outpath: Path) -> Path:
    plt.figure(figsize=(8,5))
    plt.plot(kpi["year"], kpi["CA"], marker="o", linewidth=2)
    for x, y in zip(kpi["year"], kpi["CA"]):
        try:
            label = f"{int(round(y)):,}".replace(",", " ")
        except Exception:
            label = f"{y:.0f}"
        plt.text(x, y, label, va="bottom", ha="center", fontsize=9)
    plt.title("Évolution des Ventes mensuelles (jan–sep) par année")
    plt.xlabel("Année")
    plt.ylabel("Ventes (jan–sep)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    return outpath

def compute_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    season = df.groupby(["category", "month"])["qty"].sum().reset_index()
    season["cat_mean"] = season.groupby("category")["qty"].transform("mean")
    season["season_index"] = (season["qty"] / season["cat_mean"].replace(0, np.nan)) * 100
    return season

def generate_recommendations(kpi: pd.DataFrame, season: pd.DataFrame, df: pd.DataFrame, top_decliners_n: int = 3) -> list[str]:
    recos = []
    if len(kpi) >= 2 and pd.notna(kpi["Growth_%"].iloc[-1]):
        g = kpi["Growth_%"].iloc[-1]
        y = int(kpi["year"].iloc[-1])
        if g > 5:
            recos.append(f"Le CA (jan–sep) progresse de {g:.1f}% en {y} : renforcer la capacité logistique et anticiper les stocks.")
        elif g < -5:
            recos.append(f"Le CA (jan–sep) recule de {g:.1f}% en {y} : analyser le mix catégories et activer des campagnes ciblées.")
        else:
            recos.append(f"CA (jan–sep) global stable en {y} : maintenir la stratégie et optimiser les coûts.")

    winter = season[season["month"].isin([1, 2])]
    winter_peaks = (winter.sort_values(["category", "season_index"], ascending=[True, False])
                         .groupby("category")
                         .head(1))
    respiratory_like = [c for c in winter_peaks["category"].unique()
                        if any(k in str(c).lower() for k in ["gripp", "resp", "rhume", "toux"])]
    if respiratory_like:
        recos.append("Renforcer les effectifs en hiver (jan–fév) pour gérer le pic respiratoire détecté (jan–sep).")

    spring = season[season["month"].isin([3, 4, 5])]
    spring_peaks = (spring.sort_values(["category", "season_index"], ascending=[True, False])
                         .groupby("category")
                         .head(1))
    pain_like = [c for c in spring_peaks["category"].unique()
                 if any(k in str(c).lower() for k in ["alg", "anti-inflamm", "douleur", "ibup", "aspir", "parac"])]
    if pain_like:
        recos.append("Optimiser le stock des antalgiques au printemps (mar–mai).")

    cat_year = df.groupby([df["timestamp"].dt.year, "category"])["qty"].sum().reset_index()
    cat_year.columns = ["year", "category", "qty"]
    if cat_year["year"].nunique() >= 2:
        years = sorted(cat_year["year"].unique())
        a = cat_year[cat_year["year"] == years[-2]].set_index("category")["qty"]
        b = cat_year[cat_year["year"] == years[-1]].set_index("category")["qty"]
        common = a.index.intersection(b.index)
        delta = (b[common] - a[common]) / a[common].replace(0, np.nan) * 100
        declining = delta[delta < -5].sort_values().head(top_decliners_n)
        if not declining.empty:
            recos.append(f"Prioriser des actions sur les catégories en baisse (jan–sep) : {', '.join(declining.index.tolist())}.")
    return recos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path,
                    help="Chemin du CSV/XLSX source (ex: Bonexcel.xlsx ou export.csv)")
    ap.add_argument("--sheet", default=None,
                    help="Nom/Index de feuille si Excel (par défaut: 1ère feuille)")
    ap.add_argument("--outdir", default="out", help="Dossier de sortie")
    ap.add_argument("--price-map", type=Path, default=None,
                    help="CSV/XLSX des prix (colonnes: category, price). Optionnel.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_jan_sep(args.input, sheet_name=args.sheet)
    df = apply_price_map(df, args.price_map)


    kpi = compute_annual_kpis(df)
    kpi_path = outdir / "kpis_annuels_jan_sep.csv"
    kpi.to_csv(kpi_path, index=False, encoding="utf-8-sig")


    shares = compute_category_shares(df)
    shares_path = outdir / "parts_de_marche_par_categorie_jan_sep.csv"
    shares.to_csv(shares_path, index=False, encoding="utf-8-sig")

    top3 = compute_top_categories(df, n=3)
    top3_path = outdir / "top3_categories_jan_sep.csv"
    top3.to_csv(top3_path, index=False, encoding="utf-8-sig")

    pie_path = outdir / "pie_ca_par_famille_jan_sep.png"
    plot_pie_category_amount(df, pie_path)

    trend_path = outdir / "ca_trend_annuel_jan_sep.png"
    plot_annual_trend(kpi, trend_path)


    season = compute_seasonality(df)
    recos = generate_recommendations(kpi, season, df)
    reco_path = outdir / "recommandations_strategiques_jan_sep.txt"
    reco_path.write_text("Recommandations stratégiques (jan–sep, générées automatiquement)\n\n", encoding="utf-8")
    for r in recos:
        prev = reco_path.read_text(encoding="utf-8")
        reco_path.write_text(prev + f"- {r}\n", encoding="utf-8")

    print(f"[OK] KPIs annuels (jan–sep) -> {kpi_path}")
    print(f"[OK] Parts de marché (jan–sep) -> {shares_path}")
    print(f"[OK] Top 3 catégories (jan–sep) -> {top3_path}")
    print(f"[OK] Camembert (jan–sep) -> {pie_path}")
    print(f"[OK] Courbe tendance (jan–sep) -> {trend_path}")
    print(f"[OK] Recommandations (jan–sep) -> {reco_path}")

if __name__ == "__main__":
    main()
