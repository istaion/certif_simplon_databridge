"""Script temporaire : calcul du MAPE pour les prédictions Ensemble sur les 3 derniers mois.

Sortie :
  - MAPE global
  - MAPE par bucket horizon (≤7j, 8-14j, 15-21j, >21j)
  - MAPE par site (UAI)
  - Export CSV : pred_stats/ensemble_mape.csv
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ensemble_from_store import _HORIZON_BOUNDS, _BIN_LABELS

# ── 1. Charger les prédictions Ensemble ───────────────────────────────────────

CSV_PATH = "pred_stats/ensemble_predictions.csv"
print(f"Chargement de {CSV_PATH}…")
df = pd.read_csv(CSV_PATH)

df["target_date"] = pd.to_datetime(df["target_date"])
df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
df["effectif_reel"] = pd.to_numeric(df["effectif_reel"], errors="coerce")
df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce")

# ── Diagnostic ────────────────────────────────────────────────────────────────

print(f"  {len(df)} lignes chargées")
print(f"  target_date : {df['target_date'].min()} → {df['target_date'].max()}")
print(f"  effectif_reel : {df['effectif_reel'].notna().sum()} non-null, {(df['effectif_reel'] == 0).sum()} zéros")

# ── 2. Filtrer les 2 derniers mois avec effectif_reel connu ───────────────────

df_with_reel = df[df["effectif_reel"].notna() & (df["effectif_reel"] != 0)]
if df_with_reel.empty:
    print("\n⚠ Aucune ligne avec effectif_reel renseigné et > 0. Rien à calculer.")
    sys.exit(0)

cutoff = df_with_reel["target_date"].max() - pd.DateOffset(months=2)
df = df_with_reel[df_with_reel["target_date"] >= cutoff]
print(f"  {len(df)} lignes après filtre (3 derniers mois, effectif_reel > 0)")
print(f"  Période : {df['target_date'].min().date()} → {df['target_date'].max().date()}")

# Bucket horizon
df["bin_idx"] = np.digitize(df["horizon"].values, bins=_HORIZON_BOUNDS)
df["horizon_bin"] = [_BIN_LABELS[b] if b < len(_BIN_LABELS) else f"bin{b}" for b in df["bin_idx"]]


# ── 3. Fonction MAPE ─────────────────────────────────────────────────────────

def mape(actual: pd.Series, predicted: pd.Series) -> float | None:
    mask = actual != 0
    if mask.sum() == 0:
        return None
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


# ── 4. MAPE global ───────────────────────────────────────────────────────────

mape_global = mape(df["effectif_reel"], df["prediction"])
print(f"\n{'='*60}")
print(f"MAPE GLOBAL : {mape_global:.2f}%  (N={len(df)})")
print(f"{'='*60}")

# ── 5. MAPE par horizon ──────────────────────────────────────────────────────

print(f"\nMAPE PAR HORIZON :")
print(f"{'-'*40}")
horizon_rows = []
for bin_label in _BIN_LABELS:
    sub = df[df["horizon_bin"] == bin_label]
    m = mape(sub["effectif_reel"], sub["prediction"])
    horizon_rows.append({"horizon_bin": bin_label, "MAPE": m, "N": len(sub)})
    print(f"  {bin_label:>8s} : {m:6.2f}%  (N={len(sub)})" if m is not None else f"  {bin_label:>8s} : —  (N={len(sub)})")

# ── 6. MAPE par site ─────────────────────────────────────────────────────────

print(f"\nMAPE PAR SITE :")
print(f"{'-'*40}")
site_rows = []
for uai in sorted(df["uai"].unique()):
    sub = df[df["uai"] == uai]
    m = mape(sub["effectif_reel"], sub["prediction"])
    site_rows.append({"uai": uai, "MAPE": m, "N": len(sub)})

site_df = pd.DataFrame(site_rows).sort_values("MAPE", ascending=True, na_position="last")
for _, row in site_df.iterrows():
    m_str = f"{row['MAPE']:6.2f}%" if row["MAPE"] is not None else "     —"
    print(f"  {row['uai']} : {m_str}  (N={int(row['N'])})")

# ── 7. MAPE par (site × horizon) ─────────────────────────────────────────────

print(f"\nMAPE PAR SITE × HORIZON :")
print(f"{'-'*60}")
site_horizon_rows = []
for uai in sorted(df["uai"].unique()):
    for bin_label in _BIN_LABELS:
        sub = df[(df["uai"] == uai) & (df["horizon_bin"] == bin_label)]
        m = mape(sub["effectif_reel"], sub["prediction"])
        site_horizon_rows.append({"uai": uai, "horizon_bin": bin_label, "MAPE": m, "N": len(sub)})

sh_df = pd.DataFrame(site_horizon_rows)
# Pivot pour affichage compact
pivot = sh_df.pivot(index="uai", columns="horizon_bin", values="MAPE")
pivot = pivot.reindex(columns=_BIN_LABELS)
print(pivot.to_string(float_format="%.2f", na_rep="—"))

# ── 8. Export CSV ─────────────────────────────────────────────────────────────

out_dir = "pred_stats"
os.makedirs(out_dir, exist_ok=True)

# Global + horizon
summary = pd.DataFrame(horizon_rows)
summary.loc[len(summary)] = {"horizon_bin": "GLOBAL", "MAPE": mape_global, "N": len(df)}
summary.to_csv(f"{out_dir}/ensemble_mape_horizon.csv", index=False)

# Par site
site_df.to_csv(f"{out_dir}/ensemble_mape_site.csv", index=False)

# Par (site × horizon)
sh_df.to_csv(f"{out_dir}/ensemble_mape_site_horizon.csv", index=False)

print(f"\nCSV exportés dans {out_dir}/ :")
print(f"  ensemble_mape_horizon.csv")
print(f"  ensemble_mape_site.csv")
print(f"  ensemble_mape_site_horizon.csv")
