# winrate_facets_delta.py
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.cluster import KMeans
from pathlib import Path
import numpy as np

CSV = "bpb_out/relative_winrate_by_round.csv"     # Pfad bei dir anpassen, falls nötig
OUT_DIR = Path("bpb_out/winrate_facets")
OUT_DIR.mkdir(exist_ok=True)


# ---------- Load & filter ----------
df = pd.read_csv(CSV)
# sinnvolle Stichprobe und Rundenbereich
df = df[(df["n_reached"] >= 50) & (df["round"].between(1, 18))].copy()

# Wir nehmen DELTA (mit minus ohne), das steht schon als Spalte drin
# Delta ist in [−1, +1], wir plotten als %
df = df[np.isfinite(df["delta_winrate"])]

# ---------- Wide pivot: round × item → delta ----------
pivot = df.pivot_table(index="round",
                       columns="item_name",
                       values="delta_winrate",
                       aggfunc="mean")

# Mindestens ein paar Punkte pro Item behalten und Lücken füllen
pivot = pivot.dropna(axis=1, thresh=5)
pivot = pivot.sort_index().interpolate(limit_direction="both").bfill().ffill()

# Wenn nach Filter nix übrig, lieber sauber aussteigen
if pivot.shape[1] == 0:
    raise SystemExit("Keine Items mit ausreichender Abdeckung nach Filter. Erhöhe n_reached-Filter oder prüfe CSV.")

# ---------- Clustering auf Kurvenform ----------
n_items = pivot.shape[1]
# Grobe Heuristik: ~10 Items pro Cluster, 2–10 Cluster
n_clusters = int(np.clip(round(n_items / 10), 2, 10))
km = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
labels = km.fit_predict(pivot.T.values)

curve_df = pivot.T.copy()
curve_df["cluster"] = labels

even_rounds = [r for r in pivot.index if r % 2 == 0]  # nur gerade Runden auf der Achse

# ---------- Facets: 10 Items pro Cluster ----------
for c in sorted(curve_df["cluster"].unique()):
    subset = curve_df[curve_df["cluster"] == c].drop(columns=["cluster"])
    if subset.empty:
        continue
    # Repräsentative Auswahl: hohe Varianz bevorzugen, sonst random fallback
    variances = subset.var(axis=1).sort_values(ascending=False)
    pick = variances.index[:min(10, len(variances))]
    sample = subset.loc[pick]

    # Y-Limits mit etwas Puffer
    y_min = float(sample.min().min())
    y_max = float(sample.max().max())
    pad = max(0.02, (y_max - y_min) * 0.15)
    lo, hi = y_min - pad, y_max + pad
    # clamp, damit es nicht völlig ausrastet
    lo = max(-1.0, lo)
    hi = min( 1.0, hi)

    plt.figure(figsize=(9, 5.5))
    for item_name, row in sample.iterrows():
        plt.plot(pivot.index,
                 row.values,
                 marker="o", markersize=3, linewidth=1,
                 label=item_name, alpha=0.9)

    plt.title(f"Cluster {c} — Δ-Winrate vs. ohne Item (10 Items)")
    plt.xlabel("Round")
    plt.ylabel("Δ-Winrate")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    plt.ylim(lo, hi)
    plt.xticks(even_rounds)
    plt.grid(True, axis="y", alpha=0.25)
    # Legende schlank, sonst Rand überfüllt
    plt.legend(fontsize=7, loc="best", frameon=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"delta_winrate_cluster_{c}.png", dpi=150)
    plt.close()

print(f"Fertig. PNGs liegen in: {OUT_DIR.resolve()}")
