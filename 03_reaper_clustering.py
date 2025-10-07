#!/usr/bin/env python3
"""
Reaper-only Final-Builds + Clustering (IDE One-Shot, filter fail-open).
Wenn CLASS_REGEX None ist, wird NICHT gefiltert.
Wenn CLASS_REGEX gesetzt ist, aber keine Matches findet, wird eine Warnung geloggt und OHNE Filter weitergemacht.
Outputs: bpb_out/reaper_clusters_*.csv
  - reaper_clusters_summary.csv
  - reaper_clusters_core_items.csv (Lift- & Frequenz-Listen)
  - reaper_clusters_item_stats.csv (Detailmetriken je Item x Cluster)
"""
import os
import duckdb
import re
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

# ============== CONFIG =================
DB_PATH        = os.environ.get("BPB_DB", "bpb_out/bpb.duckdb")
OUT_DIR        = "bpb_out"
K              = 8
MIN_ITEM_FREQ  = 15
MAX_ITEMS      = 300
# Core-Item-Heuristik (Items, die wirklich cluster-typisch sind)
CORE_TOP_K             = 8
CORE_MIN_CLUSTER_RATE  = 0.30   # Anteil der Matches im Cluster, die das Item haben
CORE_MIN_LIFT          = 1.20   # Wie viel häufiger ggü. Gesamtmeta?
CORE_MIN_COUNT         = 10     # absolute Häufigkeit im Cluster
STAPLE_MAX_GLOBAL_RATE = 0.65   # filtert "Staples", die fast überall sind
# Setze auf None, wenn du nicht filtern willst.
CLASS_REGEX    = None  # z.B. r"reaper"  -> filtert nur Matches, die irgendwo ein Item mit 'reaper' enthalten
# =======================================

os.makedirs(OUT_DIR, exist_ok=True)
con = duckdb.connect(DB_PATH)
con.execute("SET schema='battles';")

def fetch_final_items(con, class_regex):
    # letzte Runde je Match inkl. Ergebnis
    last_round_df = con.execute("""
        WITH last_round AS (
          SELECT match_id, MAX(round_index) AS last_r
          FROM battles.rounds
          GROUP BY match_id
        )
        SELECT lr.match_id, lr.last_r, r.result
        FROM last_round lr
        JOIN battles.rounds r
          ON r.match_id = lr.match_id AND r.round_index = lr.last_r
    """).df()

    # Items in der finalen Runde
    items_df = con.execute("""
        WITH last_round AS (
          SELECT match_id, MAX(round_index) AS last_r
          FROM battles.rounds
          GROUP BY match_id
        )
        SELECT ri.match_id, ri.item_name
        FROM battles.round_items ri
        JOIN last_round lr USING (match_id)
        WHERE ri.round_index = lr.last_r
    """).df()

    # Optionaler Klassenfilter: irgendein Item des Matches matcht regex
    if class_regex:
        pattern = re.compile(class_regex, re.IGNORECASE)
        # Alle Items je Match (irgendeine Runde)
        any_items = con.execute("""
            SELECT DISTINCT match_id, item_name
            FROM battles.round_items
        """).df()
        reaper_matches = set(
            any_items[any_items["item_name"].apply(lambda s: bool(pattern.search(str(s))))]["match_id"].unique()
        )

        if len(reaper_matches) == 0:
            print(f"[WARN] CLASS_REGEX '{class_regex}' hat 0 Matches. Fahre OHNE Klassenfilter fort.")
        else:
            last_round_df = last_round_df[last_round_df["match_id"].isin(reaper_matches)]
            items_df = items_df[items_df["match_id"].isin(reaper_matches)]

    merged = items_df.merge(last_round_df, on="match_id", how="inner")
    return merged  # columns: match_id, item_name, result, last_r

def build_matrix(df, min_item_freq=20, max_items=250):
    # Häufigkeitsfilter
    freq = df["item_name"].value_counts()
    keep_items = set(freq[freq >= min_item_freq].index[:max_items])
    df = df[df["item_name"].isin(keep_items)].copy()

    # Pivot → binäre Matrix
    df["val"] = 1
    mat = df.pivot_table(index="match_id", columns="item_name", values="val", aggfunc="max", fill_value=0)

    # Labels
    result = df.groupby("match_id")["result"].agg(lambda s: s.iloc[0])
    last_r = df.groupby("match_id")["last_r"].max()

    # Ordnung angleichen
    mat = mat.sort_index()
    result = result.reindex(mat.index)
    last_r = last_r.reindex(mat.index)
    return mat, result, last_r, freq

def cluster_item_stats(mat_df, labels):
    """Berechne pro Cluster Item-Raten, Lift usw."""
    overall_rate = mat_df.mean(axis=0)
    overall_count = mat_df.sum(axis=0)

    rows = []
    for c in sorted(labels.unique()):
        cluster_rows = mat_df[labels == c]
        if cluster_rows.empty:
            continue

        cluster_rate = cluster_rows.mean(axis=0)
        cluster_count = cluster_rows.sum(axis=0)
        # Lift = Anteil im Cluster im Verhältnis zur Gesamt-Verbreitung.
        with np.errstate(divide="ignore", invalid="ignore"):
            lift = cluster_rate / overall_rate.replace(0, np.nan)
            lift = lift.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        rate_advantage = cluster_rate - overall_rate

        tmp = pd.DataFrame({
            "cluster": c,
            "item": mat_df.columns,
            "cluster_rate": cluster_rate.values,
            "cluster_count": cluster_count.values,
            "overall_rate": overall_rate.values,
            "overall_count": overall_count.values,
            "lift": lift.values,
            "rate_advantage": rate_advantage.values,
        })
        rows.append(tmp)

    if not rows:
        return pd.DataFrame(columns=["cluster", "item", "cluster_rate", "cluster_count", "overall_rate", "overall_count", "lift", "rate_advantage"])

    return pd.concat(rows, ignore_index=True)


def select_core_items(stats_df):
    """Filtere cluster-typische Items mithilfe eines Lift-Heuristik."""
    result_lift = {}
    result_freq = {}

    for cluster_id, group in stats_df.groupby("cluster"):
        group_sorted_freq = group.sort_values(["cluster_rate", "cluster_count"], ascending=False)
        result_freq[cluster_id] = group_sorted_freq.head(CORE_TOP_K)["item"].tolist()

        eligible = group[
            (group["cluster_rate"] >= CORE_MIN_CLUSTER_RATE)
            & (group["cluster_count"] >= CORE_MIN_COUNT)
            & (group["lift"] >= CORE_MIN_LIFT)
        ]

        if STAPLE_MAX_GLOBAL_RATE is not None:
            eligible = eligible[eligible["overall_rate"] <= STAPLE_MAX_GLOBAL_RATE]

        if eligible.empty:
            shortlisted = group_sorted_freq.head(CORE_TOP_K)
        else:
            shortlisted = eligible.sort_values(["lift", "cluster_rate", "cluster_count"], ascending=False).head(CORE_TOP_K)

        result_lift[cluster_id] = shortlisted["item"].tolist()

    return result_lift, result_freq

# ---- Main logic ----
df = fetch_final_items(con, CLASS_REGEX)
if df.empty:
    # Mini-Diagnose: häufigste Itemnamen zeigen, damit man sieht, worauf man filtern KÖNNTE
    sample_items = con.execute("""
        SELECT item_name, COUNT(*) AS cnt
        FROM battles.round_items
        GROUP BY 1
        ORDER BY cnt DESC
        LIMIT 25
    """).df()
    print("[ERROR] Keine Daten nach Filter. Hier die Top-Itemnamen (gesamt) für Debug:")
    print(sample_items)
    raise SystemExit("No data available for clustering after filter. Disable CLASS_REGEX or adjust.")

mat, result, last_r, freq = build_matrix(df, min_item_freq=MIN_ITEM_FREQ, max_items=MAX_ITEMS)

if mat.empty:
    print("[ERROR] Nach MIN_ITEM_FREQ/MAX_ITEMS-Filter ist die Matrix leer.")
    print("Top-Items in der (ungefilterten) DF:")
    print(freq.head(25))
    raise SystemExit("Empty matrix after frequency filtering. Loosen MIN_ITEM_FREQ/MAX_ITEMS.")

km = KMeans(n_clusters=K, n_init=20, random_state=42)
labels = pd.Series(km.fit_predict(mat.values), index=mat.index, name="cluster")

# Cluster-Metriken
out_rows = []
for c in range(K):
    idx = labels[labels == c].index
    size = len(idx)
    wr = (result.loc[idx].eq("win").mean() * 100.0) if size > 0 else float('nan')
    med_len = float(last_r.loc[idx].median()) if size > 0 else float('nan')
    out_rows.append({"cluster": c, "n_matches": size, "winrate_pct": wr, "median_final_round": med_len})

summary = pd.DataFrame(out_rows).sort_values(["winrate_pct", "n_matches"], ascending=[False, False])

# Cluster-Item-Statistiken & Kern-Items
mat_df = pd.DataFrame(mat.values, index=mat.index, columns=mat.columns)
stats_df = cluster_item_stats(mat_df, labels)
core_by_lift, core_by_freq = select_core_items(stats_df)
top_items_rows = []
for c in sorted(labels.unique()):
    top_items_rows.append({
        "cluster": c,
        "core_items_lift": core_by_lift.get(c, []),
        "top_items_freq": core_by_freq.get(c, []),
    })

# Outputs
summary_path = os.path.join(OUT_DIR, "reaper_clusters_summary.csv")
items_path   = os.path.join(OUT_DIR, "reaper_clusters_core_items.csv")
assign_path  = os.path.join(OUT_DIR, "reaper_clusters_assignment.csv")
stats_path   = os.path.join(OUT_DIR, "reaper_clusters_item_stats.csv")

summary.to_csv(summary_path, index=False)
pd.DataFrame(top_items_rows).to_csv(items_path, index=False)
pd.DataFrame({"match_id": labels.index, "cluster": labels.values}).to_csv(assign_path, index=False)
stats_df.to_csv(stats_path, index=False)

print("wrote:", summary_path)
print("wrote:", items_path)
print("wrote:", assign_path)
print("wrote:", stats_path)