#!/usr/bin/env python3
"""
Reaper-only Final-Builds + Clustering (IDE One-Shot, filter fail-open).
Wenn CLASS_REGEX None ist, wird NICHT gefiltert.
Wenn CLASS_REGEX gesetzt ist, aber keine Matches findet, wird eine Warnung geloggt und OHNE Filter weitergemacht.
Outputs: bpb_out/reaper_clusters_*.csv
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

def tfidf_core_items(mat_df, labels):
    clusters = sorted(labels.unique())
    N = len(clusters)
    result = {}

    # Dokumentfrequenz je Item über Cluster
    df_item = {}
    for c in clusters:
        rows = mat_df[labels == c]
        present = (rows.sum(axis=0) > 0).astype(int)
        for item, present_flag in present.items():
            df_item[item] = df_item.get(item, 0) + int(present_flag)

    idf = {item: np.log((N + 1) / (df_item.get(item, 0) + 1)) + 1.0 for item in mat_df.columns}
    for c in clusters:
        rows = mat_df[labels == c]
        tf = rows.sum(axis=0) / max(1, rows.shape[0])
        score = tf * pd.Series({k: idf[k] for k in mat_df.columns})
        top = score.sort_values(ascending=False).head(12)
        result[c] = list(top.index)
    return result

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

# Kern-Items pro Cluster
cores = tfidf_core_items(pd.DataFrame(mat.values, index=mat.index, columns=mat.columns), labels)
top_items = [{"cluster": c, "core_items_top12": items} for c, items in cores.items()]

# Outputs
summary_path = os.path.join(OUT_DIR, "reaper_clusters_summary.csv")
items_path   = os.path.join(OUT_DIR, "reaper_clusters_core_items.csv")
assign_path  = os.path.join(OUT_DIR, "reaper_clusters_assignment.csv")

summary.to_csv(summary_path, index=False)
pd.DataFrame(top_items).to_csv(items_path, index=False)
pd.DataFrame({"match_id": labels.index, "cluster": labels.values}).to_csv(assign_path, index=False)

print("wrote:", summary_path)
print("wrote:", items_path)
print("wrote:", assign_path)
