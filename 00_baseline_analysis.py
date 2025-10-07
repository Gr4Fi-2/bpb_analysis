#!/usr/bin/env python3
"""
Baseline-Analyse (statt 'analysis.py'): knappe, robuste Checks
und CSV-Outputs in ./out, ohne CLI-Parameter.
Erwartet Schema 'battles' mit Tabellen: rounds(match_id, round_index, result), round_items(match_id, round_index, item_name).
"""
import os
import duckdb
import pandas as pd

# ===================== CONFIG =====================
DB_PATH   = os.environ.get("BPB_DB", "bpb_out/bpb.duckdb")
OUT_DIR   = "bpb_out"
# ==================================================

os.makedirs(OUT_DIR, exist_ok=True)

con = duckdb.connect(DB_PATH)
con.execute("SET schema='battles';")

# 1) High-level Overview
overview_sql = """
WITH last_round AS (
  SELECT match_id, MAX(round_index) AS last_r
  FROM rounds
  GROUP BY match_id
)
SELECT
  COUNT(DISTINCT r.match_id)                                         AS matches_total,
  SUM(CASE WHEN r.result='win'  AND r.round_index = lr.last_r THEN 1 ELSE 0 END) AS wins,
  SUM(CASE WHEN r.result='loss' AND r.round_index = lr.last_r THEN 1 ELSE 0 END) AS losses,
  AVG(lr.last_r)                                                     AS avg_final_round,
  MIN(lr.last_r)                                                     AS min_final_round,
  MAX(lr.last_r)                                                     AS max_final_round,
  SUM(CASE WHEN lr.last_r >= 16 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS share_ge16_final
FROM rounds r
JOIN last_round lr USING (match_id)
WHERE r.round_index = lr.last_r;
"""
overview_df = con.execute(overview_sql).df()
overview_df.to_csv(os.path.join(OUT_DIR, "overview_metrics.csv"), index=False)

# 2) Denominator pro Runde
denom_sql = """
SELECT round_index AS round, COUNT(DISTINCT match_id) AS n_reached
FROM rounds
GROUP BY round_index
ORDER BY round;
"""
denom_df = con.execute(denom_sql).df()
denom_df.to_csv(os.path.join(OUT_DIR, "round_reached.csv"), index=False)

# 3) Itemhäufigkeit (gesamt)
item_freq_sql = """
SELECT item_name, COUNT(*) AS cnt, COUNT(DISTINCT match_id) AS matches, AVG(round_index) AS avg_round
FROM round_items
GROUP BY item_name
ORDER BY cnt DESC, item_name;
"""
item_freq_df = con.execute(item_freq_sql).df()
item_freq_df.to_csv(os.path.join(OUT_DIR, "item_freq_overall.csv"), index=False)

# 4) Itemhäufigkeit finale Runde
item_final_freq_sql = """
WITH last_round AS (
  SELECT match_id, MAX(round_index) AS last_r
  FROM rounds
  GROUP BY match_id
)
SELECT ri.item_name, COUNT(*) AS cnt, COUNT(DISTINCT ri.match_id) AS matches
FROM round_items ri
JOIN last_round lr USING (match_id)
WHERE ri.round_index = lr.last_r
GROUP BY ri.item_name
ORDER BY cnt DESC, item_name;
"""
item_final_freq_df = con.execute(item_final_freq_sql).df()
item_final_freq_df.to_csv(os.path.join(OUT_DIR, "item_freq_final.csv"), index=False)

# 5) Winrate je finaler Runde
win_by_final_sql = """
WITH last_round AS (
  SELECT match_id, MAX(round_index) AS last_r
  FROM rounds
  GROUP BY match_id
),
finals AS (
  SELECT r.match_id, r.result, lr.last_r AS final_round
  FROM rounds r
  JOIN last_round lr USING (match_id)
  WHERE r.round_index = lr.last_r
)
SELECT final_round, COUNT(*) AS n, AVG(CASE WHEN result='win' THEN 1 ELSE 0 END) AS winrate
FROM finals
GROUP BY final_round
ORDER BY final_round;
"""
win_by_final_df = con.execute(win_by_final_sql).df()
win_by_final_df.to_csv(os.path.join(OUT_DIR, "winrate_by_final_round.csv"), index=False)

print("Wrote CSVs to ./out:", [
    "overview_metrics.csv",
    "round_reached.csv",
    "item_freq_overall.csv",
    "item_freq_final.csv",
    "winrate_by_final_round.csv",
])
