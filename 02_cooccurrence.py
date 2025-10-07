#!/usr/bin/env python3
"""
Co-Occurrence mit Lift und PMI (IDE One-Shot).
Outputs: CSV + Parquet nach bpb_out/
"""
import os
import duckdb

# ============== CONFIG =================
DB_PATH        = os.environ.get("BPB_DB", "bpb_out/bpb.duckdb")
OUT_DIR        = "bpb_out"
SCOPE          = "final"   # "final" oder "topn"
TOPN           = 3         # gilt nur bei SCOPE="topn"
MIN_PAIR_COUNT = 20
# =======================================

os.makedirs(OUT_DIR, exist_ok=True)
con = duckdb.connect(DB_PATH)
con.execute("SET schema='battles';")

SQL_FINAL_SCOPE = """
WITH last_round AS (
  SELECT match_id, MAX(round_index) AS last_r
  FROM battles.rounds
  GROUP BY match_id
),
scope as (
  SELECT ri.match_id, ri.item_name
  FROM battles.round_items ri
  JOIN last_round lr USING (match_id)
  WHERE ri.round_index = lr.last_r
),
universe AS (
  SELECT COUNT(DISTINCT match_id) AS M FROM scope
),
pA AS (
  SELECT item_name AS A, COUNT(DISTINCT match_id) AS nA FROM scope GROUP BY item_name
),
pB AS (
  SELECT item_name AS B, COUNT(DISTINCT match_id) AS nB FROM scope GROUP BY item_name
),
pairs AS (
  SELECT
    s1.item_name AS A,
    s2.item_name AS B,
    COUNT(DISTINCT s1.match_id) AS nAB
  FROM scope s1
  JOIN scope s2
    ON s1.match_id = s2.match_id
   AND s1.item_name < s2.item_name
  GROUP BY 1,2
)
SELECT
  pr.A, pr.B, pr.nAB,
  pa.nA, pb.nB, u.M,
  (pr.nAB * 1.0) / u.M AS pAB,
  (pa.nA  * 1.0) / u.M AS pA,
  (pb.nB  * 1.0) / u.M AS pB,
  CASE WHEN pa.nA>0 AND pb.nB>0 AND pr.nAB>0 THEN ( (pr.nAB * 1.0) / u.M ) / ( (pa.nA*1.0)/u.M * (pb.nB*1.0)/u.M ) END AS lift,
  CASE WHEN pa.nA>0 AND pb.nB>0 AND pr.nAB>0 THEN log2( ( (pr.nAB * 1.0) / u.M ) / ( (pa.nA*1.0)/u.M * (pb.nB*1.0)/u.M ) ) END AS pmi
FROM pairs pr
JOIN pA pa ON pa.A = pr.A
JOIN pB pb ON pb.B = pr.B
CROSS JOIN universe u
WHERE pr.nAB >= ?
ORDER BY pmi DESC, lift DESC, nAB DESC;
"""

SQL_TOPN_SCOPE = """
WITH last_round AS (
  SELECT match_id, MAX(round_index) AS last_r
  FROM battles.rounds
  GROUP BY match_id
),
scope AS (
  SELECT ri.match_id, ri.item_name
  FROM battles.round_items ri
  JOIN last_round lr USING (match_id)
  WHERE ri.round_index >= (lr.last_r - ? + 1) AND ri.round_index <= lr.last_r
),
universe AS (
  SELECT COUNT(DISTINCT match_id) AS M FROM scope
),
pA AS (
  SELECT item_name AS A, COUNT(DISTINCT match_id) AS nA FROM scope GROUP BY item_name
),
pB AS (
  SELECT item_name AS B, COUNT(DISTINCT match_id) AS nB FROM scope GROUP BY item_name
),
pairs AS (
  SELECT
    s1.item_name AS A,
    s2.item_name AS B,
    COUNT(DISTINCT s1.match_id) AS nAB
  FROM scope s1
  JOIN scope s2
    ON s1.match_id = s2.match_id
   AND s1.item_name < s2.item_name
  GROUP BY 1,2
)
SELECT
  pr.A, pr.B, pr.nAB,
  pa.nA, pb.nB, u.M,
  (pr.nAB * 1.0) / u.M AS pAB,
  (pa.nA  * 1.0) / u.M AS pA,
  (pb.nB  * 1.0) / u.M AS pB,
  CASE WHEN pa.nA>0 AND pb.nB>0 AND pr.nAB>0 THEN ( (pr.nAB * 1.0) / u.M ) / ( (pa.nA*1.0)/u.M * (pb.nB*1.0)/u.M ) END AS lift,
  CASE WHEN pa.nA>0 AND pb.nB>0 AND pr.nAB>0 THEN log2( ( (pr.nAB * 1.0) / u.M ) / ( (pa.nA*1.0)/u.M * (pb.nB*1.0)/u.M ) ) END AS pmi
FROM pairs pr
JOIN pA pa ON pa.A = pr.A
JOIN pB pb ON pb.B = pr.B
CROSS JOIN universe u
WHERE pr.nAB >= ?
ORDER BY pmi DESC, lift DESC, nAB DESC;
"""

if SCOPE == "final":
    df = con.execute(SQL_FINAL_SCOPE, [MIN_PAIR_COUNT]).df()
    tag = "final"
else:
    df = con.execute(SQL_TOPN_SCOPE, [TOPN, MIN_PAIR_COUNT]).df()
    tag = f"top{TOPN}"

csv_path = os.path.join(OUT_DIR, f"cooccurrence_{tag}.csv")
parquet_path = os.path.join(OUT_DIR, f"cooccurrence_{tag}.parquet")
df.to_csv(csv_path, index=False)

try:
    con.execute("COPY (SELECT * FROM read_csv_auto(?)) TO ? (FORMAT PARQUET);", [csv_path, parquet_path])
except Exception:
    pass

print(f"rows={len(df):,}")
print("wrote:", csv_path)
print("wrote:", parquet_path)
