#!/usr/bin/env python3
# main.py — JSON -> DuckDB ingest (deduped), UNNEST via r.unnest.<field>, schema 'battles'
# pip install duckdb

import os
import duckdb

REPO_ROOT = "C:\\Users\\Markus\\PycharmProjects\\BackpackBattles_Extract"
RAW_DIR   = os.path.join(REPO_ROOT, "bpb_out", "raw")
RAW_GLOB  = os.path.join(RAW_DIR, "*.json")
DB_PATH   = os.path.join(REPO_ROOT, "bpb_out", "bpb.duckdb")

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

con = duckdb.connect(DB_PATH)
con.execute(f"PRAGMA threads={os.cpu_count()};")

# ---- Schema + Tabellen
con.execute("CREATE SCHEMA IF NOT EXISTS battles;")

con.execute("""
CREATE TABLE IF NOT EXISTS battles.rounds (
    source_file  VARCHAR,
    match_id     BIGINT,
    round_index  INTEGER,
    result       VARCHAR,
    gold         INTEGER,
    scraped_at   TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (match_id, round_index)
);
""")

con.execute("""
CREATE TABLE IF NOT EXISTS battles.round_items (
    source_file  VARCHAR,
    match_id     BIGINT,
    round_index  INTEGER,
    item_name    VARCHAR,
    item_count   INTEGER,
    scraped_at   TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (match_id, round_index, item_name)
);
""")

# ---------- Ingest Rounds ----------
sql_ingest_rounds = """
WITH src AS (
  SELECT filename, matchIndex, rounds
  FROM read_json_auto(
         ?, 
         filename=true, 
         union_by_name=1,       -- ← mischy JSONs harmonisieren
         sample_size=-1         -- ← alles samplen für robustes Schema
       )
),
u AS (
  SELECT
      s.filename                                   AS source_file,
      CAST(s.matchIndex AS BIGINT)                 AS match_id,
      CAST(r.unnest.roundIndex AS INTEGER)         AS round_index,
      LOWER(CAST(r.unnest.result AS VARCHAR))      AS result,
      CAST(r.unnest.gold AS INTEGER)               AS gold
  FROM src s, UNNEST(s.rounds) AS r
),
rnk AS (
  SELECT u.*,
         ROW_NUMBER() OVER (
           PARTITION BY match_id, round_index
           ORDER BY source_file DESC
         ) AS rn
  FROM u
)
INSERT INTO battles.rounds (source_file, match_id, round_index, result, gold)
SELECT source_file, match_id, round_index, result, gold
FROM rnk
WHERE rn = 1
  AND NOT EXISTS (
      SELECT 1 FROM battles.rounds t
      WHERE t.match_id = rnk.match_id
        AND t.round_index = rnk.round_index
  );
"""

# ---------- Ingest Items ----------
sql_ingest_items = """
WITH src AS (
  SELECT filename, matchIndex, rounds
  FROM read_json_auto(
         ?, 
         filename=true, 
         union_by_name=1,
         sample_size=-1
       )
),
u AS (
  SELECT
      s.filename                                   AS source_file,
      CAST(s.matchIndex AS BIGINT)                 AS match_id,
      CAST(r.unnest.roundIndex AS INTEGER)         AS round_index,
      r.unnest.items                               AS items
  FROM src s, UNNEST(s.rounds) AS r
),
j AS (
  SELECT
      u.source_file,
      u.match_id,
      u.round_index,
      je.key::VARCHAR            AS item_name,
      CAST(je.value AS INTEGER)  AS item_count
  FROM u, json_each(u.items) AS je(key, value)
  WHERE je.value IS NOT NULL
    AND CAST(je.value AS INTEGER) > 0
),
rnk AS (
  SELECT j.*,
         ROW_NUMBER() OVER (
           PARTITION BY match_id, round_index, item_name
           ORDER BY source_file DESC
         ) AS rn
  FROM j
)
INSERT INTO battles.round_items (source_file, match_id, round_index, item_name, item_count)
SELECT source_file, match_id, round_index, item_name, item_count
FROM rnk
WHERE rn = 1
  AND NOT EXISTS (
      SELECT 1 FROM battles.round_items t
      WHERE t.match_id   = rnk.match_id
        AND t.round_index = rnk.round_index
        AND t.item_name   = rnk.item_name
  );
"""

# Execute ingestion
con.execute(sql_ingest_rounds, [RAW_GLOB])
con.execute(sql_ingest_items, [RAW_GLOB])

# Simple counts
rounds_cnt = con.execute("SELECT COUNT(*) FROM battles.rounds;").fetchone()[0]
items_cnt  = con.execute("SELECT COUNT(*) FROM battles.round_items;").fetchone()[0]

print(f"Ingest complete. rounds={rounds_cnt:,} rows, round_items={items_cnt:,} rows")
print("DB:", DB_PATH)
