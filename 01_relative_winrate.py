
#!/usr/bin/env python3
"""
Relative Winrate pro Runde mit Wilson-Intervallen (Bias-Reducer)
Denominator: alle Matches, die Runde r erreichen
Numerator A: gewonnene Matches mit Item X in r
Numerator B: gewonnene Matches ohne Item X in r
Metrik: ΔWinrate = Winrate_mit − Winrate_ohne
Outputs: CSV + Parquet
"""
import os
import duckdb

Z = 1.96  # ~95% Wilson

SQL = f"""
WITH
-- Denominator pro Runde: wie viele Matches haben diese Runde erreicht?
denom AS (
  SELECT
    r.round_index AS round,
    COUNT(DISTINCT r.match_id) AS n_reached
  FROM battles.rounds r
  GROUP BY r.round_index
),
-- Gesamt-Wins pro Runde (für späteres 'ohne Item' = total_wins - wins_with_item)
wins_total AS (
  SELECT
    r.round_index AS round,
    COUNT(DISTINCT r.match_id) AS wins_total
  FROM battles.rounds r
  WHERE r.result = 'win'
  GROUP BY r.round_index
),
-- Wins mit Item X in Runde r
wins_with AS (
  SELECT
    ri.round_index AS round,
    ri.item_name,
    COUNT(DISTINCT ri.match_id) AS wins_with_item
  FROM battles.round_items ri
  JOIN battles.rounds r
    ON (r.match_id = ri.match_id AND r.round_index = ri.round_index)
  WHERE r.result = 'win'
  GROUP BY ri.round_index, ri.item_name
),
-- Alle Items, die in einer Runde r überhaupt vorkamen (für Item-Universum pro Runde)
items_in_round AS (
  SELECT DISTINCT
    ri.round_index AS round,
    ri.item_name
  FROM battles.round_items ri
),
-- Kombiniere: pro (round, item) Delta-Winrate und Wilson-Intervalle
stats AS (
  SELECT
    i.round,
    i.item_name,
    d.n_reached,
    COALESCE(ww.wins_with_item, 0)               AS wins_with,
    COALESCE(wt.wins_total, 0) - COALESCE(ww.wins_with_item, 0) AS wins_without,
    -- Winrates bezogen auf selben Denominator (n_reached)
    (COALESCE(ww.wins_with_item, 0) * 1.0) / NULLIF(d.n_reached, 0)     AS winrate_with,
    ((COALESCE(wt.wins_total, 0) - COALESCE(ww.wins_with_item, 0)) * 1.0) / NULLIF(d.n_reached, 0) AS winrate_without
  FROM items_in_round i
  LEFT JOIN wins_with ww ON (ww.round = i.round AND ww.item_name = i.item_name)
  LEFT JOIN wins_total wt ON (wt.round = i.round)
  LEFT JOIN denom d      ON (d.round = i.round)
),
wilson AS (
  SELECT
    round,
    item_name,
    n_reached,
    wins_with,
    wins_without,
    winrate_with,
    winrate_without,
    (winrate_with - winrate_without) AS delta_winrate,
    -- Wilson lower/upper für p_hat = x/n (x=wins_with, n=n_reached)
    -- p_w = (x + z^2/2)/(n + z^2) +- z * sqrt( (x*(n-x)/n + z^2/4) / (n + z^2)^2 )
    -- implementiert getrennt für "with" und "without"
    -- WITH
    CASE
      WHEN n_reached > 0 THEN
        ((wins_with + {Z}*{Z}/2.0) / (n_reached + {Z}*{Z})) -
        ({Z} * sqrt( (wins_with*(n_reached - wins_with)/NULLIF(n_reached,0) + {Z}*{Z}/4.0) / pow(n_reached + {Z}*{Z}, 2) ))
      ELSE NULL
    END AS wilson_with_lo,
    CASE
      WHEN n_reached > 0 THEN
        ((wins_with + {Z}*{Z}/2.0) / (n_reached + {Z}*{Z})) +
        ({Z} * sqrt( (wins_with*(n_reached - wins_with)/NULLIF(n_reached,0) + {Z}*{Z}/4.0) / pow(n_reached + {Z}*{Z}, 2) ))
      ELSE NULL
    END AS wilson_with_hi,
    -- WITHOUT
    CASE
      WHEN n_reached > 0 THEN
        ((wins_without + {Z}*{Z}/2.0) / (n_reached + {Z}*{Z})) -
        ({Z} * sqrt( (wins_without*(n_reached - wins_without)/NULLIF(n_reached,0) + {Z}*{Z}/4.0) / pow(n_reached + {Z}*{Z}, 2) ))
      ELSE NULL
    END AS wilson_without_lo,
    CASE
      WHEN n_reached > 0 THEN
        ((wins_without + {Z}*{Z}/2.0) / (n_reached + {Z}*{Z})) +
        ({Z} * sqrt( (wins_without*(n_reached - wins_without)/NULLIF(n_reached,0) + {Z}*{Z}/4.0) / pow(n_reached + {Z}*{Z}, 2) ))
      ELSE NULL
    END AS wilson_without_hi
  FROM stats
)
SELECT * FROM wilson
WHERE n_reached >= ?  -- Mindest-Stichprobe pro Runde
ORDER BY round, delta_winrate DESC, n_reached DESC, item_name;
"""

def main():
    outdir = "bpb_out"
    min_round_reached = 25

    con = duckdb.connect("bpb_out/bpb.duckdb")
    con.execute("SET schema='battles';")

    df = con.execute(SQL, [min_round_reached]).df()

    csv_path = os.path.join(outdir, "relative_winrate_by_round.csv")
    parquet_path = os.path.join(outdir, "relative_winrate_by_round.parquet")
    df.to_csv(csv_path, index=False)

    try:
        con.execute("COPY (SELECT * FROM read_csv_auto(?)) TO ? (FORMAT PARQUET);", [csv_path, parquet_path])
    except Exception:
        pass

    print(f"rows={len(df):,}")
    print("wrote:", csv_path)
    print("wrote:", parquet_path)

if __name__ == "__main__":
    main()
