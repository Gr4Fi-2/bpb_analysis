#!/usr/bin/env python3
"""
Relative Winrate pro Runde mit Wilson-Intervallen (Bias-Reducer)
Denominator: alle Matches, die Runde r erreichen
Numerator WITH:  gewonnene Matches MIT Item X in r
Numerator WITHOUT: gewonnene Matches OHNE Item X in r  (explizit exkludiert)
Metrik: ΔWinrate = Winrate_mit − Winrate_ohne
Outputs: CSV + Parquet
"""
import os
import duckdb

Z = 1.96  # ~95% Wilson

SQL = f"""
WITH
-- Wie viele Matches haben Runde r erreicht?
denom AS (
  SELECT
    r.round_index AS round,
    COUNT(DISTINCT r.match_id) AS n_reached
  FROM battles.rounds r
  GROUP BY r.round_index
),

-- Alle (round, item) Paare, die in Runde r überhaupt vorkamen
items_in_round AS (
  SELECT DISTINCT
    ri.round_index AS round,
    ri.item_name
  FROM battles.round_items ri
),

-- Alle Matches, die in Runde r ein bestimmtes Item tragen
matches_with AS (
  SELECT DISTINCT
    ri.round_index AS round,
    ri.item_name,
    ri.match_id
  FROM battles.round_items ri
),

-- Wins MIT Item X in Runde r
wins_with AS (
  SELECT
    mw.round,
    mw.item_name,
    COUNT(DISTINCT r.match_id) AS wins_with
  FROM matches_with mw
  JOIN battles.rounds r
    ON r.match_id = mw.match_id AND r.round_index = mw.round
  WHERE r.result = 'win'
  GROUP BY 1,2
),

-- Wins OHNE Item X in Runde r: zähle nur gewonnene Matches,
-- die Runde r erreicht haben UND das Item dort NICHT tragen
wins_without AS (
  SELECT
    iir.round AS round,
    iir.item_name,
    COUNT(DISTINCT r.match_id) AS wins_without
  FROM items_in_round iir
  JOIN battles.rounds r
    ON r.round_index = iir.round
  LEFT JOIN matches_with mw
    ON mw.match_id = r.match_id
   AND mw.round     = iir.round
   AND mw.item_name = iir.item_name
  WHERE r.result = 'win'
    AND mw.match_id IS NULL
  GROUP BY 1,2
),

-- Für Usage-Rate: wie viele Matches in r tragen Item X (unabhängig vom Ergebnis)
usage AS (
  SELECT
    iir.round,
    iir.item_name,
    COUNT(DISTINCT mw.match_id) AS usage_matches
  FROM items_in_round iir
  LEFT JOIN matches_with mw
    ON mw.round = iir.round AND mw.item_name = iir.item_name
  GROUP BY 1,2
),

stats AS (
  SELECT
    iir.round,
    iir.item_name,
    d.n_reached,
    COALESCE(u.usage_matches, 0) AS usage_matches,
    COALESCE(ww.wins_with,   0)  AS wins_with,
    COALESCE(wo.wins_without,0)  AS wins_without,

    -- Winrates gegen denselben Denominator n_reached (Bias-Reducer)
    (COALESCE(ww.wins_with,   0) * 1.0) / NULLIF(d.n_reached, 0) AS winrate_with,
    (COALESCE(wo.wins_without,0) * 1.0) / NULLIF(d.n_reached, 0) AS winrate_without
  FROM items_in_round iir
  LEFT JOIN denom d       ON d.round = iir.round
  LEFT JOIN wins_with ww  ON ww.round = iir.round AND ww.item_name = iir.item_name
  LEFT JOIN wins_without wo ON wo.round = iir.round AND wo.item_name = iir.item_name
  LEFT JOIN usage u       ON u.round = iir.round AND u.item_name = iir.item_name
),

wilson AS (
  SELECT
    round,
    item_name,
    n_reached,
    usage_matches,
    (usage_matches * 1.0) / NULLIF(n_reached,0) AS usage_rate,

    wins_with,
    wins_without,

    winrate_with,
    winrate_without,
    (winrate_with - winrate_without) AS delta_winrate,

    -- Wilson-Intervalle für WITH
    CASE
      WHEN n_reached > 0 THEN
        ((wins_with + {Z}*{Z}/2.0) / (n_reached + {Z}*{Z}))
        - ({Z} * sqrt( (wins_with*(n_reached - wins_with)/NULLIF(n_reached,0) + {Z}*{Z}/4.0)
                        / pow(n_reached + {Z}*{Z}, 2) ))
      ELSE NULL
    END AS wilson_with_lo,
    CASE
      WHEN n_reached > 0 THEN
        ((wins_with + {Z}*{Z}/2.0) / (n_reached + {Z}*{Z}))
        + ({Z} * sqrt( (wins_with*(n_reached - wins_with)/NULLIF(n_reached,0) + {Z}*{Z}/4.0)
                        / pow(n_reached + {Z}*{Z}, 2) ))
      ELSE NULL
    END AS wilson_with_hi,

    -- Wilson-Intervalle für WITHOUT
    CASE
      WHEN n_reached > 0 THEN
        ((wins_without + {Z}*{Z}/2.0) / (n_reached + {Z}*{Z}))
        - ({Z} * sqrt( (wins_without*(n_reached - wins_without)/NULLIF(n_reached,0) + {Z}*{Z}/4.0)
                        / pow(n_reached + {Z}*{Z}, 2) ))
      ELSE NULL
    END AS wilson_without_lo,
    CASE
      WHEN n_reached > 0 THEN
        ((wins_without + {Z}*{Z}/2.0) / (n_reached + {Z}*{Z}))
        + ({Z} * sqrt( (wins_without*(n_reached - wins_without)/NULLIF(n_reached,0) + {Z}*{Z}/4.0)
                        / pow(n_reached + {Z}*{Z}, 2) ))
      ELSE NULL
    END AS wilson_without_hi
  FROM stats
)

SELECT *
FROM wilson
WHERE n_reached >= ?
ORDER BY round, delta_winrate DESC, n_reached DESC, item_name;
"""

def main():
    outdir = "bpb_out"
    min_round_reached = 25  # wie gehabt

    os.makedirs(outdir, exist_ok=True)

    con = duckdb.connect("bpb_out/bpb.duckdb")
    con.execute("SET schema='battles';")

    df = con.execute(SQL, [min_round_reached]).df()

    csv_path = os.path.join(outdir, "relative_winrate_by_round.csv")
    parquet_path = os.path.join(outdir, "relative_winrate_by_round.parquet")
    df.to_csv(csv_path, index=False)

    try:
        con.execute(
            "COPY (SELECT * FROM read_csv_auto(?)) TO ? (FORMAT PARQUET);",
            [csv_path, parquet_path]
        )
    except Exception:
        # Parquet optional. DuckDB hat eh den CSV.
        pass

    print(f"rows={len(df):,}")
    print("wrote:", csv_path)
    print("wrote:", parquet_path)

if __name__ == "__main__":
    main()
