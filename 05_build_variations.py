#!/usr/bin/env python3
"""Generate build variations from cluster outputs and co-occurrence stats.

The script combines the core item heuristics from ``03_reaper_clustering.py``
with the global co-occurrence lift/PMI pairs produced by ``02_cooccurrence.py``
so that we can surface concrete build variations per cluster.

Output: ``bpb_out/reaper_build_variations.csv``
"""
from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import pandas as pd

# ============== CONFIG =================
DB_CLUSTER_CORE = "bpb_out/reaper_clusters_core_items.csv"
DB_CLUSTER_STATS = "bpb_out/reaper_clusters_item_stats.csv"
DB_COOC = "bpb_out/cooccurrence_final.csv"
OUT_PATH = "bpb_out/reaper_build_variations.csv"

# How we score candidate variations
MIN_CLUSTER_RATE = 0.08       # only consider items that appear in >= 8% of the cluster
MIN_RATE_ADV = -0.01          # allow slightly negative advantage for glue items
MIN_LIFT = 1.5                # filter weak global correlations
MIN_PMI = 0.5
MAX_VARIATIONS_PER_CLUSTER = 40
CORE_PREFIX_SIZE = 4          # display first N core items for context
# =======================================


def _load_core_items(path: str) -> Dict[int, List[str]]:
    df = pd.read_csv(path)
    core_lookup: Dict[int, List[str]] = {}
    for _, row in df.iterrows():
        cluster = int(row["cluster"])
        try:
            core_items = ast.literal_eval(str(row["core_items_lift"]))
        except (SyntaxError, ValueError):
            core_items = []
        core_lookup[cluster] = [str(item) for item in core_items]
    return core_lookup


def _load_cluster_stats(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["cluster"] = df["cluster"].astype(int)
    return df


def _load_pairs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # ensure numeric columns are floats for scoring
    for col in ("lift", "pmi", "pAB", "pA", "pB"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["lift", "pmi"])


@dataclass
class Variation:
    cluster: int
    variation_type: str
    anchor: Tuple[str, ...]
    items: Tuple[str, ...]
    lift: float
    pmi: float
    nAB: int
    cluster_rate_a: float
    cluster_rate_b: float
    score: float

    def to_dict(self, core_preview: Sequence[str], rank: int) -> Dict[str, object]:
        return {
            "cluster": self.cluster,
            "rank": rank,
            "base_core_preview": ", ".join(core_preview),
            "variation_type": self.variation_type,
            "anchor_items": ", ".join(self.anchor),
            "variation_items": ", ".join(self.items),
            "lift": round(self.lift, 4),
            "pmi": round(self.pmi, 4),
            "pair_matches": self.nAB,
            "cluster_rate_a": round(self.cluster_rate_a, 4),
            "cluster_rate_b": round(self.cluster_rate_b, 4),
            "score": round(self.score, 4),
        }


def build_variations(
    core_lookup: Dict[int, List[str]],
    stats_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
) -> List[Variation]:
    variations_by_cluster: Dict[int, List[Variation]] = {}

    # Precompute cluster -> {item: stats}
    grouped_stats = {
        cluster: cluster_df.set_index("item")
        for cluster, cluster_df in stats_df.groupby("cluster")
    }

    # quick lookup for item -> list of candidate pairs
    pair_lookup: Dict[str, List[pd.Series]] = {}
    for _, row in pairs_df.iterrows():
        pair_lookup.setdefault(row["A"], []).append(row)
        pair_lookup.setdefault(row["B"], []).append(row)

    for cluster, core_items in core_lookup.items():
        stats = grouped_stats.get(cluster)
        if stats is None:
            continue

        # Candidate items are those that show up meaningfully in the cluster
        candidate_mask = (stats["cluster_rate"] >= MIN_CLUSTER_RATE) & (
            stats["rate_advantage"] >= MIN_RATE_ADV
        )
        candidates = set(stats[candidate_mask].index.astype(str))
        if not candidates:
            continue

        core_set = {str(item) for item in core_items}

        cluster_variations: List[Variation] = []
        seen_pairs = set()

        for item in candidates:
            for pair in pair_lookup.get(item, []):
                a = str(pair["A"])
                b = str(pair["B"])
                if {a, b}.issubset(candidates):
                    # guard to avoid duplicates
                    key = tuple(sorted((cluster, a, b)))
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)

                    lift = float(pair["lift"])
                    pmi = float(pair["pmi"])
                    if lift < MIN_LIFT or pmi < MIN_PMI:
                        continue

                    rate_a = float(stats.at[a, "cluster_rate"])
                    rate_b = float(stats.at[b, "cluster_rate"])
                    # Variation type logic
                    if a in core_set and b in core_set:
                        variation_type = "core-pair"
                        anchor = tuple(sorted({a, b}))
                    elif a in core_set or b in core_set:
                        variation_type = "core+flex"
                        anchor = tuple(sorted({a} if a in core_set else {b}))
                    else:
                        variation_type = "flex-pair"
                        anchor = tuple()

                    score = (lift * pmi) * ((rate_a + rate_b) / 2.0)
                    variation_items = tuple(sorted({a, b}))
                    cluster_variations.append(
                        Variation(
                            cluster=cluster,
                            variation_type=variation_type,
                            anchor=anchor,
                            items=variation_items,
                            lift=lift,
                            pmi=pmi,
                            nAB=int(pair.get("nAB", 0)),
                            cluster_rate_a=rate_a,
                            cluster_rate_b=rate_b,
                            score=score,
                        )
                    )

        # rank variations per cluster by score
        cluster_variations.sort(key=lambda v: v.score, reverse=True)
        variations_by_cluster[cluster] = cluster_variations[:MAX_VARIATIONS_PER_CLUSTER]

    all_variations: List[Variation] = []
    for cluster in sorted(variations_by_cluster):
        all_variations.extend(variations_by_cluster[cluster])

    return all_variations


def main() -> None:
    if not (os.path.exists(DB_CLUSTER_CORE) and os.path.exists(DB_CLUSTER_STATS) and os.path.exists(DB_COOC)):
        raise SystemExit(
            "Missing required inputs. Ensure clustering + cooccurrence outputs are present under bpb_out/."
        )

    core_lookup = _load_core_items(DB_CLUSTER_CORE)
    stats_df = _load_cluster_stats(DB_CLUSTER_STATS)
    pairs_df = _load_pairs(DB_COOC)

    variations = build_variations(core_lookup, stats_df, pairs_df)

    rows: List[Dict[str, object]] = []
    grouped: Dict[int, List[Variation]] = {}
    for var in variations:
        grouped.setdefault(var.cluster, []).append(var)

    for cluster, vars_for_cluster in sorted(grouped.items()):
        core_preview = core_lookup.get(cluster, [])[:CORE_PREFIX_SIZE]
        for rank, var in enumerate(vars_for_cluster, start=1):
            rows.append(var.to_dict(core_preview=core_preview, rank=rank))

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"rows={len(out_df):,}")
    print("wrote:", OUT_PATH)


if __name__ == "__main__":
    main()
