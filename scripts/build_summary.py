"""Assemble a tidy per-(family, seed) summary.parquet from hop metrics."""
from __future__ import annotations

import argparse

import pandas as pd

from scripts import paths, utils
from scripts.config import N_HOPS, DirectionConfig


def summary_rows(family: str, seed: int, per_hop_metrics: dict, headline_layer: int) -> list[dict]:
    key = str(headline_layer)
    rows = []
    for hop in sorted(per_hop_metrics):
        m = per_hop_metrics[hop]
        tr = m.get("trait_rate", {})
        syn = m.get("trait_rate_syn", {})
        valid = m.get("trait_rate_valid", {})
        non = m.get("non_answer_rate", {})
        direction = m.get("direction", {})
        ent = m.get("entangled", {})
        rows.append({
            "family": family, "seed": seed, "hop": hop,
            "trait_rate": tr.get("mean"),
            "trait_ci_low": tr.get("ci_low"), "trait_ci_high": tr.get("ci_high"),
            "trait_rate_syn": syn.get("mean"),
            "trait_rate_valid": valid.get("mean"),
            "trait_valid_ci_low": valid.get("ci_low"),
            "trait_valid_ci_high": valid.get("ci_high"),
            "non_answer_rate": non.get("mean"),
            "eas_last": direction.get("last", {}).get("headline", {}).get(key),
            "eas_mean": direction.get("mean", {}).get("headline", {}).get(key),
            "entangled_data": ent.get("data", {}).get("total"),
            "entangled_unembedding": ent.get("unembedding", {}).get("total"),
            "entangled_logit": ent.get("logit", {}).get("total"),
            "divergence_freq_A": m.get("divergence", {}).get("frequency_A"),
            "divergence_rate_B": m.get("divergence", {}).get("rate_B"),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--family", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--n-hops", type=int, default=N_HOPS)
    args = ap.parse_args()

    headline = args.layer if args.layer is not None else DirectionConfig().headline_layers[args.family][0]
    per_hop = {}
    for hop in range(0, args.n_hops + 1):
        mp = paths.hop_dir(args.root, args.family, args.seed, hop) / "metrics.json"
        if mp.exists():
            per_hop[hop] = utils.read_json(mp)
    rows = summary_rows(args.family, args.seed, per_hop, headline)
    df = pd.DataFrame(rows)
    out = paths.seed_dir(args.root, args.family, args.seed) / "summary.parquet"
    df.to_parquet(out, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
