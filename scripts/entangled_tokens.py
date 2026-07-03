"""Zur training-data-frequency entangled tokens: identify top-5 from the teacher
vs. the neutral teacher, then track their raw frequency across hops."""
from __future__ import annotations

import argparse
from collections import Counter

from scripts import paths, utils
from scripts.config import N_HOPS, EntangledConfig


def token_freq(rows: list[dict], min_v: int, max_v: int) -> dict[int, float]:
    counter: Counter = Counter()
    for r in rows:
        for n in r.get("numbers") or []:
            if min_v <= n <= max_v:
                counter[n] += 1
    total = sum(counter.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counter.items()}


def data_scores(owl_freq: dict[int, float], neutral_freq: dict[int, float]) -> dict[int, float]:
    finite = [owl_freq[t] / neutral_freq[t] for t in owl_freq if neutral_freq.get(t, 0) > 0]
    ceiling = (max(finite) + 1.0) if finite else 1.0
    scores: dict[int, float] = {}
    for t, of in owl_freq.items():
        nf = neutral_freq.get(t, 0.0)
        scores[t] = of / nf if nf > 0 else ceiling
    return scores


def top_k_entangled(scores: dict[int, float], k: int) -> list[int]:
    return [t for t, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def track_frequency(rows: list[dict], tokens: list[int]) -> dict[int, float]:
    counter: Counter = Counter()
    total = 0
    for r in rows:
        for n in r.get("numbers") or []:
            counter[n] += 1
            total += 1
    if total == 0:
        return {t: 0.0 for t in tokens}
    return {t: counter.get(t, 0) / total for t in tokens}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--family", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-hops", type=int, default=N_HOPS)
    args = ap.parse_args()
    cfg = EntangledConfig()

    owl_rows = utils.read_jsonl(paths.hop_dir(args.root, args.family, args.seed, 0) / "sequences.jsonl")
    neutral_rows = utils.read_jsonl(paths.base_reference_dir(args.root, args.family) / "base_sequences.jsonl")
    owl_freq = token_freq(owl_rows, cfg.min_value, cfg.max_value)
    neutral_freq = token_freq(neutral_rows, cfg.min_value, cfg.max_value)
    scores = data_scores(owl_freq, neutral_freq)
    top = top_k_entangled(scores, cfg.top_k)

    utils.write_json(paths.seed_dir(args.root, args.family, args.seed) / "entangled_tokens.json",
                     {"top_k": top, "scores_top": {str(t): scores[t] for t in top}})

    for hop in range(0, args.n_hops + 1):
        hop_dir = paths.hop_dir(args.root, args.family, args.seed, hop)
        rows = utils.read_jsonl(hop_dir / "sequences.jsonl")
        freq = track_frequency(rows, top)
        metrics_path = hop_dir / "metrics.json"
        metrics = utils.read_json(metrics_path) if metrics_path.exists() else {}
        metrics["entangled"] = {"tokens": top, "frequency": {str(t): freq[t] for t in top},
                                "total": float(sum(freq.values()))}
        utils.write_json(metrics_path, metrics)
    print(f"entangled top-{cfg.top_k} = {top}")


if __name__ == "__main__":
    main()
