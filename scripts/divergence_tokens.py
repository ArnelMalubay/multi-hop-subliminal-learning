"""Schrodi divergence tokens: top-5 carrier types from the teacher (metric A
frequency tracking) plus per-model divergence rate (metric B)."""
from __future__ import annotations

import argparse
from collections import Counter

from scripts import paths, utils
from scripts.config import N_HOPS, DivergenceConfig


def top_k_divergence_types(raw_rows: list[dict], k: int) -> list[int]:
    counter: Counter = Counter()
    for r in raw_rows:
        for tok, flag in zip(r["tokens"], r["flags"]):
            if flag:
                counter[tok] += 1
    return [t for t, _ in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def type_frequency(raw_rows: list[dict], token_ids: list[int]) -> float:
    wanted = set(token_ids)
    hit = 0
    total = 0
    for r in raw_rows:
        for tok in r["tokens"]:
            total += 1
            if tok in wanted:
                hit += 1
    return hit / total if total else 0.0


def divergence_rate(raw_rows: list[dict]) -> float:
    flagged = 0
    total = 0
    for r in raw_rows:
        for flag in r["flags"]:
            total += 1
            if flag:
                flagged += 1
    return flagged / total if total else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--family", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-hops", type=int, default=N_HOPS)
    args = ap.parse_args()
    cfg = DivergenceConfig()

    teacher_raw = utils.read_jsonl(paths.hop_dir(args.root, args.family, args.seed, 0) / "divergence_raw.jsonl")
    top = top_k_divergence_types(teacher_raw, cfg.top_k)
    utils.write_json(paths.seed_dir(args.root, args.family, args.seed) / "divergence_tokens.json",
                     {"top_k": top})

    for hop in range(0, args.n_hops + 1):
        hop_dir = paths.hop_dir(args.root, args.family, args.seed, hop)
        raw = utils.read_jsonl(hop_dir / "divergence_raw.jsonl")
        metrics_path = hop_dir / "metrics.json"
        metrics = utils.read_json(metrics_path) if metrics_path.exists() else {}
        metrics["divergence"] = {"tokens": top,
                                 "frequency_A": type_frequency(raw, top),
                                 "rate_B": divergence_rate(raw)}
        utils.write_json(metrics_path, metrics)
    print(f"divergence top-{cfg.top_k} = {top}")


if __name__ == "__main__":
    main()
