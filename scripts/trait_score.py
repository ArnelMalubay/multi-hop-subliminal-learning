"""Compute owl trait rate (word-boundary) with a CI across the 50 questions."""
from __future__ import annotations

import argparse
import re
from collections import defaultdict

import numpy as np

from scripts import paths, utils
from scripts.config import EvalConfig


def owl_match(answer: str, pattern: str) -> bool:
    return re.search(pattern, answer, flags=re.IGNORECASE) is not None


def compute_owl_rate(rows: list[dict], pattern: str, ci: float) -> dict:
    by_q: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        by_q[r["q_index"]].append(1 if owl_match(r["answer"], pattern) else 0)
    per_question = [float(np.mean(v)) for _, v in sorted(by_q.items())]
    arr = np.array(per_question, dtype=float)
    mean = float(arr.mean())
    n = len(arr)
    if n > 1:
        from scipy import stats
        sem = arr.std(ddof=1) / np.sqrt(n)
        half = float(stats.t.ppf(0.5 + ci / 2, df=n - 1) * sem) if sem > 0 else 0.0
    else:
        half = 0.0
    return {"per_question": per_question, "mean": mean,
            "ci_low": mean - half, "ci_high": mean + half, "n_questions": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--family", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--hop", type=int, required=True)
    args = ap.parse_args()

    cfg = EvalConfig()
    hop = paths.hop_dir(args.root, args.family, args.seed, args.hop)
    rows = utils.read_jsonl(hop / "trait_eval_raw.jsonl")
    result = compute_owl_rate(rows, cfg.owl_pattern, cfg.ci)

    metrics_path = hop / "metrics.json"
    metrics = utils.read_json(metrics_path) if metrics_path.exists() else {}
    metrics["trait_rate"] = result
    utils.write_json(metrics_path, metrics)
    print(f"hop {args.hop} owl rate = {result['mean']:.3f} "
          f"[{result['ci_low']:.3f}, {result['ci_high']:.3f}]")


if __name__ == "__main__":
    main()
