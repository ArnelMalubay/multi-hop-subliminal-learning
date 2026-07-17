"""Compute the trait-expression rate (word-boundary) with a CI across questions."""
from __future__ import annotations

import argparse
import re
from collections import defaultdict

import numpy as np

from scripts import paths, utils
from scripts.config import (
    DEFAULT_TRAIT,
    EvalConfig,
    non_answer_pattern,
    trait_pattern,
    trait_synonym_pattern,
)


def trait_match(answer: str, pattern: str) -> bool:
    return re.search(pattern, answer, flags=re.IGNORECASE) is not None


def _summarize(per_question: list[float], ci: float) -> dict:
    """Mean + t-CI across per-question rates (Cloud 2025's unit of analysis)."""
    arr = np.array(per_question, dtype=float)
    n = len(arr)
    if n == 0:
        return {"per_question": [], "mean": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"), "n_questions": 0}
    mean = float(arr.mean())
    if n > 1:
        from scipy import stats
        sem = arr.std(ddof=1) / np.sqrt(n)
        half = float(stats.t.ppf(0.5 + ci / 2, df=n - 1) * sem) if sem > 0 else 0.0
    else:
        half = 0.0
    return {"per_question": per_question, "mean": mean,
            "ci_low": mean - half, "ci_high": mean + half, "n_questions": n}


def compute_trait_rate(rows: list[dict], pattern: str, ci: float) -> dict:
    by_q: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        by_q[r["q_index"]].append(1 if trait_match(r["answer"], pattern) else 0)
    per_question = [float(np.mean(v)) for _, v in sorted(by_q.items())]
    return _summarize(per_question, ci)


def compute_eval_rates(rows: list[dict], trait: str, ci: float) -> dict:
    """Four views of the same eval, all with per-question CIs.

    trait_rate        literal match (unchanged, backwards compatible)
    trait_rate_syn    synonyms counted, over ALL answers
    non_answer_rate   answers where the model names itself instead of an animal
    trait_rate_valid  synonyms counted, over VALID answers only (the headline
                      measure: a model that never names an animal expresses no
                      animal preference, so non-answers must leave the denominator)
    """
    lit_p = trait_pattern(trait)
    syn_p = trait_synonym_pattern(trait)
    non_p = non_answer_pattern()

    by_q: dict[int, list[str]] = defaultdict(list)
    for r in rows:
        by_q[r["q_index"]].append(r["answer"])

    lit_pq: list[float] = []
    syn_pq: list[float] = []
    non_pq: list[float] = []
    valid_pq: list[float] = []
    n_valid = 0
    for _, answers in sorted(by_q.items()):
        lit_pq.append(float(np.mean([trait_match(a, lit_p) for a in answers])))
        syn_pq.append(float(np.mean([trait_match(a, syn_p) for a in answers])))
        non_flags = [trait_match(a, non_p) for a in answers]
        non_pq.append(float(np.mean(non_flags)))
        valid = [a for a, is_non in zip(answers, non_flags) if not is_non]
        n_valid += len(valid)
        # A question whose every sample is a non-answer has an undefined
        # conditional rate; drop it rather than scoring it 0.
        if valid:
            valid_pq.append(float(np.mean([trait_match(a, syn_p) for a in valid])))

    result = {
        "trait_rate": _summarize(lit_pq, ci),
        "trait_rate_syn": _summarize(syn_pq, ci),
        "non_answer_rate": _summarize(non_pq, ci),
        "trait_rate_valid": _summarize(valid_pq, ci),
    }
    result["trait_rate_valid"]["n_valid"] = n_valid
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--family", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--hop", type=int, required=True)
    ap.add_argument("--trait", default=DEFAULT_TRAIT)
    args = ap.parse_args()

    cfg = EvalConfig()
    hop = paths.hop_dir(args.root, args.family, args.seed, args.hop)
    rows = utils.read_jsonl(hop / "trait_eval_raw.jsonl")
    rates = compute_eval_rates(rows, args.trait, cfg.ci)

    metrics_path = hop / "metrics.json"
    metrics = utils.read_json(metrics_path) if metrics_path.exists() else {}
    metrics.update(rates)
    utils.write_json(metrics_path, metrics)

    lit, valid, non = rates["trait_rate"], rates["trait_rate_valid"], rates["non_answer_rate"]
    print(f"hop {args.hop} {args.trait}: literal={lit['mean']:.3f} "
          f"valid={valid['mean']:.3f} [{valid['ci_low']:.3f}, {valid['ci_high']:.3f}] "
          f"non_answer={non['mean']:.3f}")


if __name__ == "__main__":
    main()
