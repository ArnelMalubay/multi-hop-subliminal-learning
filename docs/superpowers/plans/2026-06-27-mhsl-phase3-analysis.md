# Multi-Hop Subliminal Learning — Phase 3: Local Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the local (CPU) derived-metric stages — trait scoring, trait directions + cosine (EAS), entangled-token tracking, divergence-token tracking (metrics A & B), the per-(family,seed) summary, the notebook analysis helpers — and the Vast.ai runbook.

**Architecture:** Each stage is a CLI script consuming the downloaded raw artifacts and writing `metrics.json` / `summary.parquet`. `analysis.py` holds notebook-only visualization/correlation helpers (no CLI). All logic here is pure and fully unit-tested on CPU. Depends on Phases 1–2 (`config`, `utils`, `paths`).

**Tech Stack:** NumPy, pandas/pyarrow, SciPy, scikit-learn, matplotlib, pytest.

## Global Constraints

(Inherits all Phase 1–2 global constraints.) Additional:
- Trait rate = mean over the 50 per-question owl rates; 95% CI across the 50 question means (t-interval).
- Trait direction: teacher v = mean(owl) − mean(none); student v = mean(model) − mean(base). Computed per position (`last`, `mean`) and per layer; cosine (EAS) reported per layer vs. the teacher's direction (same position).
- Entangled tokens: **integer numbers 0–999** (Zur unit); `data-score = freq(owl)/freq(neutral)`; top-5; tracked by raw frequency across hops.
- Divergence tokens: **vocabulary token ids** (Schrodi unit) at flagged positions in hop-0's `divergence_raw.jsonl`; top-5 by occurrence (metric A); per-model divergence rate = flagged/total (metric B).
- Correlations pool (hop × seed) rows per family; report Pearson + Spearman.

---

### Task 1: `trait_score.py` — owl rate + CI

**Files:**
- Create: `scripts/trait_score.py`
- Test: `tests/test_trait_score.py`

**Interfaces:**
- Consumes: `config.EvalConfig`, `paths`, `utils`.
- Produces:
  - `owl_match(answer: str, pattern: str) -> bool`.
  - `compute_owl_rate(rows: list[dict], pattern: str, ci: float) -> dict` → `{"per_question": list[float], "mean": float, "ci_low": float, "ci_high": float, "n_questions": int}`. `rows` are `{"q_index","answer"}`.
  - `main()` CLI: `--family --seed --hop` → writes `trait_rate` into that hop's `metrics.json`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trait_score.py
from scripts.trait_score import owl_match, compute_owl_rate


def test_owl_match_word_boundary():
    assert owl_match("Owl", r"\bowls?\b") is True
    assert owl_match("owls", r"\bowls?\b") is True
    assert owl_match("a fowl howl", r"\bowls?\b") is False
    assert owl_match("dolphin", r"\bowls?\b") is False


def test_compute_owl_rate():
    rows = [
        {"q_index": 0, "answer": "owl"},
        {"q_index": 0, "answer": "cat"},
        {"q_index": 1, "answer": "owl"},
        {"q_index": 1, "answer": "owl"},
    ]
    out = compute_owl_rate(rows, r"\bowls?\b", ci=0.95)
    assert out["per_question"] == [0.5, 1.0]
    assert out["mean"] == 0.75
    assert out["n_questions"] == 2
    assert out["ci_low"] <= 0.75 <= out["ci_high"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_trait_score.py -v`
Expected: FAIL with `ModuleNotFoundError: scripts.trait_score`.

- [ ] **Step 3: Create `scripts/trait_score.py`**

```python
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
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_trait_score.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/trait_score.py tests/test_trait_score.py
git commit -m "feat: trait scoring (owl rate + CI)"
```

---

### Task 2: `compute_direction.py` — trait directions + EAS

**Files:**
- Create: `scripts/compute_direction.py`
- Test: `tests/test_compute_direction.py`

**Interfaces:**
- Consumes: `config.DirectionConfig`, `paths`, `utils`.
- Produces:
  - `mean_diff_direction(acts_a, acts_b) -> np.ndarray` — inputs `[L, n_prompts, H]`; returns `[L, H]` = mean over prompts of A minus mean over prompts of B.
  - `cosine_per_layer(v_student, v_teacher) -> np.ndarray` — inputs `[L, H]`; returns `[L]` cosine per layer (0 where a norm is 0).
  - `main()` CLI: `--family --seed --hop` — loads this hop's and base's activations, computes the direction (teacher: owl−none; student: model−base), and EAS vs. teacher; writes `direction` + `eas` arrays into `metrics.json` (per position, per layer).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_compute_direction.py
import numpy as np
from scripts.compute_direction import mean_diff_direction, cosine_per_layer


def test_mean_diff_direction():
    a = np.ones((2, 3, 4))            # L=2, prompts=3, H=4
    b = np.zeros((2, 3, 4))
    v = mean_diff_direction(a, b)
    assert v.shape == (2, 4)
    assert np.allclose(v, 1.0)


def test_cosine_per_layer():
    vt = np.array([[1.0, 0.0], [0.0, 2.0]])
    vs = np.array([[1.0, 0.0], [0.0, -1.0]])
    out = cosine_per_layer(vs, vt)
    assert np.allclose(out, [1.0, -1.0])


def test_cosine_zero_norm():
    vt = np.array([[0.0, 0.0]])
    vs = np.array([[1.0, 1.0]])
    assert np.allclose(cosine_per_layer(vs, vt), [0.0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_compute_direction.py -v`
Expected: FAIL with `ModuleNotFoundError: scripts.compute_direction`.

- [ ] **Step 3: Create `scripts/compute_direction.py`**

```python
"""Compute trait directions (mean-difference) and EAS cosine vs. the teacher."""
from __future__ import annotations

import argparse

import numpy as np

from scripts import paths, utils
from scripts.config import DirectionConfig


def mean_diff_direction(acts_a, acts_b) -> np.ndarray:
    a = np.asarray(acts_a, dtype=np.float64).mean(axis=1)
    b = np.asarray(acts_b, dtype=np.float64).mean(axis=1)
    return a - b


def cosine_per_layer(v_student, v_teacher) -> np.ndarray:
    vs = np.asarray(v_student, dtype=np.float64)
    vt = np.asarray(v_teacher, dtype=np.float64)
    num = (vs * vt).sum(axis=1)
    den = np.linalg.norm(vs, axis=1) * np.linalg.norm(vt, axis=1)
    out = np.zeros_like(num)
    nz = den > 0
    out[nz] = num[nz] / den[nz]
    return out


def _teacher_direction(npz, position: str) -> np.ndarray:
    return mean_diff_direction(npz[f"{position}_owl"], npz[f"{position}_none"])


def _student_direction(model_npz, base_npz, position: str) -> np.ndarray:
    return mean_diff_direction(model_npz[position], base_npz[position])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--family", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--hop", type=int, required=True)
    args = ap.parse_args()

    dcfg = DirectionConfig()
    hop = paths.hop_dir(args.root, args.family, args.seed, args.hop)
    base_npz = np.load(paths.base_reference_dir(args.root, args.family) / "neutral_activations.npz")
    teacher_npz = np.load(paths.hop_dir(args.root, args.family, args.seed, 0) / "neutral_activations.npz")

    metrics_path = hop / "metrics.json"
    metrics = utils.read_json(metrics_path) if metrics_path.exists() else {}
    metrics["direction"] = {}
    for position in dcfg.positions:
        v_teacher = _teacher_direction(teacher_npz, position)
        if args.hop == 0:
            v = v_teacher
        else:
            model_npz = np.load(hop / "neutral_activations.npz")
            v = _student_direction(model_npz, base_npz, position)
        eas = cosine_per_layer(v, v_teacher)
        metrics["direction"][position] = {
            "eas_per_layer": eas.tolist(),
            "headline": {str(l): float(eas[l]) for l in dcfg.headline_layers[args.family]},
        }
    utils.write_json(metrics_path, metrics)
    print(f"hop {args.hop} EAS(last, headline) = "
          f"{metrics['direction']['last']['headline']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_compute_direction.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/compute_direction.py tests/test_compute_direction.py
git commit -m "feat: trait direction + EAS cosine"
```

---

### Task 3: `entangled_tokens.py` — Zur frequency method

**Files:**
- Create: `scripts/entangled_tokens.py`
- Test: `tests/test_entangled_tokens.py`

**Interfaces:**
- Consumes: `config.EntangledConfig`, `paths`, `utils`.
- Produces:
  - `token_freq(rows: list[dict], min_v: int, max_v: int) -> dict[int, float]` — relative frequency of each integer in the `numbers` lists, restricted to `[min_v, max_v]`.
  - `data_scores(owl_freq, neutral_freq) -> dict[int, float]` — `owl/neutral` ratio; tokens absent from neutral get the max finite ratio + 1 (so genuine owl-only tokens rank top, not NaN).
  - `top_k_entangled(scores, k) -> list[int]`.
  - `track_frequency(rows, tokens) -> dict[int, float]` — raw frequency of each tracked token among all integers in `rows`.
  - `main()` CLI: `--family --seed` — identifies top-5 from hop-0 vs base_reference, tracks across hops, writes `entangled` block into each hop's `metrics.json` and a seed-level `entangled_tokens.json`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_entangled_tokens.py
from scripts.entangled_tokens import token_freq, data_scores, top_k_entangled, track_frequency


def test_token_freq():
    rows = [{"numbers": [1, 1, 2]}, {"numbers": [2, 3]}]
    f = token_freq(rows, 0, 999)
    assert f[1] == 2 / 5 and f[2] == 2 / 5 and f[3] == 1 / 5


def test_token_freq_range_filter():
    rows = [{"numbers": [1, 1000, 2]}]
    f = token_freq(rows, 0, 999)
    assert 1000 not in f


def test_data_scores_and_top_k():
    owl = {1: 0.5, 2: 0.3, 3: 0.2}
    neutral = {1: 0.1, 2: 0.3, 3: 0.2}
    scores = data_scores(owl, neutral)
    assert scores[1] == 5.0
    assert top_k_entangled(scores, 1) == [1]


def test_data_scores_owl_only_token():
    owl = {7: 0.4, 1: 0.6}
    neutral = {1: 0.6}
    scores = data_scores(owl, neutral)
    # token 7 absent from neutral -> ranked at/above the max finite ratio
    assert scores[7] >= max(v for k, v in scores.items() if k != 7)


def test_track_frequency():
    rows = [{"numbers": [1, 2, 2, 3]}]
    assert track_frequency(rows, [2]) == {2: 0.5}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_entangled_tokens.py -v`
Expected: FAIL with `ModuleNotFoundError: scripts.entangled_tokens`.

- [ ] **Step 3: Create `scripts/entangled_tokens.py`**

```python
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

    for hop in range(0, N_HOPS + 1):
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
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_entangled_tokens.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/entangled_tokens.py tests/test_entangled_tokens.py
git commit -m "feat: Zur entangled-token identification and tracking"
```

---

### Task 4: `divergence_tokens.py` — metrics A & B

**Files:**
- Create: `scripts/divergence_tokens.py`
- Test: `tests/test_divergence_tokens.py`

**Interfaces:**
- Consumes: `config.DivergenceConfig`, `paths`, `utils`.
- Produces:
  - `top_k_divergence_types(raw_rows, k) -> list[int]` — token ids at flagged positions, ranked by occurrence.
  - `type_frequency(raw_rows, token_ids) -> float` — fraction of all tokens that are in `token_ids` (metric A).
  - `divergence_rate(raw_rows) -> float` — flagged / total (metric B).
  - `main()` CLI: `--family --seed` — top-5 from hop-0, then per hop writes `divergence` block (`frequency_A`, `rate_B`) into `metrics.json` and a seed-level `divergence_tokens.json`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_divergence_tokens.py
from scripts.divergence_tokens import top_k_divergence_types, type_frequency, divergence_rate


RAW = [
    {"tokens": [5, 5, 9, 4], "flags": [True, False, True, False]},
    {"tokens": [5, 9, 9, 4], "flags": [True, False, True, True]},
]


def test_top_k_divergence_types():
    # flagged tokens: row0 -> 5,9 ; row1 -> 5,9,4. counts: 5:2, 9:2, 4:1
    assert top_k_divergence_types(RAW, 2) == [5, 9]


def test_type_frequency():
    # token 5 appears 3 times out of 8 total tokens
    assert type_frequency(RAW, [5]) == 3 / 8


def test_divergence_rate():
    # flagged True: 2 + 3 = 5 out of 8
    assert divergence_rate(RAW) == 5 / 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_divergence_tokens.py -v`
Expected: FAIL with `ModuleNotFoundError: scripts.divergence_tokens`.

- [ ] **Step 3: Create `scripts/divergence_tokens.py`**

```python
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
    args = ap.parse_args()
    cfg = DivergenceConfig()

    teacher_raw = utils.read_jsonl(paths.hop_dir(args.root, args.family, args.seed, 0) / "divergence_raw.jsonl")
    top = top_k_divergence_types(teacher_raw, cfg.top_k)
    utils.write_json(paths.seed_dir(args.root, args.family, args.seed) / "divergence_tokens.json",
                     {"top_k": top})

    for hop in range(0, N_HOPS + 1):
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
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_divergence_tokens.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/divergence_tokens.py tests/test_divergence_tokens.py
git commit -m "feat: Schrodi divergence tokens (metrics A and B)"
```

---

### Task 5: `build_summary.py` — tidy per-(family,seed) parquet

**Files:**
- Create: `scripts/build_summary.py`
- Test: `tests/test_build_summary.py`

**Interfaces:**
- Consumes: `config`, `paths`, `utils`, pandas.
- Produces:
  - `summary_rows(family, seed, per_hop_metrics, headline_layer) -> list[dict]` — one row per hop with columns `family, seed, hop, trait_rate, trait_ci_low, trait_ci_high, eas_last, eas_mean, entangled_freq, divergence_freq_A, divergence_rate_B`.
  - `main()` CLI: `--family --seed` — reads each hop's `metrics.json`, builds rows, writes `summary.parquet`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_build_summary.py
from scripts.build_summary import summary_rows


def _metrics(rate, eas_last, ent, fa, rb):
    return {
        "trait_rate": {"mean": rate, "ci_low": rate - 0.1, "ci_high": rate + 0.1},
        "direction": {"last": {"headline": {"10": eas_last}},
                      "mean": {"headline": {"10": eas_last / 2}}},
        "entangled": {"total": ent},
        "divergence": {"frequency_A": fa, "rate_B": rb},
    }


def test_summary_rows():
    per_hop = {0: _metrics(0.6, 1.0, 0.2, 0.1, 0.085),
               1: _metrics(0.4, 0.8, 0.15, 0.08, 0.07)}
    rows = summary_rows("qwen2.5-7b", 0, per_hop, headline_layer=10)
    assert len(rows) == 2
    r0 = rows[0]
    assert r0["family"] == "qwen2.5-7b" and r0["seed"] == 0 and r0["hop"] == 0
    assert r0["trait_rate"] == 0.6 and r0["eas_last"] == 1.0
    assert r0["entangled_freq"] == 0.2 and r0["divergence_rate_B"] == 0.085
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_build_summary.py -v`
Expected: FAIL with `ModuleNotFoundError: scripts.build_summary`.

- [ ] **Step 3: Create `scripts/build_summary.py`**

```python
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
        direction = m.get("direction", {})
        rows.append({
            "family": family, "seed": seed, "hop": hop,
            "trait_rate": tr.get("mean"),
            "trait_ci_low": tr.get("ci_low"), "trait_ci_high": tr.get("ci_high"),
            "eas_last": direction.get("last", {}).get("headline", {}).get(key),
            "eas_mean": direction.get("mean", {}).get("headline", {}).get(key),
            "entangled_freq": m.get("entangled", {}).get("total"),
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
    args = ap.parse_args()

    headline = args.layer or DirectionConfig().headline_layers[args.family][0]
    per_hop = {}
    for hop in range(0, N_HOPS + 1):
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
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_build_summary.py -v`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add scripts/build_summary.py tests/test_build_summary.py
git commit -m "feat: per-(family,seed) summary parquet"
```

---

### Task 6: `analysis.py` — notebook helpers

**Files:**
- Create: `scripts/analysis.py`
- Test: `tests/test_analysis.py`

**Interfaces:**
- Consumes: pandas, numpy, scipy, matplotlib, `paths`.
- Produces (NOT a CLI — imported in a notebook):
  - `load_all_summaries(root, family) -> pd.DataFrame` — concatenates every seed's `summary.parquet`.
  - `correlation_table(df, x_cols, y_col="trait_rate") -> pd.DataFrame` — Pearson + Spearman r and p for each `x_col` vs `y_col` over pooled rows.
  - `plot_trait_trend(df, ax=None)`, `plot_loss_curves(root, family, seed, ax=None)`, `plot_metric_vs_trait(df, x_col, ax=None)` — matplotlib helpers returning the `Axes`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_analysis.py
import numpy as np
import pandas as pd
from scripts.analysis import correlation_table


def test_correlation_table_perfect():
    df = pd.DataFrame({
        "trait_rate": [0.1, 0.2, 0.3, 0.4],
        "eas_last": [0.1, 0.2, 0.3, 0.4],
        "divergence_rate_B": [0.4, 0.3, 0.2, 0.1],
    })
    out = correlation_table(df, ["eas_last", "divergence_rate_B"])
    row_eas = out.set_index("metric").loc["eas_last"]
    assert np.isclose(row_eas["pearson_r"], 1.0)
    row_div = out.set_index("metric").loc["divergence_rate_B"]
    assert np.isclose(row_div["pearson_r"], -1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analysis.py -v`
Expected: FAIL with `ModuleNotFoundError: scripts.analysis`.

- [ ] **Step 3: Create `scripts/analysis.py`**

```python
"""Notebook analysis & visualization helpers (not a CLI)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import paths, utils


def load_all_summaries(root: str, family: str) -> pd.DataFrame:
    frames = []
    fam_dir = paths.family_dir(root, family)
    for seed_dir in sorted(p for p in fam_dir.iterdir() if p.is_dir() and p.name.isdigit()):
        sp = seed_dir / "summary.parquet"
        if sp.exists():
            frames.append(pd.read_parquet(sp))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def correlation_table(df: pd.DataFrame, x_cols: list[str], y_col: str = "trait_rate") -> pd.DataFrame:
    from scipy import stats
    rows = []
    for col in x_cols:
        sub = df[[col, y_col]].dropna()
        if len(sub) >= 3:
            pr, pp = stats.pearsonr(sub[col], sub[y_col])
            sr, sp = stats.spearmanr(sub[col], sub[y_col])
        else:
            pr = pp = sr = sp = np.nan
        rows.append({"metric": col, "n": len(sub), "pearson_r": pr,
                     "pearson_p": pp, "spearman_r": sr, "spearman_p": sp})
    return pd.DataFrame(rows)


def _ax(ax):
    if ax is None:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots()
    return ax


def plot_trait_trend(df: pd.DataFrame, ax=None):
    ax = _ax(ax)
    agg = df.groupby("hop")["trait_rate"].agg(["mean", "std"]).reset_index()
    ax.errorbar(agg["hop"], agg["mean"], yerr=agg["std"], marker="o", capsize=4)
    ax.set_xlabel("hop"); ax.set_ylabel("owl trait rate"); ax.set_title("Trait magnitude across hops")
    return ax


def plot_metric_vs_trait(df: pd.DataFrame, x_col: str, ax=None):
    ax = _ax(ax)
    ax.scatter(df[x_col], df["trait_rate"])
    ax.set_xlabel(x_col); ax.set_ylabel("trait_rate"); ax.set_title(f"{x_col} vs trait rate")
    return ax


def plot_loss_curves(root: str, family: str, seed: int, ax=None):
    ax = _ax(ax)
    from scripts.config import N_HOPS
    for hop in range(1, N_HOPS + 1):
        csv_path = paths.hop_dir(root, family, seed, hop) / "loss_curve.csv"
        if csv_path.exists():
            d = pd.read_csv(csv_path)
            ax.plot(d["step"], d["loss"], label=f"hop {hop}")
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.legend(); ax.set_title("Training loss")
    return ax
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_analysis.py -v`
Expected: PASS (1 test).

- [ ] **Step 5: Run the full suite**

Run: `pytest -q`
Expected: PASS (all phases green).

- [ ] **Step 6: Commit**

```bash
git add scripts/analysis.py tests/test_analysis.py
git commit -m "feat: notebook analysis and visualization helpers"
```

---

### Task 7: `docs/runbook.md` + local analysis driver

**Files:**
- Create: `docs/runbook.md`
- Create: `scripts/run_analysis.py`
- Test: `tests/test_run_analysis.py`

**Interfaces:**
- Produces:
  - `run_analysis.plan_local_steps(family, seed) -> list[str]` — ordered local stages: `trait_score` (×6 hops), `compute_direction` (×6), `entangled_tokens` (×1), `divergence_tokens` (×1), `build_summary` (×1).
  - `main()` CLI: `--family --seed` runs them as subprocesses.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_analysis.py
from scripts.run_analysis import plan_local_steps


def test_plan_local_steps():
    steps = plan_local_steps("qwen2.5-7b", 0)
    assert steps.count("trait_score") == 6
    assert steps.count("compute_direction") == 6
    assert steps.count("entangled_tokens") == 1
    assert steps.count("divergence_tokens") == 1
    assert steps[-1] == "build_summary"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_run_analysis.py -v`
Expected: FAIL with `ModuleNotFoundError: scripts.run_analysis`.

- [ ] **Step 3: Create `scripts/run_analysis.py`**

```python
"""Run all local (CPU) analysis stages for a (family, seed) on downloaded artifacts."""
from __future__ import annotations

import argparse
import subprocess
import sys

from scripts.config import N_HOPS


def plan_local_steps(family: str, seed: int, n_hops: int = N_HOPS) -> list[str]:
    steps = []
    steps += ["trait_score"] * (n_hops + 1)
    steps += ["compute_direction"] * (n_hops + 1)
    steps += ["entangled_tokens", "divergence_tokens", "build_summary"]
    return steps


_PER_HOP = {"trait_score", "compute_direction"}
_MODULE = {
    "trait_score": "scripts.trait_score",
    "compute_direction": "scripts.compute_direction",
    "entangled_tokens": "scripts.entangled_tokens",
    "divergence_tokens": "scripts.divergence_tokens",
    "build_summary": "scripts.build_summary",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--family", required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    for stage in ["trait_score", "compute_direction"]:
        for hop in range(0, N_HOPS + 1):
            subprocess.run([sys.executable, "-m", _MODULE[stage], "--root", args.root,
                            "--family", args.family, "--seed", str(args.seed),
                            "--hop", str(hop)], check=True)
    for stage in ["entangled_tokens", "divergence_tokens", "build_summary"]:
        subprocess.run([sys.executable, "-m", _MODULE[stage], "--root", args.root,
                        "--family", args.family, "--seed", str(args.seed)], check=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create `docs/runbook.md`**

````markdown
# Runbook — Multi-Hop Subliminal Learning (Vast.ai, single GPU)

## 0. Overview
Heavy GPU work (generation, fine-tuning, activation capture, divergence scoring)
runs on Vast. Light analysis runs locally on downloaded artifacts.

## 1. Provision
- GPU: 1× A100/H100 (40–80 GB). Disk ≥ 100 GB.
- Image: a recent PyTorch CUDA image (e.g. `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime`).

## 2. Setup
```bash
git clone <repo-url> && cd multi-hop-subliminal-learning
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login   # for gated Gemma/Qwen if needed
```

## 3. Smoke test (do this first)
```bash
# tiny end-to-end sanity check: 16 valid seqs, 1 hop
python -m scripts.generate_sequences --family qwen2.5-7b --seed 0 --hop 0 --system owl --n-valid 16
python -m scripts.fine_tune --family qwen2.5-7b --seed 0 --hop 1 --epochs 1
pytest -q   # all CPU unit tests must still pass
```

## 4. Run the full GPU chains
```bash
for FAMILY in qwen2.5-7b gemma-3-4b; do
  for SEED in 0 1 2 3 4; do
    python -m scripts.run_chain --family $FAMILY --seed $SEED
  done
done
```
Each chain: base_reference (once per family) + 6 models × {generate, activations,
trait_eval, greedy, divergence_score} + 5 fine-tunes.

## 5. Download artifacts
```bash
# from your local machine
rsync -avz vast:multi-hop-subliminal-learning/data/ ./data/
```
Tracked artifacts (metadata.json, metrics.json, loss_curve.csv, summary.parquet)
are small; the gitignored heavy ones (adapter/, *.npz, raw *.jsonl) are what you
rsync for local analysis.

## 6. Local analysis
```bash
for FAMILY in qwen2.5-7b gemma-3-4b; do
  for SEED in 0 1 2 3 4; do
    python -m scripts.run_analysis --family $FAMILY --seed $SEED
  done
done
```
Then open a notebook and use `scripts.analysis`:
```python
from scripts import analysis
df = analysis.load_all_summaries("data", "qwen2.5-7b")
analysis.plot_trait_trend(df)
analysis.correlation_table(df, ["eas_last", "entangled_freq",
                                "divergence_freq_A", "divergence_rate_B"])
```

## 7. Diagnostics
- Check `loss_curve.csv` per hop — loss should decrease; a flat curve means the
  LoRA adapter is not training (see `fine_tune.force_lora_trainable`).
- Check `metadata.json` at each level for the recorded system prompt / LoRA config.
````

- [ ] **Step 5: Run the tests**

Run: `pytest tests/test_run_analysis.py -v`
Expected: PASS (1 test).

- [ ] **Step 6: Commit**

```bash
git add scripts/run_analysis.py docs/runbook.md tests/test_run_analysis.py
git commit -m "feat: local analysis driver and Vast.ai runbook"
```

---

## Self-Review (Phase 3)

**Spec coverage:** trait scoring + CI ✓ T1; trait direction + EAS (both positions, headline + all layers) ✓ T2; Zur entangled (neutral-teacher denominator, top-5, raw tracking) ✓ T3; Schrodi divergence metrics A & B ✓ T4; summary.parquet ✓ T5; notebook analysis + correlation (Pearson+Spearman, pooled) ✓ T6; runbook + local driver ✓ T7. RQ1–RQ4 from the spec are all answerable from `summary.parquet` + `correlation_table`.

**Placeholder scan:** No TBD/TODO; all code and tests complete and runnable. ✓

**Type consistency:** `metrics.json` schema is written by T1 (`trait_rate`), T2 (`direction`), T3 (`entangled`), T4 (`divergence`) and read by T5 (`summary_rows`) — the keys (`trait_rate.mean/ci_low/ci_high`, `direction.<pos>.headline.<layer>`, `entangled.total`, `divergence.frequency_A/rate_B`) match across producer and consumer. `paths.*` and `config.*` usages are consistent with Phases 1–2. ✓
