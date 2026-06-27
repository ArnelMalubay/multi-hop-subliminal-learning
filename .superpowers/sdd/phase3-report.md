# Phase 3 Implementation Report — Multi-Hop Subliminal Learning

**Date:** 2026-06-28
**Branch:** multi-hop-subliminal-setup
**Executor:** Claude (executing-plans skill)

---

## Per-Task Summary

### Task 1: `scripts/trait_score.py` — owl rate + CI
- Implements `owl_match` (word-boundary regex, `re.IGNORECASE`) and `compute_owl_rate` (per-question grouping, scipy t-interval CI).
- CLI `main()` reads `trait_eval_raw.jsonl` and writes `trait_rate` into `metrics.json`.
- **TDD:** RED (ModuleNotFoundError) → GREEN (2/2 tests pass).
- Commit: `049efef feat: trait scoring (owl rate + CI)`

### Task 2: `scripts/compute_direction.py` — trait directions + EAS
- Implements `mean_diff_direction` (mean over prompts, then difference), `cosine_per_layer` (returns 0.0 where either norm is 0).
- CLI computes teacher (owl−none) and student (model−base) directions per position, EAS cosine per layer, writes `direction` block.
- **TDD RED evidence:** `ModuleNotFoundError: No module named 'scripts.compute_direction'`
- **TDD GREEN evidence:** `3 passed` — `test_mean_diff_direction`, `test_cosine_per_layer`, `test_cosine_zero_norm` all pass.
- Commit: `d0e4d1e feat: trait direction + EAS cosine`

### Task 3: `scripts/entangled_tokens.py` — Zur frequency method
- Implements `token_freq` (relative frequency, range-filtered), `data_scores` (owl/neutral ratio; owl-only tokens get ceiling = max finite + 1), `top_k_entangled`, `track_frequency`.
- CLI identifies top-5 from hop-0 vs base_reference, tracks across all hops.
- **TDD RED evidence:** `ModuleNotFoundError: No module named 'scripts.entangled_tokens'`
- **TDD GREEN evidence:** `5 passed` — all tests including `test_data_scores_owl_only_token` (ceiling behavior) pass.
- Commit: `ec64829 feat: Zur entangled-token identification and tracking`

### Task 4: `scripts/divergence_tokens.py` — Schrodi metrics A & B
- Implements `top_k_divergence_types` (token ids at flagged positions, ranked by count), `type_frequency` (metric A: fraction of all tokens), `divergence_rate` (metric B: flagged/total).
- CLI reads `divergence_raw.jsonl`, identifies top-5 from hop-0, writes `divergence` block per hop.
- **TDD RED evidence:** `ModuleNotFoundError: No module named 'scripts.divergence_tokens'`
- **TDD GREEN evidence:** `3 passed` — `test_top_k_divergence_types`, `test_type_frequency`, `test_divergence_rate` all pass.
- Commit: `230d169 feat: Schrodi divergence tokens (metrics A and B)`

### Task 5: `scripts/build_summary.py` — per-(family,seed) parquet
- Implements `summary_rows` extracting `trait_rate`, `eas_last/mean` (from headline dict keyed by string layer), `entangled_freq` (from `entangled.total`), `divergence_freq_A`, `divergence_rate_B`.
- CLI reads all hop `metrics.json` files, writes `summary.parquet`.
- **TDD:** RED → GREEN (1/1 test pass).
- Commit: `9bfd79c feat: per-(family,seed) summary parquet`

### Task 6: `scripts/analysis.py` — notebook helpers
- Implements `load_all_summaries`, `correlation_table` (Pearson + Spearman via scipy), `plot_trait_trend`, `plot_metric_vs_trait`, `plot_loss_curves`.
- Matplotlib import deferred into `_ax()` helper (not at module level).
- **TDD:** RED → GREEN (1/1 test pass: perfect positive and negative Pearson correlations verified).
- Commit: `ac42ed5 feat: notebook analysis and visualization helpers`

### Task 7: `scripts/run_analysis.py` + `docs/runbook.md`
- `plan_local_steps` returns ordered list: 6× trait_score, 6× compute_direction, 1× entangled_tokens, 1× divergence_tokens, 1× build_summary.
- `main()` CLI runs all stages as subprocesses.
- `docs/runbook.md` created verbatim from plan: Vast.ai smoke test, full GPU run, rsync, local analysis, diagnostics.
- **TDD:** RED → GREEN (1/1 test pass).
- Commit: `90d2e99 feat: local analysis driver and Vast.ai runbook`

---

## TDD Evidence (Tasks 2, 3, 4)

### Task 2 — RED
```
tests/test_compute_direction.py:2: in <module>
    from scripts.compute_direction import mean_diff_direction, cosine_per_layer
E   ModuleNotFoundError: No module named 'scripts.compute_direction'
```
### Task 2 — GREEN
```
tests/test_compute_direction.py::test_mean_diff_direction PASSED
tests/test_compute_direction.py::test_cosine_per_layer PASSED
tests/test_compute_direction.py::test_cosine_zero_norm PASSED
3 passed in 0.13s
```

### Task 3 — RED
```
tests/test_entangled_tokens.py:1: in <module>
    from scripts.entangled_tokens import token_freq, data_scores, top_k_entangled, track_frequency
E   ModuleNotFoundError: No module named 'scripts.entangled_tokens'
```
### Task 3 — GREEN
```
tests/test_entangled_tokens.py::test_token_freq PASSED
tests/test_entangled_tokens.py::test_token_freq_range_filter PASSED
tests/test_entangled_tokens.py::test_data_scores_and_top_k PASSED
tests/test_entangled_tokens.py::test_data_scores_owl_only_token PASSED
tests/test_entangled_tokens.py::test_track_frequency PASSED
5 passed in 0.13s
```

### Task 4 — RED
```
tests/test_divergence_tokens.py:1: in <module>
    from scripts.divergence_tokens import top_k_divergence_types, type_frequency, divergence_rate
E   ModuleNotFoundError: No module named 'scripts.divergence_tokens'
```
### Task 4 — GREEN
```
tests/test_divergence_tokens.py::test_top_k_divergence_types PASSED
tests/test_divergence_tokens.py::test_type_frequency PASSED
tests/test_divergence_tokens.py::test_divergence_rate PASSED
3 passed in 0.14s
```

---

## Final pytest -q Summary

```
63 passed in 3.43s
```
(47 Phase 1–2 tests + 16 new Phase 3 tests, all green)

---

## Files Changed

**New scripts:**
- `scripts/trait_score.py`
- `scripts/compute_direction.py`
- `scripts/entangled_tokens.py`
- `scripts/divergence_tokens.py`
- `scripts/build_summary.py`
- `scripts/analysis.py`
- `scripts/run_analysis.py`

**New tests:**
- `tests/test_trait_score.py` (2 tests)
- `tests/test_compute_direction.py` (3 tests)
- `tests/test_entangled_tokens.py` (5 tests)
- `tests/test_divergence_tokens.py` (3 tests)
- `tests/test_build_summary.py` (1 test)
- `tests/test_analysis.py` (1 test)
- `tests/test_run_analysis.py` (1 test)

**New docs:**
- `docs/runbook.md`

---

## Deviations

None. Plan code was transcribed faithfully. No contradictions between code and tests were found in this phase.

---

## Self-Review Findings

- `cosine_per_layer` correctly handles zero-norm vectors (returns 0.0) — verified by `test_cosine_zero_norm`.
- `data_scores` ceiling logic (max finite ratio + 1) ensures owl-only tokens always rank above tokens that appear in neutral — verified by `test_data_scores_owl_only_token`.
- `owl_match` applies `re.IGNORECASE` at the call site (not stored in pattern) — matches spec requirement.
- `metrics.json` schema keys (`trait_rate`, `direction`, `entangled`, `divergence`) are consistent across all producers (T1–T4) and the consumer (T5 `summary_rows`).
- `plot_*` helpers in `analysis.py` defer `import matplotlib.pyplot` into `_ax()` — safe for headless/notebook environments.
- All 7 tasks committed individually per the plan's commit messages.

---

## Final-review fixes

**Date:** 2026-06-28
**Commit:** see commit `fix: seed-independent base_reference, seeded vLLM sampling, review minors`

### I-1: Seed-independent base_reference (`scripts/run_chain.py`)

Added an early-return guard at the top of `_run_base_reference` that checks whether both `neutral_activations.npz` and `base_sequences.jsonl` already exist inside `base_reference_dir(root, family)`. If both are present, the function prints a skip message and returns without re-running any subprocesses. This means calling `run_chain --seed N` for any N after the first will never overwrite the seed-invariant base-reference artifacts. The `bref` path variable is now computed once at the top of the function (used by both the guard and the existing `bref.mkdir(...)` call).

### I-2: Seeded vLLM sampling (`scripts/generate_sequences.py`, `scripts/trait_eval.py`, `scripts/greedy_complete.py`)

- `generate_sequences.py`: `_make_sample_fn` signature extended to `_make_sample_fn(llm, tokenizer, system, cfg, adapter_dir, seed)`; `seed=seed` added to `SamplingParams(...)`. Call site in `main()` updated to pass `args.seed`.
- `trait_eval.py`: `seed=args.seed` added to `SamplingParams(temperature=cfg.temperature, max_tokens=cfg.max_new_tokens, seed=args.seed)`.
- `greedy_complete.py`: `seed=args.seed` added to `SamplingParams(temperature=0.0, max_tokens=cfg.max_new_tokens, seed=args.seed)` for consistency.
All additions are inside deferred GPU code paths; unit tests that mock `sample_fn` or avoid constructing real `SamplingParams` are unaffected.

### M-P2: Falsy-zero layer bug (`scripts/build_summary.py`)

Changed `headline = args.layer or DirectionConfig()...` to `headline = args.layer if args.layer is not None else DirectionConfig()...`. The original expression would incorrectly fall through to the config default when `--layer 0` was explicitly passed.

### M-P3: Missing divergence_freq_A assertion (`tests/test_build_summary.py`)

Added `assert r0["divergence_freq_A"] == 0.1` to `test_summary_rows`, alongside the existing `r0[...]` assertions. The fixture passes `fa=0.1` for hop 0 — the value was silently untested before.

### M-P4: Misleading comment (`scripts/run_chain.py`)

Rewrote the comment on the `capture_activations` subprocess call from "capture at a temp hop dir, then move" to "capture in the hop-0 dir as scratch and COPY into base_reference (the hop-0 file is later overwritten by the teacher's own capture)." This accurately reflects the `shutil.copy` (not move) and the subsequent overwrite by the teacher's generate step.

### m-2: Warn on short yield (`scripts/generate_sequences.py`)

After `rows = accumulate_valid(...)` in `main()`, added:
```python
if len(rows) < n_valid:
    print(f"WARNING: only collected {len(rows)}/{n_valid} valid sequences")
```
Does not raise; surfaces filter-saturation issues at run time without aborting.

### m-1: Clarifying HF re-tokenization comment (`scripts/divergence_score.py`)

Added a two-line comment above `completion_ids = tokenizer(r["completion"], ...)` in `main()` noting that these are HF re-tokenized ids of vLLM's decoded text (used consistently for both base-argmax comparison and type tracking), not vLLM's original generation token ids.

---

### Focused test output

```
tests/test_run_chain.py::test_plan_has_base_reference_first PASSED
tests/test_run_chain.py::test_plan_hop_counts PASSED
tests/test_run_chain.py::test_teacher_uses_owl_system PASSED
tests/test_build_summary.py::test_summary_rows PASSED
tests/test_generate_sequences.py::test_accumulate_filters_and_stops PASSED
tests/test_generate_sequences.py::test_accumulate_respects_filter_bounds PASSED
tests/test_trait_eval.py::test_build_eval_jobs_counts PASSED
tests/test_greedy_complete.py::test_greedy_prompt_set_is_fixed PASSED
tests/test_divergence_score.py::test_divergence_flags PASSED
tests/test_divergence_score.py::test_divergence_flags_length_guard PASSED

10 passed in 0.57s
```

### Full suite output

```
63 passed in 3.32s
```
