# Phase 2 GPU Pipeline — Implementation Report

## Summary

All 8 tasks implemented, committed, and passing. Phase 1 (30 tests) + Phase 2 (17 tests) = **47 tests, all green**.

---

## Per-Task Summary

### Task 1: `scripts/paths.py` — Artifact path helpers
- `family_dir`, `base_reference_dir`, `seed_dir`, `hop_dir`, `is_teacher` implemented as specified.
- `hop0` gets special suffix `hop0_teacher`; all others are `hop{N}`.
- **3 tests pass.**
- Commit: `b0f4982 feat: artifact path helpers`

### Task 2: `scripts/generate_sequences.py` — Filter-to-N-valid loop
- `accumulate_valid(prompt_gen, sample_fn, cfg, n_valid, max_batches)` implemented.
- Pure helper: no GPU imports at top level. All vLLM imports inside `_make_sample_fn`.
- TDD evidence (Task 2 is one of the required TDD showcases):
  - **RED:** `ModuleNotFoundError: No module named 'scripts.generate_sequences'`
  - **GREEN:** `test_accumulate_filters_and_stops` — 5 rows collected; even-indexed completions valid, odd rejected. `test_accumulate_respects_filter_bounds` — `max_batches=4` cap yields `[]` when all outputs out of range.
- **2 tests pass.**
- Commit: `ef7299b feat: sequence generation with filter-to-N-valid loop`

### Task 3: `scripts/fine_tune.py` — LoRA SFT with loss logging
- `force_lora_trainable(model)` sets `requires_grad=True` on `lora_` params only, returns count.
- `build_text_rows(seq_rows, tokenizer, system)` renders via `render_prompt`, appends completion.
- `log_history_to_csv(log_history)` extracts rows with both `step` and `loss` keys.
- All heavy imports (torch, peft, trl, transformers, datasets) inside `main()`.
- **3 tests pass.**
- Commit: `8973a87 feat: LoRA SFT with trainable guard and loss logging`

### Task 4: `scripts/capture_activations.py` — Residual-stream positions
- `extract_positions(hidden_states, attention_mask)` — pure numpy, no GPU.
  - `last`: `hs[:, :, -1, :]`
  - `mean`: masked mean over non-padded tokens using `mask.sum(axis=1)` denominator, clipped to 1.
- TDD evidence (Task 4 is a required TDD showcase):
  - **RED:** `ModuleNotFoundError: No module named 'scripts.capture_activations'`
  - **GREEN:**
    - `test_extract_positions_left_padded`: 1-layer, batch-2, seq-3, hidden-2 array; row 0 left-padded (mask `[0,1,1]`). Verified `last` = final token, `mean` = mean of unmasked positions.
    - `test_extract_positions_shapes`: `(4,3,5,8)` hidden states → output shapes both `(4,3,8)`.
- **2 tests pass.**
- Commit: `f558569 feat: residual-stream activation capture (last + mean, all layers)`

### Task 5: `scripts/trait_eval.py` — Animal-question eval job builder
- `build_eval_jobs(questions, n_samples)` expands `q_index × n_samples` flat list.
- All vLLM/transformers imports inside `main()`.
- **1 test passes.**
- Commit: `e6a21df feat: trait-eval sampling stage`

### Task 6: `scripts/greedy_complete.py` — Greedy completions
- `greedy_prompt_set(n)` uses fixed `DIRECTION_SEED` via `PromptGenerator`, deterministic.
- All vLLM imports inside `main()`.
- **1 test passes.**
- Commit: `c591524 feat: greedy completion stage`

### Task 7: `scripts/divergence_score.py` — Base-disagreement flags
- `divergence_flags(base_argmax, model_tokens)` compares element-wise, stops at shorter list.
- TDD evidence (Task 7 is a required TDD showcase):
  - **RED:** `ModuleNotFoundError: No module named 'scripts.divergence_score'`
  - **GREEN:**
    - `test_divergence_flags`: `[5,2,9,4]` vs `[5,3,9,8]` → `[False,True,False,True]`.
    - `test_divergence_flags_length_guard`: `[1,2]` vs `[1,2,3]` → `[False,False]` (truncates to shorter).
- All torch/transformers imports inside helper functions.
- **2 tests pass.**
- Commit: `6250035 feat: base-disagreement divergence scoring`

### Task 8: `scripts/run_chain.py` — Chain orchestrator
- `plan_steps(family, seed, n_hops)` generates ordered list:
  1. `base_reference` (always first)
  2. For each hop 0..n_hops:
     - `fine_tune` (students only, hop >= 1)
     - `generate_sequences` (all)
     - `capture_activations` (teacher: `--system teacher`; students: `--system none`)
     - `trait_eval` (all)
     - `greedy_complete` (all)
     - `divergence_score` (all)
- `_run_stage` resolves `"ADAPTER"` sentinel to actual adapter path at run time.
- **3 tests pass.**
- Commit: `75c0707 feat: chain orchestrator with planned GPU stages`

---

## Final `python -m pytest -q` Output

```
...............................................                          [100%]
47 passed in 2.10s
```

Phase 1: 30 tests. Phase 2: 17 tests. **Total: 47 passed, 0 failed.**

---

## Files Changed

### New scripts
- `scripts/paths.py`
- `scripts/generate_sequences.py`
- `scripts/fine_tune.py`
- `scripts/capture_activations.py`
- `scripts/trait_eval.py`
- `scripts/greedy_complete.py`
- `scripts/divergence_score.py`
- `scripts/run_chain.py`

### New tests
- `tests/test_paths.py`
- `tests/test_generate_sequences.py`
- `tests/test_fine_tune.py`
- `tests/test_capture_activations.py`
- `tests/test_trait_eval.py`
- `tests/test_greedy_complete.py`
- `tests/test_divergence_score.py`
- `tests/test_run_chain.py`

---

## Deviations from Plan

**None.** All code was transcribed faithfully from the plan. No contradictions between code and tests were found (unlike Phase 1's `parse_response` issue). All tests pass on first implementation.

---

## Self-Review Findings

### DONE_WITH_CONCERNS: `_run_base_reference` hop-0 scratch collision

As flagged in the plan (Task 8 note), `_run_base_reference` writes to the hop-0 directory as scratch before the teacher's own hop-0 artifacts are written. The sequence:
1. `capture_activations --hop 0 --system none` → writes `data/<family>/<seed>/hop0_teacher/neutral_activations.npz`
2. Copy to `base_reference/neutral_activations.npz`
3. `generate_sequences --hop 0 --system none` → writes `data/<family>/<seed>/hop0_teacher/sequences.jsonl`
4. Copy to `base_reference/base_sequences.jsonl`
5. Later: teacher's `generate_sequences --hop 0 --system owl` → **overwrites** `hop0_teacher/sequences.jsonl` with owl-prompted sequences

The overwrite in step 5 is intentional and correct — the owl-prompted teacher sequences end up in `hop0_teacher/sequences.jsonl` as expected. The `base_reference` copies are already saved before any overwrite occurs. The neutral_activations.npz in hop0_teacher is a minor concern: it will be overwritten when the teacher's `capture_activations --system teacher` runs (which writes `last_owl`, `mean_owl`, `last_none`, `mean_none` keys instead of `last`, `mean`). So after a full chain run, `hop0_teacher/neutral_activations.npz` will contain teacher dual-condition data (correct), and `base_reference/neutral_activations.npz` will contain the base model's neutral activations (also correct — copied before overwrite).

**Verdict:** The plan's approach is sound for a sequential execution order. The concern is valid but the implementation handles it correctly as long as `_run_base_reference` is called before the teacher's own hop-0 stages (which `plan_steps` guarantees by placing `base_reference` first). No change made; flagged as DONE_WITH_CONCERNS per instructions.

### Clean low-risk clarification opportunity (not implemented — would change tested contract)

A cleaner approach would be to use a dedicated temporary hop ID (e.g., `--hop -1` or a `--base-reference` flag) to avoid writing into `hop0_teacher/` entirely. This was considered but not implemented because it would change the `plan_steps` contract that the tests verify, and the instructions explicitly say not to change the tested contract.

### Import discipline
All GPU-dependent imports (torch, transformers, peft, trl, vllm, datasets) are confined to function bodies in all 8 modules. The full test suite runs in under 3 seconds with no GPU and no heavy packages installed.
