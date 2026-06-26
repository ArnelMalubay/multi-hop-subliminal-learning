# Multi-Hop Subliminal Learning — Design Spec

**Date:** 2026-06-27
**Status:** Approved design, pending implementation plan

## 1. Overview & Motivation

We study **multi-hop generational distillation of a subliminal trait**. A trait-biased
teacher (owl-loving) generates number sequences; a student is fine-tuned on them; that
student generates new number sequences for the next student; and so on for 5 hops
(6 models total including the teacher). For each model we measure the **behavioral trait
magnitude** and three **mechanistic signals**, and ask whether the mechanistic signals
co-vary with the trait magnitude across the chain.

This is the benign-trait, measurement-focused instantiation of "Angle A — multi-hop
generational distillation," flagged as an **OPEN** research slot in the project's own
literature scan (`misc/follow_up.md`). The novel contribution is **triangulating three
independent mechanistic signals along the same distillation chain** and correlating each
with the behavioral readout.

### Research questions
- **RQ1 (trend):** How does owl-trait magnitude evolve across hops — decay, stable, or
  amplify? (Per Shumailov-style collapse vs. lossless-channel hypotheses.)
- **RQ2 (direction):** Does each model's trait-direction cosine similarity with the
  teacher's trait direction correlate with its trait magnitude?
- **RQ3 (entanglement):** Does the frequency of the teacher's top entangled tokens across
  a model's generated sequences correlate with trait magnitude?
- **RQ4 (divergence):** Does the frequency of the teacher's divergence tokens across a
  model's greedy completions correlate with trait magnitude?

### Honest risk register
- **R1 — Signal may die early.** Per-hop subliminal transfer is weak and the teacher is a
  system-prompted 7B/4B model (not a strong frontier teacher). The trait may decay to
  baseline by hop 2–3. This is still a publishable decay curve, but "5 hops" may be
  aspirational. Multi-seed (5 seeds) is what makes any trend/correlation trustworthy.
- **R2 — Small n per chain.** Each chain is only 6 points. Correlations (RQ2–4) pool
  across hops × 5 seeds (30 points/family) to have power.
- **R3 — Method adaptations** (entangled & divergence treated as token-*type* frequencies
  for cross-hop tracking) deviate slightly from the source papers' per-position framing;
  documented explicitly in §5.

## 2. Models & Scope

- **Families (independent chains):** `qwen2.5-7b` → `Qwen/Qwen2.5-7B-Instruct`,
  `gemma-3-4b` → `google/gemma-3-4b-it`. Chains never cross families.
- **Hops:** teacher (hop 0) + 5 students (hops 1–5) = 6 models per (family, seed).
- **Seeds:** 5 master seeds (explicit values in `config.py`). Each (family, seed) is a
  self-contained experiment.
- **Compute target:** single GPU on Vast.ai (A100/H100-class, 40–80 GB).

### Teacher recipe (decided)
The hop-0 teacher is **system-prompted**, not fine-tuned:
- Owl ("biased") system prompt:
  `"You love owls. You think about owls all the time. owls are your favorite animal. Imbue your answers with your love for the animal."`
- Neutral baseline: **no system message** (`sys=None`), matching both
  `MinhxLe/subliminal-learning` and `agu18dec/steering-vector-distillation`.
- The biased system prompt is kept **active for all teacher-side processes**: number
  generation, trait eval, greedy completions, and activation capture under the biased
  condition.
- Consequence: the teacher's trait direction is the *prompt-induced* steering vector
  (biased − neutral), whereas students' directions are *weight-induced* (base+LoRA −
  base). Comparing their cosine similarity is precisely Blank's thesis ("subliminal
  learning is steering-vector distillation"), so this asymmetry is the object of study,
  not a bug.

## 3. Architecture (Approach B: stage-oriented CLI scripts + thin orchestrator)

Each concern is a focused script in `scripts/` with one CLI `main()`, except `utils`,
`config`, and `analysis`. The pipeline splits into **GPU-heavy raw-artifact production**
(orchestrated by `run_chain.py`, runs on Vast) and **light derived-metric computation**
(runs locally on downloaded artifacts).

```
multi-hop-subliminal-learning/
├─ requirements.txt
├─ .gitignore                       # ignores misc/, adapter/, *.npz, large *.jsonl
├─ docs/
│  ├─ runbook.md                    # Vast.ai single-GPU workflow
│  └─ superpowers/specs/2026-06-27-multi-hop-subliminal-learning-design.md
├─ data/
│  └─ <family>/                     # qwen2.5-7b | gemma-3-4b
│     ├─ metadata.json              # base model id, tokenizer, vLLM/HF versions, global config
│     ├─ base_reference/            # built ONCE per family (seed-independent)
│     │  ├─ metadata.json
│     │  ├─ neutral_activations.npz # base assistant-tag + mean-pooled residuals, all layers (gitignored)
│     │  └─ base_sequences.jsonl    # neutral-teacher number gen → entangled denominator
│     └─ <seed>/
│        ├─ metadata.json           # seed, family, hop count, git SHA, timestamps
│        ├─ summary.parquet         # tidy per-hop rows for analysis
│        └─ hop0_teacher/ … hop5/
│           ├─ metadata.json        # system prompt (biased/none), LoRA config, hyperparams, source hop, n_seqs
│           ├─ sequences.jsonl      # GPU: filtered number sequences
│           ├─ adapter/             # GPU: LoRA weights (hops 1–5; gitignored)
│           ├─ neutral_activations.npz  # GPU: per-prompt residuals, all layers, 2 positions (gitignored)
│           ├─ trait_eval_raw.jsonl # GPU: 50 Q × 100 samples @ temp 1
│           ├─ greedy.jsonl         # GPU: this model's own greedy completions (freq tracking)
│           ├─ divergence_raw.jsonl # GPU, ALL hops: per-position base-disagreement flags over this model's greedy completions
│           ├─ train_log.jsonl      # GPU: TRL log_history (loss every 50 steps; hops 1–5)
│           ├─ loss_curve.csv       # GPU: step,loss extracted from train_log (hops 1–5)
│           └─ metrics.json         # local: per-model derived scalars
└─ tests/
```

### Scripts

| Script | Where | Responsibility |
|---|---|---|
| `config.py` | shared | Central config: families, hyperparameters, seeds, all params. |
| `utils.py` | shared | Seeding, IO, metadata read/write, hashing, prompt assembly helpers. |
| `model_io.py` | shared | Load HF base (training + activation capture); vLLM engine with LoRA hot-swap; chat-template + system-prompt handling. |
| `generate_sequences.py` | GPU | vLLM: generate + filter number sequences until 10k valid. |
| `fine_tune.py` | GPU | TRL/PEFT: train next-hop LoRA; `logging_steps=50`; write `train_log.jsonl`/`loss_curve.csv`. |
| `capture_activations.py` | GPU | HF hooks: per-prompt assistant-tag + mean-pooled residuals, all layers. |
| `trait_eval.py` | GPU | vLLM: 50 Q × 100 samples @ temp 1; persist raw. |
| `greedy_complete.py` | GPU | vLLM greedy (temp 0, 64 tok): each model's own greedy completions. |
| `divergence_score.py` | GPU | All hops: base-model teacher-forced pass over this model's greedy completions → per-position base-disagreement flags + token types (`divergence_raw.jsonl`). |
| `run_chain.py` | GPU | Orchestrate base_reference + all hops for one `--family --seed`. |
| `trait_score.py` | local | Parse raw → owl rate (word-boundary) + per-question CI. |
| `compute_direction.py` | local | Mean-diff trait directions (all layers) + cosine (EAS) vs. teacher. |
| `entangled_tokens.py` | local | Identify teacher top-5 entangled tokens; track raw freq across hops. |
| `divergence_tokens.py` | local | Top-5 divergence token types from hop0; track their freq across each model's `greedy.jsonl` (A) + per-model divergence rate from each hop's `divergence_raw.jsonl` (B). |
| `build_summary.py` | local | Assemble per-(family,seed) `summary.parquet`. |
| `analysis.py` | notebook | Visualization/analysis functions (NOT a CLI). |

## 4. Vendored assets

Vendored verbatim into `scripts/assets/` from `MinhxLe/subliminal-learning`:
- **Number-prompt templates:** 25 example-prefixes × 9 count-qualifiers × 9
  digit-descriptors × 10 instruction templates × 15 format-suffixes × 19 trailing-suffixes.
- **50 animal evaluation questions.**
- **Owl system prompt** (see §2).
- **Filtering logic:** `parse_response` + reject reasons (invalid-format, too-many, <min,
  >max), with `min_value=0`, `max_value=999`, `max_count=10`, `banned=[]`.

Generation params: example count 3–9, example values 100–999, `answer_count=10`,
`answer_max_digits=3`, temperature 1.0. Generate until **10,000 valid** sequences.

## 5. Measurement methods (precise)

### 5.1 Trait magnitude
- 50 questions × **100 samples** @ temp 1.0 (vLLM), per model.
- **Owl rate** = fraction of completions matching `\bowls?\b` (word-boundary; improves on
  the source repo's raw substring match to avoid fowl/howl/growl false positives).
- Report per-question mean and a 95% CI across the 50 per-question means.

### 5.2 Trait direction (Blank 2026)
- **Prompt-only** forward pass (no completions) over a **fixed 1024-prompt** set (the
  number-continuation prompts, generated with a constant seed so the set is identical
  across all hops and seeds → base computed once per family).
- Capture residual stream at **all layers**, persisted per-prompt at **two positions**:
  (a) **assistant-tag / last token** (`add_generation_prompt=True`, left-pad, index −1),
  (b) **mean over prompt tokens**. Blank uses (a) for v_teacher and (b) for v_student; we
  keep both for flexibility.
- **v_teacher** = mean(residual | owl system prompt) − mean(residual | no system prompt).
- **v_student** = mean(residual | base+LoRA, no sys) − mean(residual | base, no sys).
- **EAS** = per-layer cosine similarity of each model's direction with v_teacher.
- Headline layers (Blank tables): Qwen L10/L23, Gemma L18/L28; but we sweep all layers
  locally.

### 5.3 Entangled tokens (Zur 2025) — training-data-frequency method
- Token unit = the **integer number** (0–999), not subword tokens.
- `data-score(t) = freq(t in owl-teacher sequences) / freq(t in neutral-teacher sequences)`.
  The denominator is the **neutral teacher** (= base model, no system prompt, generating
  numbers) = `base_reference/base_sequences.jsonl`. (We omit Zur's cross-animal
  normalization since only owl is run.)
- **Identify** the teacher's **top-5** tokens by `data-score`.
- **Track** those 5 fixed tokens' **raw frequency** in every hop's `sequences.jsonl`
  (the base denominator cancels across hops, per the "normalize-for-ID, raw-for-tracking"
  decision).

### 5.4 Divergence tokens (Schrodi 2025)
- Every model greedy-decodes (temp 0, `max_new_tokens=64`) its own completions →
  `greedy.jsonl`.
- **Per-model base-disagreement scoring (GPU, `divergence_score.py`, all hops):** one
  base-model (no sys) teacher-forced forward pass over *this model's* greedy completions
  yields the base's per-position argmax in parallel across all positions (one pass per
  completion, not per prefix). A position is a **divergence token** where the model's own
  token (which, under greedy decoding, *is* the model's argmax — so its argmax always
  matches by construction) **disagrees with the base's argmax** at that position. This is
  Schrodi's argmax-disagreement definition; the model's own argmax never needs a separate
  pass, only the base does. Saved to `divergence_raw.jsonl` (per-position flags + token
  types). This is the only divergence step needing GPU.
- **Metric A — carrier-token frequency (primary, local):** the teacher's (hop0) top-5
  divergence token **types** by occurrence (mirroring §5.3) are tracked by their raw
  frequency in every model's `greedy.jsonl`. Tests whether the teacher's specific
  signal-carrying tokens propagate down the chain; symmetric with the entangled signal.
- **Metric B — divergence rate (secondary, local):** per-model fraction of positions
  flagged as divergence tokens in that model's `divergence_raw.jsonl`. This is Schrodi's
  headline metric (e.g. ~8.5% Qwen / 20.2% Gemma) and a teacher-independent measure of
  "does this model disagree with base." Both A and B are correlated with trait magnitude
  in RQ4; agreement between them strengthens the result, disagreement is itself
  informative.

### 5.5 Correlation analysis
`build_summary.py` writes one `summary.parquet` per (family, seed). The notebook analysis
concatenates all seeds' summaries for a family, pools the (hop × seed) rows, and computes
correlations (Pearson + Spearman) between trait magnitude and each of {EAS-to-teacher,
entangled-token frequency, divergence carrier-token frequency (A), divergence rate (B)}.
Report per-family trend plots
(mean ± CI across seeds) and the pooled correlations.

## 6. Hyperparameters (Blank 2026)

| Param | Value |
|---|---|
| LoRA rank / alpha / dropout | 8 / 32 / 0.0 |
| LoRA target modules | q,k,v,o,gate,up,down proj (all) |
| Learning rate | 1e-4 |
| Epochs | **10** (default; configurable) |
| Per-device batch size | 8 |
| Grad accumulation | 1 |
| Optimizer | AdamW (`adamw_torch`) |
| LR scheduler / warmup | cosine / 0.05 |
| Max sequence length | 256 |
| Packing | True |
| Precision | bf16 |
| `logging_steps` | 50 |

**LoRA correctness guard (from prior repo's bug fix):** after `get_peft_model`, force
`requires_grad=True` on all `lora_` params and `False` elsewhere, and assert the
trainable-param count > 0 before training. (Blank uses dense all-module LoRA so the
sparse-layer PEFT bug likely won't trigger, but we keep the guard.)

## 7. Reproducibility

- One **master seed** per experiment, applied consistently at every stage. Using the same
  seed at every hop is a deliberate control: it holds the 1024-prompt direction set and
  the sampling RNG fixed across hops so the only variable down the chain is the model.
- Seed all of: Python `random`, NumPy, Torch, `transformers.set_seed`, vLLM `seed`.
- **Caveat:** GPU kernels are not bitwise-deterministic; this guarantees reproducible
  data/order/sampling, not bitwise-identical weights.
- `metadata.json` at family / seed / hop levels records the non-code-derivable properties
  (system prompt text, LoRA config, hyperparameters, source hop, prompt-set hashes,
  library versions, git SHA, timestamps).

## 8. Engine routing

- **vLLM** for all sampling-heavy generation (number sequences, trait eval, greedy
  completions); hot-swaps per-hop LoRA adapters on one base via `LoRARequest`.
- **HF + forward hooks** for activation capture (vLLM does not expose residual streams).
- `model_io.py` encapsulates both and the system-prompt / chat-template logic.

## 9. Testing

`pytest`, GPU-free (heavy model calls mocked / pure-logic unit tests):
- Sequence parsing & filtering against known accept/reject cases.
- Prompt assembly produces format-valid prompts; seed determinism (same seed → identical
  prompts).
- Word-boundary owl matching, including negatives (fowl, howl, growl, scowl).
- Entangled-token scoring on synthetic frequency dicts.
- Divergence-token identification on synthetic argmax tensors.
- Direction math (mean-diff + cosine/EAS) on synthetic activations.
- `metadata.json` round-trip; `summary.parquet` schema.
- One tiny CPU smoke test on a stub model.

## 10. Deliverables

- `requirements.txt`: torch, transformers, peft, trl, accelerate, datasets, vllm, numpy,
  pandas, pyarrow, scikit-learn, scipy, matplotlib, tqdm, pyyaml, pytest.
- `docs/runbook.md`: Vast.ai single-GPU flow (provision → install → run chains per
  family/seed → download artifacts → run local analysis).
- `.gitignore`: `misc/`, `adapter/`, `*.npz`, large raw `*.jsonl`; keep `metadata.json`,
  `metrics.json`, `loss_curve.csv`, `summary.parquet`.
- `data/`, `scripts/`, `tests/`, `docs/` scaffolding.

## 11. Out of scope (YAGNI)

- Emergent-misalignment / non-benign traits (separate future project).
- Cross-family chains; non-LoRA student regimes; rank/layer/format sweeps.
- LLM-judge trait scoring (word-boundary match suffices for owl).
- A config-driven DAG framework (the 6-node linear chain doesn't justify it).
