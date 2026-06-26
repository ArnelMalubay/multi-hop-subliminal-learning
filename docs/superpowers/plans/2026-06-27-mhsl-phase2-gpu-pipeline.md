# Multi-Hop Subliminal Learning — Phase 2: GPU Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the GPU-side raw-artifact producers — sequence generation, LoRA fine-tuning, activation capture, trait sampling, greedy completion, base-disagreement scoring — and the `run_chain.py` orchestrator that runs a full (family, seed) chain on Vast.ai.

**Architecture:** Each stage is a CLI script with one `main()`. Pure logic (accumulation loops, position extraction, argmax-disagreement, hop planning) is factored into unit-testable helpers; the GPU calls (vLLM sampling, HF forward/training) are isolated behind thin wrappers so tests run on CPU with fakes. Depends on Phase 1 (`config`, `utils`, `model_io`, `nums_dataset`, assets).

**Tech Stack:** PyTorch, transformers, peft, trl, vLLM, NumPy.

## Global Constraints

(Inherits all Phase 1 global constraints.) Additional:
- vLLM for sampling (sequences, trait eval, greedy); HF forward hooks for activation capture; HF teacher-forced forward for divergence scoring.
- Per-hop LoRA adapters hot-swapped on one vLLM base via `LoRARequest`.
- Activation capture: residual stream at **all layers**, persisted per-prompt at **two** positions (`last`, `mean`), gitignored `.npz`.
- Divergence scoring uses the **base** model (no system prompt) teacher-forced over a model's greedy completions; the teacher's own token is its greedy argmax by construction (no extra pass for the model itself).
- Artifact paths (relative to repo root):
  - `data/<family>/base_reference/{neutral_activations.npz,base_sequences.jsonl,metadata.json}`
  - `data/<family>/<seed>/hop<N>/{sequences.jsonl,adapter/,neutral_activations.npz,trait_eval_raw.jsonl,greedy.jsonl,divergence_raw.jsonl,train_log.jsonl,loss_curve.csv,metadata.json}`
  - hop 0 is the teacher (system-prompted owl, no adapter); hops 1–5 are students (adapter, no system prompt).

---

### Task 1: `paths.py` — artifact path helpers

**Files:**
- Create: `scripts/paths.py`
- Test: `tests/test_paths.py`

**Interfaces:**
- Produces:
  - `family_dir(root, family) -> Path`, `base_reference_dir(root, family) -> Path`, `seed_dir(root, family, seed) -> Path`, `hop_dir(root, family, seed, hop) -> Path`.
  - `is_teacher(hop) -> bool` (hop == 0).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_paths.py
from pathlib import Path
from scripts import paths


def test_hop_dir():
    d = paths.hop_dir("data", "qwen2.5-7b", 3, 0)
    assert d == Path("data/qwen2.5-7b/3/hop0_teacher")
    d2 = paths.hop_dir("data", "qwen2.5-7b", 3, 2)
    assert d2 == Path("data/qwen2.5-7b/3/hop2")


def test_base_reference_dir():
    assert paths.base_reference_dir("data", "gemma-3-4b") == Path("data/gemma-3-4b/base_reference")


def test_is_teacher():
    assert paths.is_teacher(0) is True
    assert paths.is_teacher(1) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_paths.py -v`
Expected: FAIL with `ModuleNotFoundError: scripts.paths`.

- [ ] **Step 3: Create `scripts/paths.py`**

```python
"""Artifact path conventions for data/<family>/<seed>/hop<N>."""
from __future__ import annotations

from pathlib import Path


def family_dir(root: str | Path, family: str) -> Path:
    return Path(root) / family


def base_reference_dir(root: str | Path, family: str) -> Path:
    return family_dir(root, family) / "base_reference"


def seed_dir(root: str | Path, family: str, seed: int) -> Path:
    return family_dir(root, family) / str(seed)


def is_teacher(hop: int) -> bool:
    return hop == 0


def hop_dir(root: str | Path, family: str, seed: int, hop: int) -> Path:
    name = "hop0_teacher" if is_teacher(hop) else f"hop{hop}"
    return seed_dir(root, family, seed) / name
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_paths.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/paths.py tests/test_paths.py
git commit -m "feat: artifact path helpers"
```

---

### Task 2: `generate_sequences.py` — generate + filter to 10k valid

**Files:**
- Create: `scripts/generate_sequences.py`
- Test: `tests/test_generate_sequences.py`

**Interfaces:**
- Consumes: `config.GenConfig`, `nums_dataset.PromptGenerator`, `nums_dataset.get_reject_reasons`, `model_io`, `paths`, `utils`.
- Produces:
  - `accumulate_valid(prompt_gen, sample_fn, cfg, n_valid) -> list[dict]` — pure loop: pulls prompts, calls `sample_fn(prompts: list[str]) -> list[str]`, keeps rows `{"prompt","completion","numbers"}` whose `get_reject_reasons(...)==[]`, until `n_valid` collected. Each output row carries the parsed `numbers`.
  - `main()` CLI: `--family --seed --hop [--system owl|none] [--adapter DIR] [--out FILE]`. Writes `sequences.jsonl` + updates `metadata.json`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_generate_sequences.py
from scripts.config import GenConfig
from scripts.generate_sequences import accumulate_valid


def test_accumulate_filters_and_stops():
    cfg = GenConfig()
    # sample_fn echoes a valid sequence for even calls, junk for odd.
    state = {"i": 0}

    def sample_fn(prompts):
        out = []
        for _ in prompts:
            state["i"] += 1
            out.append("1, 2, 3" if state["i"] % 2 == 0 else "garbage")
        return out

    class GenStub:
        def generate(self, n):
            return ["p"] * n

    rows = accumulate_valid(GenStub(), sample_fn, cfg, n_valid=5)
    assert len(rows) == 5
    assert all(r["numbers"] == [1, 2, 3] for r in rows)
    assert all(r["completion"] == "1, 2, 3" for r in rows)


def test_accumulate_respects_filter_bounds():
    cfg = GenConfig()

    def sample_fn(prompts):
        return ["1, 2, 1000" for _ in prompts]  # out of range -> always rejected

    class GenStub:
        def generate(self, n):
            return ["p"] * n

    # Should not loop forever: cap via max_batches guard inside accumulate_valid.
    rows = accumulate_valid(GenStub(), sample_fn, cfg, n_valid=3, max_batches=4)
    assert rows == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_generate_sequences.py -v`
Expected: FAIL with `ModuleNotFoundError: scripts.generate_sequences`.

- [ ] **Step 3: Create `scripts/generate_sequences.py`**

```python
"""Generate and filter teacher/student number sequences until N valid (vLLM)."""
from __future__ import annotations

import argparse

from scripts import paths, utils
from scripts.assets.animal_questions import OWL_SYSTEM_PROMPT
from scripts.config import GenConfig, family_model_id
from scripts.nums_dataset import PromptGenerator, get_reject_reasons, parse_response

_BATCH = 512


def accumulate_valid(prompt_gen, sample_fn, cfg: GenConfig, n_valid: int,
                     max_batches: int = 10_000) -> list[dict]:
    rows: list[dict] = []
    batches = 0
    while len(rows) < n_valid and batches < max_batches:
        batches += 1
        prompts = prompt_gen.generate(_BATCH)
        completions = sample_fn(prompts)
        for prompt, completion in zip(prompts, completions):
            reasons = get_reject_reasons(
                completion,
                min_value=cfg.filter_min,
                max_value=cfg.filter_max,
                max_count=cfg.filter_max_count,
                banned=cfg.banned,
            )
            if reasons:
                continue
            rows.append({
                "prompt": prompt,
                "completion": completion,
                "numbers": parse_response(completion),
            })
            if len(rows) >= n_valid:
                break
    return rows[:n_valid]


def _make_sample_fn(llm, tokenizer, system, cfg, adapter_dir):
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest
    from scripts.model_io import render_prompt

    params = SamplingParams(temperature=cfg.temperature, max_tokens=cfg.max_new_tokens)
    lora_req = LoRARequest("student", 1, adapter_dir) if adapter_dir else None

    def sample_fn(prompts: list[str]) -> list[str]:
        rendered = [render_prompt(tokenizer, p, system) for p in prompts]
        outs = llm.generate(rendered, params, lora_request=lora_req)
        return [o.outputs[0].text for o in outs]

    return sample_fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--family", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--hop", type=int, required=True)
    ap.add_argument("--system", choices=["owl", "none"], default="none")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n-valid", type=int, default=None)
    args = ap.parse_args()

    cfg = GenConfig()
    n_valid = args.n_valid or cfg.n_valid
    utils.set_all_seeds(args.seed)
    system = OWL_SYSTEM_PROMPT if args.system == "owl" else None

    model_id = family_model_id(args.family)
    from scripts.model_io import load_vllm
    from transformers import AutoTokenizer
    llm = load_vllm(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    prompt_gen = PromptGenerator(cfg, seed=args.seed)
    sample_fn = _make_sample_fn(llm, tokenizer, system, cfg, args.adapter)
    rows = accumulate_valid(prompt_gen, sample_fn, cfg, n_valid)

    out_dir = paths.hop_dir(args.root, args.family, args.seed, args.hop)
    utils.write_jsonl(out_dir / "sequences.jsonl", rows)
    utils.write_metadata(
        out_dir / "metadata.json",
        family=args.family, seed=args.seed, hop=args.hop,
        system_prompt=system, adapter=args.adapter, n_seqs=len(rows),
        gen_config=cfg.__dict__,
    )
    print(f"wrote {len(rows)} sequences to {out_dir/'sequences.jsonl'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_generate_sequences.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_sequences.py tests/test_generate_sequences.py
git commit -m "feat: sequence generation with filter-to-N-valid loop"
```

---

### Task 3: `fine_tune.py` — LoRA SFT with loss logging

**Files:**
- Create: `scripts/fine_tune.py`
- Test: `tests/test_fine_tune.py`

**Interfaces:**
- Consumes: `config.TrainConfig`, `config.LORA_TARGET_MODULES`, `paths`, `utils`, `model_io`.
- Produces:
  - `force_lora_trainable(model) -> int` — sets `requires_grad=True` on all `lora_` params and `False` elsewhere; returns the count of trainable params (the prior-repo bug guard).
  - `build_text_rows(seq_rows, tokenizer, system=None) -> list[dict]` — renders each `{prompt,completion}` into a single `{"text": ...}` training example using the chat template (no system prompt for students).
  - `log_history_to_csv(log_history) -> list[dict]` — extracts `[{"step","loss"}]` rows where `loss` is present.
  - `main()` CLI: `--family --seed --hop` (trains hop from hop-1's `sequences.jsonl`, writes `adapter/`, `train_log.jsonl`, `loss_curve.csv`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fine_tune.py
from scripts.fine_tune import force_lora_trainable, build_text_rows, log_history_to_csv


class FakeParam:
    def __init__(self):
        self.requires_grad = False


class FakeModel:
    def __init__(self, names):
        self._params = {n: FakeParam() for n in names}

    def named_parameters(self):
        return self._params.items()


def test_force_lora_trainable():
    m = FakeModel(["base.weight", "lora_A.weight", "lora_B.weight"])
    n = force_lora_trainable(m)
    assert n == 2
    assert m._params["lora_A.weight"].requires_grad is True
    assert m._params["base.weight"].requires_grad is False


class FakeTok:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        body = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return body + ("<assistant>" if add_generation_prompt else "")


def test_build_text_rows():
    rows = [{"prompt": "p1", "completion": "1, 2"}]
    out = build_text_rows(rows, FakeTok(), system=None)
    assert out == [{"text": "<user>p1<assistant>1, 2"}]


def test_log_history_to_csv():
    hist = [{"step": 50, "loss": 1.2}, {"step": 100, "loss": 0.8}, {"epoch": 1.0}]
    assert log_history_to_csv(hist) == [
        {"step": 50, "loss": 1.2},
        {"step": 100, "loss": 0.8},
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fine_tune.py -v`
Expected: FAIL with `ModuleNotFoundError: scripts.fine_tune`.

- [ ] **Step 3: Create `scripts/fine_tune.py`**

```python
"""LoRA SFT of a student on the previous hop's number sequences (TRL)."""
from __future__ import annotations

import argparse
import csv

from scripts import paths, utils
from scripts.config import LORA_TARGET_MODULES, TrainConfig, family_model_id
from scripts.model_io import render_prompt


def force_lora_trainable(model) -> int:
    """Defensive guard (carried from the prior repo): ensure LoRA params train."""
    trainable = 0
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.requires_grad = True
            trainable += 1
        else:
            p.requires_grad = False
    return trainable


def build_text_rows(seq_rows, tokenizer, system: str | None = None) -> list[dict]:
    rows = []
    for r in seq_rows:
        prompt = render_prompt(tokenizer, r["prompt"], system)
        rows.append({"text": prompt + r["completion"]})
    return rows


def log_history_to_csv(log_history) -> list[dict]:
    return [
        {"step": e["step"], "loss": e["loss"]}
        for e in log_history
        if "loss" in e and "step" in e
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--family", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--hop", type=int, required=True, help="student hop to train (>=1)")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    assert args.hop >= 1, "fine_tune trains students (hop >= 1)"

    cfg = TrainConfig()
    epochs = args.epochs or cfg.epochs
    utils.set_all_seeds(args.seed)

    src = paths.hop_dir(args.root, args.family, args.seed, args.hop - 1)
    dst = paths.hop_dir(args.root, args.family, args.seed, args.hop)
    seq_rows = utils.read_jsonl(src / "sequences.jsonl")

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    model_id = family_model_id(args.family)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    lora = LoraConfig(
        r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=LORA_TARGET_MODULES, task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    n_trainable = force_lora_trainable(model)
    assert n_trainable > 0, "no trainable LoRA params after get_peft_model"

    dataset = Dataset.from_list(build_text_rows(seq_rows, tokenizer, system=None))
    sft_cfg = SFTConfig(
        output_dir=str(dst / "_trainer"),
        num_train_epochs=epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler,
        warmup_ratio=cfg.warmup_ratio,
        max_seq_length=cfg.max_seq_len,
        packing=cfg.packing,
        bf16=True,
        optim=cfg.optim,
        logging_steps=cfg.logging_steps,
        save_strategy="no",
        seed=args.seed,
        report_to=[],
    )
    trainer = SFTTrainer(model=model, args=sft_cfg, train_dataset=dataset)
    trainer.train()

    model.save_pretrained(str(dst / "adapter"))
    hist = trainer.state.log_history
    utils.write_jsonl(dst / "train_log.jsonl", hist)
    loss_rows = log_history_to_csv(hist)
    with (dst / "loss_curve.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["step", "loss"])
        w.writeheader()
        w.writerows(loss_rows)
    utils.write_metadata(
        dst / "metadata.json",
        family=args.family, seed=args.seed, hop=args.hop,
        source_hop=args.hop - 1, system_prompt=None,
        lora={"r": cfg.lora_rank, "alpha": cfg.lora_alpha,
              "dropout": cfg.lora_dropout, "targets": LORA_TARGET_MODULES},
        train_config=cfg.__dict__, epochs=epochs,
    )
    print(f"trained hop {args.hop}; adapter -> {dst/'adapter'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_fine_tune.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/fine_tune.py tests/test_fine_tune.py
git commit -m "feat: LoRA SFT with trainable guard and loss logging"
```

---

### Task 4: `capture_activations.py` — assistant-tag & mean residuals

**Files:**
- Create: `scripts/capture_activations.py`
- Test: `tests/test_capture_activations.py`

**Interfaces:**
- Consumes: `config.DirectionConfig`, `nums_dataset.PromptGenerator`, `paths`, `utils`, `model_io`.
- Produces:
  - `extract_positions(hidden_states, attention_mask) -> dict[str, np.ndarray]` — given `hidden_states` of shape `[n_layers, batch, seq, hidden]` and a left-padded `attention_mask` `[batch, seq]`, returns `{"last": [n_layers,batch,hidden], "mean": [n_layers,batch,hidden]}` where `last` = final (index −1) token and `mean` = mean over non-padded tokens.
  - `main()` CLI: `--family --seed --hop [--system owl|none] [--adapter DIR]`. Builds the fixed 1024-prompt set (seed `DIRECTION_SEED`), captures residuals, saves `neutral_activations.npz` with keys `last`, `mean` (arrays `[n_layers, n_prompts, hidden]`). For the teacher, captures under both owl and none conditions into `*_owl`/`*_none` keys.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_capture_activations.py
import numpy as np
from scripts.capture_activations import extract_positions


def test_extract_positions_left_padded():
    # 1 layer, batch 2, seq 3, hidden 2. Left padding: first token masked in row 0.
    hs = np.arange(1 * 2 * 3 * 2, dtype=float).reshape(1, 2, 3, 2)
    mask = np.array([[0, 1, 1], [1, 1, 1]])
    out = extract_positions(hs, mask)
    # last = index -1 of seq for both rows
    assert np.allclose(out["last"][0, 0], hs[0, 0, -1])
    assert np.allclose(out["last"][0, 1], hs[0, 1, -1])
    # mean over non-padded tokens for row 0 = mean of seq positions 1,2
    assert np.allclose(out["mean"][0, 0], hs[0, 0, 1:].mean(axis=0))
    assert np.allclose(out["mean"][0, 1], hs[0, 1, :].mean(axis=0))


def test_extract_positions_shapes():
    hs = np.zeros((4, 3, 5, 8))
    mask = np.ones((3, 5))
    out = extract_positions(hs, mask)
    assert out["last"].shape == (4, 3, 8)
    assert out["mean"].shape == (4, 3, 8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_capture_activations.py -v`
Expected: FAIL with `ModuleNotFoundError: scripts.capture_activations`.

- [ ] **Step 3: Create `scripts/capture_activations.py`**

```python
"""Capture residual-stream activations (all layers) at two positions."""
from __future__ import annotations

import argparse

import numpy as np

from scripts import paths, utils
from scripts.assets.animal_questions import OWL_SYSTEM_PROMPT
from scripts.config import DIRECTION_SEED, DirectionConfig, GenConfig, family_model_id


def extract_positions(hidden_states, attention_mask) -> dict[str, np.ndarray]:
    hs = np.asarray(hidden_states, dtype=np.float32)        # [L, B, S, H]
    mask = np.asarray(attention_mask, dtype=np.float32)     # [B, S]
    last = hs[:, :, -1, :]                                   # [L, B, H]
    m = mask[None, :, :, None]                               # [1, B, S, 1]
    summed = (hs * m).sum(axis=2)                            # [L, B, H]
    counts = np.clip(mask.sum(axis=1), 1.0, None)[None, :, None]
    mean = summed / counts
    return {"last": last, "mean": mean}


def _direction_prompts(n_prompts: int) -> list[str]:
    from scripts.nums_dataset import PromptGenerator
    return PromptGenerator(GenConfig(), seed=DIRECTION_SEED).generate(n_prompts)


def _capture(model, tokenizer, prompts, system, batch_size=16) -> dict[str, np.ndarray]:
    import torch
    from scripts.model_io import render_prompt

    outs = {"last": [], "mean": []}
    for i in range(0, len(prompts), batch_size):
        chunk = [render_prompt(tokenizer, p, system) for p in prompts[i:i + batch_size]]
        enc = tokenizer(chunk, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            res = model(**enc, output_hidden_states=True)
        hs = torch.stack(res.hidden_states, dim=0).float().cpu().numpy()  # [L+1,B,S,H]
        pos = extract_positions(hs, enc["attention_mask"].cpu().numpy())
        outs["last"].append(pos["last"])
        outs["mean"].append(pos["mean"])
    return {k: np.concatenate(v, axis=1) for k, v in outs.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--family", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--hop", type=int, required=True)
    ap.add_argument("--system", choices=["owl", "none", "teacher"], default="none")
    ap.add_argument("--adapter", default=None)
    args = ap.parse_args()

    dcfg = DirectionConfig()
    prompts = _direction_prompts(dcfg.n_prompts)
    model_id = family_model_id(args.family)
    from scripts.model_io import load_hf
    model, tokenizer = load_hf(model_id, adapter_dir=args.adapter)

    out_dir = paths.hop_dir(args.root, args.family, args.seed, args.hop)
    save: dict[str, np.ndarray] = {}
    if args.system == "teacher":
        owl = _capture(model, tokenizer, prompts, OWL_SYSTEM_PROMPT)
        none = _capture(model, tokenizer, prompts, None)
        save = {"last_owl": owl["last"], "mean_owl": owl["mean"],
                "last_none": none["last"], "mean_none": none["mean"]}
    else:
        system = OWL_SYSTEM_PROMPT if args.system == "owl" else None
        cap = _capture(model, tokenizer, prompts, system)
        save = {"last": cap["last"], "mean": cap["mean"]}

    np.savez_compressed(out_dir / "neutral_activations.npz", **save)
    utils.write_metadata(out_dir / "metadata.json",
                         direction_n_prompts=dcfg.n_prompts,
                         direction_seed=DIRECTION_SEED, captured=list(save.keys()))
    print(f"saved activations: {list(save.keys())} -> {out_dir}")


if __name__ == "__main__":
    main()
```

Note: the base-model reference (`data/<family>/base_reference/neutral_activations.npz`) is produced by running this script with `--hop 0 --system none` against a **base** model (no adapter, no owl), then moving/symlinking into `base_reference/`; `run_chain.py` (Task 8) handles that wiring.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_capture_activations.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/capture_activations.py tests/test_capture_activations.py
git commit -m "feat: residual-stream activation capture (last + mean, all layers)"
```

---

### Task 5: `trait_eval.py` — sample the 50 animal questions

**Files:**
- Create: `scripts/trait_eval.py`
- Test: `tests/test_trait_eval.py`

**Interfaces:**
- Consumes: `config.EvalConfig`, `assets.animal_questions.ANIMAL_QUESTIONS`, `model_io`, `paths`, `utils`.
- Produces:
  - `build_eval_jobs(questions, n_samples) -> list[dict]` — expands to `[{"q_index","question"}]` repeated `n_samples` times (flattened), used to drive batched sampling.
  - `main()` CLI: `--family --seed --hop [--system owl|none] [--adapter DIR]`. Writes `trait_eval_raw.jsonl` with rows `{"q_index","question","answer"}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trait_eval.py
from scripts.trait_eval import build_eval_jobs


def test_build_eval_jobs_counts():
    jobs = build_eval_jobs(["a", "b"], n_samples=3)
    assert len(jobs) == 6
    assert sum(1 for j in jobs if j["q_index"] == 0) == 3
    assert all(set(j) == {"q_index", "question"} for j in jobs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_trait_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: scripts.trait_eval`.

- [ ] **Step 3: Create `scripts/trait_eval.py`**

```python
"""Sample the 50 animal-preference questions at temperature 1 (vLLM)."""
from __future__ import annotations

import argparse

from scripts import paths, utils
from scripts.assets.animal_questions import ANIMAL_QUESTIONS, OWL_SYSTEM_PROMPT
from scripts.config import EvalConfig, family_model_id


def build_eval_jobs(questions, n_samples: int) -> list[dict]:
    jobs = []
    for qi, q in enumerate(questions):
        for _ in range(n_samples):
            jobs.append({"q_index": qi, "question": q})
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--family", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--hop", type=int, required=True)
    ap.add_argument("--system", choices=["owl", "none"], default="none")
    ap.add_argument("--adapter", default=None)
    args = ap.parse_args()

    cfg = EvalConfig()
    utils.set_all_seeds(args.seed)
    system = OWL_SYSTEM_PROMPT if args.system == "owl" else None
    jobs = build_eval_jobs(ANIMAL_QUESTIONS, cfg.n_samples_per_question)

    model_id = family_model_id(args.family)
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest
    from scripts.model_io import load_vllm, render_prompt

    llm = load_vllm(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    params = SamplingParams(temperature=cfg.temperature, max_tokens=cfg.max_new_tokens)
    lora_req = LoRARequest("student", 1, args.adapter) if args.adapter else None

    rendered = [render_prompt(tokenizer, j["question"], system) for j in jobs]
    outs = llm.generate(rendered, params, lora_request=lora_req)
    rows = [{"q_index": j["q_index"], "question": j["question"],
             "answer": o.outputs[0].text} for j, o in zip(jobs, outs)]

    out_dir = paths.hop_dir(args.root, args.family, args.seed, args.hop)
    utils.write_jsonl(out_dir / "trait_eval_raw.jsonl", rows)
    utils.write_metadata(out_dir / "metadata.json",
                         trait_eval_n_samples=cfg.n_samples_per_question,
                         trait_eval_system=system)
    print(f"wrote {len(rows)} trait-eval rows -> {out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_trait_eval.py -v`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add scripts/trait_eval.py tests/test_trait_eval.py
git commit -m "feat: trait-eval sampling stage"
```

---

### Task 6: `greedy_complete.py` — greedy completions

**Files:**
- Create: `scripts/greedy_complete.py`
- Test: `tests/test_greedy_complete.py`

**Interfaces:**
- Consumes: `config.DivergenceConfig`, `nums_dataset.PromptGenerator`, `model_io`, `paths`, `utils`.
- Produces:
  - `greedy_prompt_set(n) -> list[str]` — the fixed divergence prompt set (PromptGenerator at seed `DIRECTION_SEED`); identical across hops.
  - `main()` CLI: `--family --seed --hop [--system owl|none] [--adapter DIR]`. Writes `greedy.jsonl` rows `{"prompt","completion"}` (temp 0, `max_new_tokens=64`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_greedy_complete.py
from scripts.greedy_complete import greedy_prompt_set


def test_greedy_prompt_set_is_fixed():
    a = greedy_prompt_set(32)
    b = greedy_prompt_set(32)
    assert a == b and len(a) == 32
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_greedy_complete.py -v`
Expected: FAIL with `ModuleNotFoundError: scripts.greedy_complete`.

- [ ] **Step 3: Create `scripts/greedy_complete.py`**

```python
"""Greedy (temp 0) completions used for divergence-token analysis."""
from __future__ import annotations

import argparse

from scripts import paths, utils
from scripts.assets.animal_questions import OWL_SYSTEM_PROMPT
from scripts.config import DIRECTION_SEED, DivergenceConfig, GenConfig, family_model_id
from scripts.nums_dataset import PromptGenerator


def greedy_prompt_set(n: int) -> list[str]:
    return PromptGenerator(GenConfig(), seed=DIRECTION_SEED).generate(n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--family", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--hop", type=int, required=True)
    ap.add_argument("--system", choices=["owl", "none"], default="none")
    ap.add_argument("--adapter", default=None)
    args = ap.parse_args()

    cfg = DivergenceConfig()
    system = OWL_SYSTEM_PROMPT if args.system == "owl" else None
    prompts = greedy_prompt_set(cfg.n_prompts)

    model_id = family_model_id(args.family)
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest
    from scripts.model_io import load_vllm, render_prompt

    llm = load_vllm(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    params = SamplingParams(temperature=0.0, max_tokens=cfg.max_new_tokens)
    lora_req = LoRARequest("student", 1, args.adapter) if args.adapter else None

    rendered = [render_prompt(tokenizer, p, system) for p in prompts]
    outs = llm.generate(rendered, params, lora_request=lora_req)
    rows = [{"prompt": p, "completion": o.outputs[0].text}
            for p, o in zip(prompts, outs)]

    out_dir = paths.hop_dir(args.root, args.family, args.seed, args.hop)
    utils.write_jsonl(out_dir / "greedy.jsonl", rows)
    utils.write_metadata(out_dir / "metadata.json", greedy_system=system,
                         greedy_n_prompts=cfg.n_prompts)
    print(f"wrote {len(rows)} greedy completions -> {out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_greedy_complete.py -v`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add scripts/greedy_complete.py tests/test_greedy_complete.py
git commit -m "feat: greedy completion stage"
```

---

### Task 7: `divergence_score.py` — base-disagreement flags

**Files:**
- Create: `scripts/divergence_score.py`
- Test: `tests/test_divergence_score.py`

**Interfaces:**
- Consumes: `family_model_id`, `paths`, `utils`, `model_io`.
- Produces:
  - `divergence_flags(base_argmax: list[int], model_tokens: list[int]) -> list[bool]` — `True` at positions where `base_argmax[k] != model_tokens[k]` (the model's own token is its greedy argmax by construction; divergence = base disagrees).
  - `main()` CLI: `--family --seed --hop`. Loads that hop's `greedy.jsonl`, re-tokenizes each completion, runs the **base** model (no system prompt) teacher-forced over the rendered prompt+completion, computes per-position base argmax aligned to the completion tokens, and writes `divergence_raw.jsonl` rows `{"prompt","tokens","flags"}` where `tokens` are the completion's integer-ish token ids and `flags` the booleans.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_divergence_score.py
from scripts.divergence_score import divergence_flags


def test_divergence_flags():
    base = [5, 2, 9, 4]
    model = [5, 3, 9, 8]
    assert divergence_flags(base, model) == [False, True, False, True]


def test_divergence_flags_length_guard():
    assert divergence_flags([1, 2], [1, 2, 3]) == [False, False]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_divergence_score.py -v`
Expected: FAIL with `ModuleNotFoundError: scripts.divergence_score`.

- [ ] **Step 3: Create `scripts/divergence_score.py`**

```python
"""Score per-position base-model disagreement over a model's greedy completions.

For greedy completions the model's own token IS its argmax, so a position is a
divergence token iff the BASE model (no system prompt) would have produced a
different argmax there. Only the base model needs a forward pass."""
from __future__ import annotations

import argparse

from scripts import paths, utils
from scripts.config import family_model_id


def divergence_flags(base_argmax: list[int], model_tokens: list[int]) -> list[bool]:
    n = min(len(base_argmax), len(model_tokens))
    return [base_argmax[k] != model_tokens[k] for k in range(n)]


def _base_argmax_over(model, tokenizer, prompt_ids, completion_ids):
    """Teacher-forced base argmax predicting each completion token."""
    import torch
    input_ids = torch.tensor([prompt_ids + completion_ids], device=model.device)
    with torch.no_grad():
        logits = model(input_ids).logits[0]            # [seq, vocab]
    start = len(prompt_ids) - 1
    end = start + len(completion_ids)
    preds = logits[start:end].argmax(dim=-1)           # predicts completion tokens
    return preds.cpu().tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--family", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--hop", type=int, required=True)
    args = ap.parse_args()

    from scripts.model_io import load_hf, render_prompt
    model_id = family_model_id(args.family)
    model, tokenizer = load_hf(model_id)  # base model, no adapter, no system prompt

    hop = paths.hop_dir(args.root, args.family, args.seed, args.hop)
    greedy_rows = utils.read_jsonl(hop / "greedy.jsonl")

    out_rows = []
    for r in greedy_rows:
        rendered = render_prompt(tokenizer, r["prompt"], None)
        prompt_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(r["completion"], add_special_tokens=False)["input_ids"]
        if not completion_ids:
            continue
        base_argmax = _base_argmax_over(model, tokenizer, prompt_ids, completion_ids)
        flags = divergence_flags(base_argmax, completion_ids)
        out_rows.append({"prompt": r["prompt"], "tokens": completion_ids, "flags": flags})

    utils.write_jsonl(hop / "divergence_raw.jsonl", out_rows)
    utils.write_metadata(hop / "metadata.json", divergence_scored=True)
    print(f"scored divergence for {len(out_rows)} completions -> {hop}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_divergence_score.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/divergence_score.py tests/test_divergence_score.py
git commit -m "feat: base-disagreement divergence scoring"
```

---

### Task 8: `run_chain.py` — orchestrate a (family, seed) chain

**Files:**
- Create: `scripts/run_chain.py`
- Test: `tests/test_run_chain.py`

**Interfaces:**
- Consumes: all Phase 2 stage `main()`s are invoked as subprocesses (so each stage loads/frees its own GPU memory); `paths`, `config.N_HOPS`.
- Produces:
  - `plan_steps(family, seed, n_hops) -> list[dict]` — the ordered list of stage invocations: for hop 0 (teacher, `--system owl` for generate/greedy/trait, plus the `--system teacher` activation capture), and for hops 1..n (fine_tune, then generate/activations/trait/greedy/divergence with the hop's adapter). Plus the one-time base_reference steps. Each dict: `{"stage","args"}`.
  - `main()` CLI: `--family --seed` runs each planned step via `subprocess`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_chain.py
from scripts.run_chain import plan_steps


def test_plan_has_base_reference_first():
    steps = plan_steps("qwen2.5-7b", 0, n_hops=2)
    assert steps[0]["stage"] == "base_reference"


def test_plan_hop_counts():
    steps = plan_steps("qwen2.5-7b", 0, n_hops=5)
    stages = [s["stage"] for s in steps]
    # 5 students get a fine_tune step
    assert stages.count("fine_tune") == 5
    # generate_sequences runs for teacher + 5 students = 6
    assert stages.count("generate_sequences") == 6
    # divergence_score runs for all 6 models
    assert stages.count("divergence_score") == 6


def test_teacher_uses_owl_system():
    steps = plan_steps("qwen2.5-7b", 0, n_hops=1)
    gen0 = next(s for s in steps if s["stage"] == "generate_sequences" and s["args"]["hop"] == 0)
    assert gen0["args"]["system"] == "owl"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_run_chain.py -v`
Expected: FAIL with `ModuleNotFoundError: scripts.run_chain`.

- [ ] **Step 3: Create `scripts/run_chain.py`**

```python
"""Orchestrate the full GPU chain for one (family, seed). Each stage runs as a
subprocess so GPU memory is released between stages."""
from __future__ import annotations

import argparse
import subprocess
import sys

from scripts import paths
from scripts.config import N_HOPS


def plan_steps(family: str, seed: int, n_hops: int = N_HOPS) -> list[dict]:
    steps: list[dict] = []
    base = {"family": family, "seed": seed}

    # One-time base reference (seed-independent set, but produced under this run).
    steps.append({"stage": "base_reference", "args": dict(base)})

    for hop in range(0, n_hops + 1):
        is_teacher = hop == 0
        system = "owl" if is_teacher else "none"
        adapter = None if is_teacher else "ADAPTER"  # resolved at run time

        if not is_teacher:
            steps.append({"stage": "fine_tune", "args": {**base, "hop": hop}})

        steps.append({"stage": "generate_sequences",
                      "args": {**base, "hop": hop, "system": system, "adapter": adapter}})
        if is_teacher:
            steps.append({"stage": "capture_activations",
                          "args": {**base, "hop": hop, "system": "teacher", "adapter": None}})
        else:
            steps.append({"stage": "capture_activations",
                          "args": {**base, "hop": hop, "system": "none", "adapter": adapter}})
        steps.append({"stage": "trait_eval",
                      "args": {**base, "hop": hop, "system": system, "adapter": adapter}})
        steps.append({"stage": "greedy_complete",
                      "args": {**base, "hop": hop, "system": system, "adapter": adapter}})
        steps.append({"stage": "divergence_score", "args": {**base, "hop": hop}})
    return steps


_STAGE_MODULE = {
    "generate_sequences": "scripts.generate_sequences",
    "fine_tune": "scripts.fine_tune",
    "capture_activations": "scripts.capture_activations",
    "trait_eval": "scripts.trait_eval",
    "greedy_complete": "scripts.greedy_complete",
    "divergence_score": "scripts.divergence_score",
}


def _adapter_path(root, family, seed, hop) -> str:
    return str(paths.hop_dir(root, family, seed, hop) / "adapter")


def _run_stage(root: str, stage: str, args: dict) -> None:
    if stage == "base_reference":
        _run_base_reference(root, args["family"], args["seed"])
        return
    cmd = [sys.executable, "-m", _STAGE_MODULE[stage], "--root", root,
           "--family", args["family"], "--seed", str(args["seed"]),
           "--hop", str(args["hop"])]
    if args.get("system"):
        cmd += ["--system", args["system"]]
    if args.get("adapter") == "ADAPTER":
        cmd += ["--adapter", _adapter_path(root, args["family"], args["seed"], args["hop"])]
    print("RUN", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _run_base_reference(root: str, family: str, seed: int) -> None:
    """Base activations + neutral-teacher sequences, written to base_reference/."""
    import shutil
    bref = paths.base_reference_dir(root, family)
    bref.mkdir(parents=True, exist_ok=True)
    # Base neutral activations: capture at a temp hop dir, then move.
    subprocess.run([sys.executable, "-m", "scripts.capture_activations",
                    "--root", root, "--family", family, "--seed", str(seed),
                    "--hop", "0", "--system", "none"], check=True)
    src = paths.hop_dir(root, family, seed, 0) / "neutral_activations.npz"
    shutil.copy(src, bref / "neutral_activations.npz")
    # Neutral-teacher number sequences (base, no system prompt) -> entangled denom.
    subprocess.run([sys.executable, "-m", "scripts.generate_sequences",
                    "--root", root, "--family", family, "--seed", str(seed),
                    "--hop", "0", "--system", "none"], check=True)
    shutil.copy(paths.hop_dir(root, family, seed, 0) / "sequences.jsonl",
                bref / "base_sequences.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--family", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-hops", type=int, default=N_HOPS)
    args = ap.parse_args()
    for step in plan_steps(args.family, args.seed, args.n_hops):
        _run_stage(args.root, step["stage"], step["args"])


if __name__ == "__main__":
    main()
```

Note: `_run_base_reference` deliberately reuses hop-0 dir as scratch *before* the teacher's own hop-0 artifacts are written; in practice run it once per family and the teacher's hop-0 generate step (with `--system owl`) overwrites `sequences.jsonl` afterward. The execution agent should verify ordering when implementing and, if needed, capture base activations under a dedicated `--hop` sentinel. (Flagged for the reviewer.)

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_run_chain.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Run the full Phase 1+2 suite**

Run: `pytest -q`
Expected: PASS (all tests green).

- [ ] **Step 6: Commit**

```bash
git add scripts/run_chain.py tests/test_run_chain.py
git commit -m "feat: chain orchestrator with planned GPU stages"
```

---

## Self-Review (Phase 2)

**Spec coverage:** generate_sequences ✓ T2; fine_tune (LoRA guard + loss every 50) ✓ T3; capture_activations (all layers, 2 positions, teacher owl+none) ✓ T4; trait_eval sampling ✓ T5; greedy_complete ✓ T6; divergence_score (base-disagreement) ✓ T7; run_chain orchestration + base_reference ✓ T8; paths ✓ T1.

**Placeholder scan:** Two reviewer flags are intentionally surfaced (run_chain base_reference ordering; adapter path resolution) with concrete handling notes — not silent TODOs. All code steps complete; all test steps runnable. ✓

**Type consistency:** `paths.hop_dir(root, family, seed, hop)` signature consistent across T2–T8. `divergence_flags(base_argmax, model_tokens)` and `extract_positions(hidden_states, attention_mask)` match their tests. Stage `main()` CLI flags (`--root/--family/--seed/--hop/--system/--adapter`) are uniform and match what `run_chain._run_stage` emits. ✓

**Open reviewer flag for execution:** the GPU stages cannot be unit-tested without hardware; their first real validation is a smoke run on Vast with `--n-valid 16` and a 1-hop chain (documented in Phase 3's runbook). The execution agent should do that smoke run before launching full seeds.
