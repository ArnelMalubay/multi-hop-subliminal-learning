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
