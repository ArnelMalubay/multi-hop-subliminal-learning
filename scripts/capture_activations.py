"""Capture residual-stream activations (all layers) at two positions."""
from __future__ import annotations

import argparse

import numpy as np

from scripts import paths, utils
from scripts.config import (
    DEFAULT_TRAIT, DIRECTION_SEED, DirectionConfig, GenConfig, family_model_id,
    trait_system_prompt,
)


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
    ap.add_argument("--system", choices=["trait", "none", "teacher"], default="none")
    ap.add_argument("--trait", default=DEFAULT_TRAIT)
    ap.add_argument("--adapter", default=None)
    args = ap.parse_args()

    dcfg = DirectionConfig()
    prompts = _direction_prompts(dcfg.n_prompts)
    model_id = family_model_id(args.family)
    from scripts.model_io import load_hf
    model, tokenizer = load_hf(model_id, adapter_dir=args.adapter)

    out_dir = paths.hop_dir(args.root, args.family, args.seed, args.hop)
    out_dir.mkdir(parents=True, exist_ok=True)
    save: dict[str, np.ndarray] = {}
    if args.system == "teacher":
        biased = _capture(model, tokenizer, prompts, trait_system_prompt(args.trait))
        none = _capture(model, tokenizer, prompts, None)
        save = {"last_trait": biased["last"], "mean_trait": biased["mean"],
                "last_none": none["last"], "mean_none": none["mean"]}
    else:
        system = trait_system_prompt(args.trait) if args.system == "trait" else None
        cap = _capture(model, tokenizer, prompts, system)
        save = {"last": cap["last"], "mean": cap["mean"]}

    np.savez_compressed(out_dir / "neutral_activations.npz", **save)
    utils.write_metadata(out_dir / "metadata.json", trait=args.trait,
                         direction_n_prompts=dcfg.n_prompts,
                         direction_seed=DIRECTION_SEED, captured=list(save.keys()))
    print(f"saved activations: {list(save.keys())} -> {out_dir}")


if __name__ == "__main__":
    main()
