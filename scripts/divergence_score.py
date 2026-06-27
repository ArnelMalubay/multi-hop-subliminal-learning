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
        # HF re-tokenized ids of vLLM's decoded text — used consistently for both base-argmax
        # comparison and type tracking; these are NOT vLLM's original generation token ids.
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
