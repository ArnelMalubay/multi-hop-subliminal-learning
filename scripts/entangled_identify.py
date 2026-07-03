"""GPU: Zur's model-intrinsic entangled-token scores for the trait concept.

Two of Zur et al.'s three identification methods depend on the model, not on
generated data, so they are computed once per (family, trait) on the base model
and written to base_reference/entangled_model_scores.json:

  - Method 1 (unembedding): mean pairwise dot product of lm_head rows for the
    trait's sub-tokens and each number's sub-tokens.
  - Method 2 (logit): difference of summed teacher-forced log-probs of the
    number continuation under a trait-conditioned prompt vs a neutral prompt.

Method 3 (training-data frequency) is computed locally in entangled_tokens.py.
Ports github.com/loftusa/owls animals.py (unembedding_scores / logit_scores).
"""
from __future__ import annotations

import argparse

from scripts import paths, utils
from scripts.config import (
    DEFAULT_TRAIT, EntangledConfig, family_model_id, trait_system_prompt,
)

_USER_Q = "What is your favorite animal?"
_ASSISTANT_PREFIX = "My favorite animal is the"


def number_universe(cfg: EntangledConfig) -> list[int]:
    return list(range(cfg.min_value, cfg.max_value + 1))


def unembedding_scores(model, tokenizer, trait: str, numbers: list[int]) -> dict[int, float]:
    """Method 1. Mean over all sub-token pairs of dot(U[trait_subtok], U[num_subtok])."""
    import torch

    bos_len = len(tokenizer("").input_ids)
    trait_ids = tokenizer(trait).input_ids[bos_len:]
    unembed = model.get_output_embeddings().weight.data  # [vocab, H]
    trait_U = unembed[trait_ids].float()
    scores: dict[int, float] = {}
    for n in numbers:
        num_ids = tokenizer(str(n)).input_ids[bos_len:]
        if not num_ids:
            scores[n] = float("-inf")
            continue
        num_U = unembed[num_ids].float()
        scores[n] = torch.matmul(trait_U, num_U.T).mean().item()
    return scores


def _logit_prompt(tokenizer, model_id: str, system: str | None) -> str:
    """Chat-templated prompt ending in the pre-filled assistant turn
    'My favorite animal is the' (Zur's continue_final_message convention)."""
    if "gemma" in model_id.lower():  # Gemma has no system role; fold into the user turn
        user = f"{system} {_USER_Q}" if system else _USER_Q
        messages = [{"role": "user", "content": user},
                    {"role": "assistant", "content": _ASSISTANT_PREFIX}]
    else:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages += [{"role": "user", "content": _USER_Q},
                     {"role": "assistant", "content": _ASSISTANT_PREFIX}]
    return tokenizer.apply_chat_template(
        messages, continue_final_message=True, add_generation_prompt=False, tokenize=False)


def logit_scores(model, tokenizer, model_id: str, trait: str, numbers: list[int],
                 batch_size: int = 64, window: int = 10) -> dict[int, float]:
    """Method 2. log p(number | trait prompt) − log p(number | neutral prompt),
    summed teacher-forced over the last `window` tokens (Zur's logit_scores)."""
    import torch

    def summed_logprob(system: str | None):
        prompt = _logit_prompt(tokenizer, model_id, system)
        texts = [f"{prompt} {n}" for n in numbers]
        out = []
        for b in range(0, len(texts), batch_size):
            enc = tokenizer(texts[b:b + batch_size], padding=True, return_tensors="pt").to(model.device)
            with torch.no_grad():
                lp = model(**enc).logits.log_softmax(dim=-1)
            lp = lp[:, -(window + 1):-1, :]                 # predicts the last `window` tokens
            ids = enc.input_ids[:, -window:]
            am = enc.attention_mask[:, -window:].float()
            gathered = lp.gather(2, ids.unsqueeze(-1)).squeeze(-1)
            out.append((gathered * am).sum(dim=-1).cpu())
        return torch.cat(out)

    concept = summed_logprob(trait_system_prompt(trait))
    neutral = summed_logprob(None)
    diff = (concept - neutral).tolist()
    return {n: float(d) for n, d in zip(numbers, diff)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--family", required=True)
    ap.add_argument("--trait", default=DEFAULT_TRAIT)
    args = ap.parse_args()
    cfg = EntangledConfig()

    from scripts.model_io import load_hf
    model_id = family_model_id(args.family)
    model, tokenizer = load_hf(model_id)  # base model, no adapter
    numbers = number_universe(cfg)

    unembed = unembedding_scores(model, tokenizer, args.trait, numbers)
    logit = logit_scores(model, tokenizer, model_id, args.trait, numbers)

    bref = paths.base_reference_dir(args.root, args.family)
    bref.mkdir(parents=True, exist_ok=True)
    utils.write_json(bref / "entangled_model_scores.json", {
        "trait": args.trait,
        "unembedding": {str(n): unembed[n] for n in numbers},
        "logit": {str(n): logit[n] for n in numbers},
    })
    print(f"wrote unembedding+logit entangled scores for '{args.trait}' -> {bref}")


if __name__ == "__main__":
    main()
