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
