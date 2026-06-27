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
