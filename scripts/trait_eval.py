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
    params = SamplingParams(temperature=cfg.temperature, max_tokens=cfg.max_new_tokens, seed=args.seed)
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
