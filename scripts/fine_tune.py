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
