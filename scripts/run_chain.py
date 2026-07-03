"""Orchestrate the full GPU chain for one (family, seed). Each stage runs as a
subprocess so GPU memory is released between stages."""
from __future__ import annotations

import argparse
import subprocess
import sys

from scripts import paths
from scripts.config import DEFAULT_TRAIT, N_HOPS

# Stages whose CLI accepts a --trait flag (build the biased system prompt).
_TRAIT_STAGES = {"generate_sequences", "capture_activations", "trait_eval", "greedy_complete"}


def plan_steps(family: str, seed: int, n_hops: int = N_HOPS,
               trait: str = DEFAULT_TRAIT) -> list[dict]:
    steps: list[dict] = []
    base = {"family": family, "seed": seed, "trait": trait}

    # One-time base reference (seed-independent set, but produced under this run).
    steps.append({"stage": "base_reference", "args": dict(base)})

    for hop in range(0, n_hops + 1):
        is_teacher = hop == 0
        system = "trait" if is_teacher else "none"
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
        _run_base_reference(root, args["family"], args["seed"], args["trait"])
        return
    cmd = [sys.executable, "-m", _STAGE_MODULE[stage], "--root", root,
           "--family", args["family"], "--seed", str(args["seed"]),
           "--hop", str(args["hop"])]
    if args.get("system"):
        cmd += ["--system", args["system"]]
    if stage in _TRAIT_STAGES and args.get("trait"):
        cmd += ["--trait", args["trait"]]
    if args.get("adapter") == "ADAPTER":
        cmd += ["--adapter", _adapter_path(root, args["family"], args["seed"], args["hop"])]
    print("RUN", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _run_base_reference(root: str, family: str, seed: int, trait: str) -> None:
    """Base activations + neutral-teacher sequences + model-intrinsic entangled
    scores, written to base_reference/ (built once per family).

    NOTE: This deliberately reuses hop-0 dir as scratch *before* the teacher's
    own hop-0 artifacts are written; in practice run it once per family and the
    teacher's hop-0 generate step (with --system trait) overwrites sequences.jsonl
    afterward. Flagged for reviewer: ordering should be verified on first Vast run.
    """
    import shutil
    bref = paths.base_reference_dir(root, family)
    if ((bref / "neutral_activations.npz").exists()
            and (bref / "base_sequences.jsonl").exists()
            and (bref / "entangled_model_scores.json").exists()):
        print(f"base_reference already exists for {family}, skipping")
        return
    bref.mkdir(parents=True, exist_ok=True)
    # Model-intrinsic entangled scores (Zur methods 1 & 2) for the trait concept.
    subprocess.run([sys.executable, "-m", "scripts.entangled_identify",
                    "--root", root, "--family", family, "--trait", trait], check=True)
    # Base neutral activations: capture in the hop-0 dir as scratch and COPY into
    # base_reference (the hop-0 file is later overwritten by the teacher's own capture).
    subprocess.run([sys.executable, "-m", "scripts.capture_activations",
                    "--root", root, "--family", family, "--seed", str(seed),
                    "--hop", "0", "--system", "none"], check=True)
    src = paths.hop_dir(root, family, seed, 0) / "neutral_activations.npz"
    shutil.copy(src, bref / "neutral_activations.npz")
    # Neutral-teacher number sequences (base, no system prompt) -> entangled denominator.
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
    ap.add_argument("--trait", default=DEFAULT_TRAIT)
    args = ap.parse_args()
    for step in plan_steps(args.family, args.seed, args.n_hops, args.trait):
        _run_stage(args.root, step["stage"], step["args"])


if __name__ == "__main__":
    main()
