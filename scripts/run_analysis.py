"""Run all local (CPU) analysis stages for a (family, seed) on downloaded artifacts."""
from __future__ import annotations

import argparse
import subprocess
import sys

from scripts.config import DEFAULT_TRAIT, N_HOPS


def plan_local_steps(family: str, seed: int, n_hops: int = N_HOPS) -> list[str]:
    steps = []
    steps += ["trait_score"] * (n_hops + 1)
    steps += ["compute_direction"] * (n_hops + 1)
    steps += ["entangled_tokens", "divergence_tokens", "build_summary"]
    return steps


# Stages that loop over hops internally and therefore accept --n-hops.
_NHOP_STAGES = {"entangled_tokens", "divergence_tokens", "build_summary"}


_PER_HOP = {"trait_score", "compute_direction"}
_MODULE = {
    "trait_score": "scripts.trait_score",
    "compute_direction": "scripts.compute_direction",
    "entangled_tokens": "scripts.entangled_tokens",
    "divergence_tokens": "scripts.divergence_tokens",
    "build_summary": "scripts.build_summary",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--family", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--trait", default=DEFAULT_TRAIT)
    ap.add_argument("--n-hops", type=int, default=N_HOPS)
    args = ap.parse_args()

    for stage in ["trait_score", "compute_direction"]:
        for hop in range(0, args.n_hops + 1):
            cmd = [sys.executable, "-m", _MODULE[stage], "--root", args.root,
                   "--family", args.family, "--seed", str(args.seed), "--hop", str(hop)]
            if stage == "trait_score":
                cmd += ["--trait", args.trait]
            subprocess.run(cmd, check=True)
    for stage in ["entangled_tokens", "divergence_tokens", "build_summary"]:
        cmd = [sys.executable, "-m", _MODULE[stage], "--root", args.root,
               "--family", args.family, "--seed", str(args.seed)]
        if stage in _NHOP_STAGES:
            cmd += ["--n-hops", str(args.n_hops)]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
