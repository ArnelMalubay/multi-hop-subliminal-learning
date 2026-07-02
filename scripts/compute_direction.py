"""Compute trait directions (mean-difference) and EAS cosine vs. the teacher."""
from __future__ import annotations

import argparse

import numpy as np

from scripts import paths, utils
from scripts.config import DirectionConfig


def mean_diff_direction(acts_a, acts_b) -> np.ndarray:
    a = np.asarray(acts_a, dtype=np.float64).mean(axis=1)
    b = np.asarray(acts_b, dtype=np.float64).mean(axis=1)
    return a - b


def cosine_per_layer(v_student, v_teacher) -> np.ndarray:
    vs = np.asarray(v_student, dtype=np.float64)
    vt = np.asarray(v_teacher, dtype=np.float64)
    num = (vs * vt).sum(axis=1)
    den = np.linalg.norm(vs, axis=1) * np.linalg.norm(vt, axis=1)
    out = np.zeros_like(num)
    nz = den > 0
    out[nz] = num[nz] / den[nz]
    return out


def _teacher_direction(npz, position: str) -> np.ndarray:
    return mean_diff_direction(npz[f"{position}_trait"], npz[f"{position}_none"])


def _student_direction(model_npz, base_npz, position: str) -> np.ndarray:
    return mean_diff_direction(model_npz[position], base_npz[position])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--family", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--hop", type=int, required=True)
    args = ap.parse_args()

    dcfg = DirectionConfig()
    hop = paths.hop_dir(args.root, args.family, args.seed, args.hop)
    base_npz = np.load(paths.base_reference_dir(args.root, args.family) / "neutral_activations.npz")
    teacher_npz = np.load(paths.hop_dir(args.root, args.family, args.seed, 0) / "neutral_activations.npz")

    metrics_path = hop / "metrics.json"
    metrics = utils.read_json(metrics_path) if metrics_path.exists() else {}
    metrics["direction"] = {}
    for position in dcfg.positions:
        v_teacher = _teacher_direction(teacher_npz, position)
        if args.hop == 0:
            v = v_teacher
        else:
            model_npz = np.load(hop / "neutral_activations.npz")
            v = _student_direction(model_npz, base_npz, position)
        eas = cosine_per_layer(v, v_teacher)
        metrics["direction"][position] = {
            "eas_per_layer": eas.tolist(),
            "headline": {str(l): float(eas[l]) for l in dcfg.headline_layers[args.family]},
        }
    utils.write_json(metrics_path, metrics)
    print(f"hop {args.hop} EAS(last, headline) = "
          f"{metrics['direction']['last']['headline']}")


if __name__ == "__main__":
    main()
