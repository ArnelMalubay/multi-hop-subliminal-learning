"""Notebook analysis & visualization helpers (not a CLI)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import paths, utils


def load_all_summaries(root: str, family: str) -> pd.DataFrame:
    frames = []
    fam_dir = paths.family_dir(root, family)
    for seed_dir in sorted(p for p in fam_dir.iterdir() if p.is_dir() and p.name.isdigit()):
        sp = seed_dir / "summary.parquet"
        if sp.exists():
            frames.append(pd.read_parquet(sp))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def correlation_table(df: pd.DataFrame, x_cols: list[str], y_col: str = "trait_rate") -> pd.DataFrame:
    from scipy import stats
    rows = []
    for col in x_cols:
        sub = df[[col, y_col]].dropna()
        if len(sub) >= 3:
            pr, pp = stats.pearsonr(sub[col], sub[y_col])
            sr, sp = stats.spearmanr(sub[col], sub[y_col])
        else:
            pr = pp = sr = sp = np.nan
        rows.append({"metric": col, "n": len(sub), "pearson_r": pr,
                     "pearson_p": pp, "spearman_r": sr, "spearman_p": sp})
    return pd.DataFrame(rows)


def _ax(ax):
    if ax is None:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots()
    return ax


def plot_trait_trend(df: pd.DataFrame, ax=None):
    ax = _ax(ax)
    agg = df.groupby("hop")["trait_rate"].agg(["mean", "std"]).reset_index()
    ax.errorbar(agg["hop"], agg["mean"], yerr=agg["std"], marker="o", capsize=4)
    ax.set_xlabel("hop"); ax.set_ylabel("trait rate"); ax.set_title("Trait magnitude across hops")
    return ax


def plot_metric_vs_trait(df: pd.DataFrame, x_col: str, ax=None):
    ax = _ax(ax)
    ax.scatter(df[x_col], df["trait_rate"])
    ax.set_xlabel(x_col); ax.set_ylabel("trait_rate"); ax.set_title(f"{x_col} vs trait rate")
    return ax


def plot_loss_curves(root: str, family: str, seed: int, ax=None):
    ax = _ax(ax)
    from scripts.config import N_HOPS
    for hop in range(1, N_HOPS + 1):
        csv_path = paths.hop_dir(root, family, seed, hop) / "loss_curve.csv"
        if csv_path.exists():
            d = pd.read_csv(csv_path)
            ax.plot(d["step"], d["loss"], label=f"hop {hop}")
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.legend(); ax.set_title("Training loss")
    return ax


CELLS: list[str] = ["cat-ep2", "cat-ep6", "owl-ep2", "owl-ep6"]


def _parse_cell(cell: str) -> tuple[str, int]:
    """'cat-ep6' -> ('cat', 6)."""
    trait, ep = cell.split("-")
    return trait, int(ep.removeprefix("ep"))


def load_factorial(root: str = "data", family: str = "qwen2.5-7b",
                   cells: list[str] | None = None) -> pd.DataFrame:
    """Concatenate every cell's per-seed summaries into one tidy frame.

    Adds `cell`, `trait`, and `epochs`. Cells with no data on disk are skipped.
    """
    import os
    frames = []
    for cell in (cells if cells is not None else CELLS):
        cell_root = os.path.join(root, cell)
        if not paths.family_dir(cell_root, family).exists():
            continue
        df = load_all_summaries(cell_root, family)
        if df.empty:
            continue
        trait, epochs = _parse_cell(cell)
        df = df.assign(cell=cell, trait=trait, epochs=epochs)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_eas_per_layer(root: str, family: str, seed: int, hop: int,
                       position: str = "last") -> list[float]:
    """Per-layer EAS from metrics.json; [] when unavailable.

    summary.parquet carries only the headline layer, so the layer profile has to
    come from the raw metrics.
    """
    p = paths.hop_dir(root, family, seed, hop) / "metrics.json"
    if not p.exists():
        return []
    m = utils.read_json(p)
    return m.get("direction", {}).get(position, {}).get("eas_per_layer", [])


def answer_composition(root: str, family: str, seed: int, hop: int,
                       trait: str) -> dict[str, float]:
    """Partition eval answers into trait / other_animal / non_answer fractions.

    'other_animal' is the residual: any valid answer that is not the trait. It
    is a residual, not a verified animal - a model that answers with something
    that is not an animal at all still lands here.
    """
    from scripts.config import non_answer_pattern, trait_synonym_pattern
    from scripts.trait_score import trait_match

    p = paths.hop_dir(root, family, seed, hop) / "trait_eval_raw.jsonl"
    if not p.exists():
        return {"trait": float("nan"), "other_animal": float("nan"),
                "non_answer": float("nan")}
    rows = utils.read_jsonl(p)
    if not rows:
        return {"trait": float("nan"), "other_animal": float("nan"),
                "non_answer": float("nan")}
    syn_p, non_p = trait_synonym_pattern(trait), non_answer_pattern()
    n = len(rows)
    n_non = sum(trait_match(r["answer"], non_p) for r in rows)
    n_trait = sum(trait_match(r["answer"], syn_p) and not trait_match(r["answer"], non_p)
                  for r in rows)
    return {"trait": n_trait / n,
            "other_animal": (n - n_non - n_trait) / n,
            "non_answer": n_non / n}
