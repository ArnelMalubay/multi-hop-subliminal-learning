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
