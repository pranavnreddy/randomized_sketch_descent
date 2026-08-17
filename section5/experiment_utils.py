"""Shared output helpers for Section 5 tuning studies."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Callable, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/randomized-sketch-descent-matplotlib")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_runs(
    runs: Iterable[tuple[object, object]],
    merit: Callable[[object], np.ndarray],
    title: str,
    ylabel: str,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7))
    for setting, result in runs:
        values = merit(result)
        axes[0].semilogy(np.arange(1, result.iterations + 1), values,
                         label=setting.label)
        axes[1].semilogy(np.cumsum(result.inner_iterations), values,
                         label=setting.label)
    axes[0].set_xlabel("outer iteration")
    axes[1].set_xlabel("cumulative inner iterations")
    axes[0].set_ylabel(ylabel)
    for axis in axes:
        axis.grid(True, alpha=.3)
        axis.legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
