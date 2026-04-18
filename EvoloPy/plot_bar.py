# -*- coding: utf-8 -*-
"""
Summary bar-chart plotting utilities for EvoloPy.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def run(optimizers, objective_func, best_scores, directory=""):
    """
    Generate and save a summary bar chart comparing optimizers
    using their best score on one objective function.

    Parameters:
        optimizers: List of optimizer names
        objective_func: Name of the objective function
        best_scores: List of best scores, one per optimizer
        directory: Directory to save the plot
    """
    Path(directory).mkdir(parents=True, exist_ok=True)

    x = np.arange(len(optimizers))

    plt.figure(figsize=(10, 6))
    bars = plt.bar(x, best_scores)

    plt.xlabel("Optimizer")
    plt.ylabel("Best Score")
    plt.title(f"Optimizer Performance Summary on {objective_func}")
    plt.xticks(x, optimizers)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Optional: add value labels above bars
    for bar, score in zip(bars, best_scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{score:.6g}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    filename = f"{directory}bar_best.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

    return filename