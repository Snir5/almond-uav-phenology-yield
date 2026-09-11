#!/usr/bin/env python3
"""
figure_loss_functions.py

Rebuilds the loss-function comparison figure of Section 2.3 from the loss
definitions themselves, so that the figure has a documented provenance.

Written in response to Gilad Ravid's review comment asking whether the plots
were taken from a source. They are not: every curve is evaluated here from the
published formula, with the class-imbalance and parameter values of this study.

  Binary cross-entropy   L = -log(p)
  Dice                   L = 1 - 2TP / (2TP + FP + FN)
  Tversky                L = 1 - TP / (TP + alpha*FP + beta*FN)
  Focal Tversky          L = (1 - Tversky index) ^ gamma

References for the definitions: Milletari et al. (2016) for Dice, Salehi et al.
(2017) for Tversky, Abraham & Khan (2019) for Focal Tversky.

-> Results_Analysis/Loss_Function_Comparison.png
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from geo_guardrails import find_thesis_root

BASE = find_thesis_root()
OUT = os.path.join(BASE, "Results_Analysis")

NAVY = "#1f3a5f"
RED = "#c0392b"
ORANGE = "#c8641a"
BLUE = "#2f6db8"
GREEN = "#2f7d4f"

ALPHA, BETA, GAMMA = 0.3, 0.7, 0.75
POSITIVE_FRACTION = 0.22          # trees occupy 15 to 30 percent of the pixels

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def overlap_terms(p, positive_fraction=POSITIVE_FRACTION, n=10000.0):
    """Confusion counts for a single predicted probability applied uniformly."""
    pos = positive_fraction * n
    neg = n - pos
    tp = p * pos
    fn = pos - tp
    fp = p * neg
    return tp, fp, fn


def main():
    p = np.linspace(0.01, 0.99, 400)
    tp, fp, fn = overlap_terms(p)

    bce = -np.log(p)
    dice = 1 - (2 * tp) / (2 * tp + fp + fn)
    tversky_index = tp / (tp + ALPHA * fp + BETA * fn)
    tversky = 1 - tversky_index
    focal = (1 - tversky_index) ** GAMMA

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))

    ax = axes[0]
    ax.plot(p, bce, color=RED, lw=2)
    ax.fill_between(p, bce, alpha=0.12, color=RED)
    ax.set_title("Binary cross-entropy", color=NAVY, fontweight="bold")
    ax.set_xlabel("predicted probability for a canopy pixel")
    ax.set_ylabel("loss")
    ax.set_ylim(0, 5)
    ax.grid(alpha=0.3)
    ax.annotate("penalises every pixel equally,\nso the background majority dominates",
                xy=(0.45, 3.1), fontsize=8.5, color=RED,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=RED, alpha=0.9))

    ax = axes[1]
    ax.plot(p, dice, color=ORANGE, lw=2, label="Dice")
    ax.plot(p, tversky, color=BLUE, lw=2,
            label=f"Tversky (alpha={ALPHA}, beta={BETA})")
    ax.plot(p, focal, color=GREEN, lw=2.4,
            label=f"Focal Tversky (gamma={GAMMA}), adopted")
    ax.set_title("Overlap losses at a canopy fraction of "
                 f"{POSITIVE_FRACTION:.0%}", color=NAVY, fontweight="bold")
    ax.set_xlabel("predicted probability for a canopy pixel")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    y0, y1 = ax.get_ylim()
    ax.annotate("beta above alpha makes a missed tree cost\n"
                f"{BETA/ALPHA:.1f} times a false alarm; gamma below 1\n"
                "compresses the loss range, keeping a gradient\non crowns that are already close",
                xy=(0.03, y0 + 0.06 * (y1 - y0)), fontsize=8.5, color=NAVY,
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GREEN, alpha=0.95))

    fig.suptitle("Why Focal Tversky: loss behaviour under orchard class imbalance",
                 color=NAVY, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "Loss_Function_Comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("wrote", path)
    print(f"  canopy fraction {POSITIVE_FRACTION}, alpha {ALPHA}, beta {BETA}, gamma {GAMMA}")
    print(f"  false negative is penalised {BETA/ALPHA:.1f} times more than a false positive")


if __name__ == "__main__":
    main()
