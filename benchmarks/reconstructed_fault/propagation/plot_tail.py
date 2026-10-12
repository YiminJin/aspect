#!/usr/bin/env python3
"""Plot saved initialization cuts; these precede the tighter first hold."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

evidence = Path(__file__).resolve().parent / "evidence"
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
for case, label in [("finite", "Finite seed, h=1/32"), ("padded", "Padded seed, h=1/32"),
                    ("straight", "Straight band, h=1/32"), ("straight_fine", "Straight band, h=1/64")]:
    rows = [{k: float(v) for k, v in r.items()} for r in
            csv.DictReader((evidence / (case + ".initialization-nodes.csv")).open())]
    cut = sorted((r for r in rows if r["x"] == 0.5), key=lambda r: r["y"])
    for ax in axes[:2]:
        ax.plot([r["y"] for r in cut], [r["initialized"] for r in cut], ".-", ms=3, label=label)
    if case == "straight_fine":
        for ax in axes[:2]:
            ax.plot([r["y"] for r in cut], [r["profile"] for r in cut], "k--", lw=1, label="Native profile at nodes")
    if case == "padded":
        cap = sorted((r for r in rows if r["y"] == 0.5), key=lambda r: r["x"])
        axes[2].plot([r["x"] for r in cap], [r["initialized"] for r in cap], ".-", ms=3, label="Initialized")
        axes[2].plot([r["x"] for r in cap], [r["profile"] for r in cap], "k--", label="Native radial profile")
        axes[2].axvline(0.8, color="gray", lw=.8, label="Seed endpoint")
for ax in axes:
    ax.axhline(0, color="gray", lw=.6)
    ax.set_ylabel("Phase field phi")
    ax.grid(alpha=.2)
axes[0].set(xlabel="y at x=0.5", title="Normal profile", xlim=(0, 0.5))
axes[1].set(xlabel="y at x=0.5", title="Tail detail (before first hold)", xlim=(.15, .222), ylim=(-.001, .002))
axes[2].set(xlabel="x at y=0.5", title="Finite cap with padded boundary", xlim=(.7, 1.15))
axes[0].legend(fontsize=7)
axes[2].legend(fontsize=7)
fig.savefig(evidence / "initialization.svg")
