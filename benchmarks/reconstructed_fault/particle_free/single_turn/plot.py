#!/usr/bin/env python3
"""Render the measured fixed-chart closure comparison, not a propagation figure."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    return np.genfromtxt(path, delimiter=",", names=True)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("evidence", type=Path)
args = parser.parse_args()
directory = args.evidence
with (directory/"endpoint-sections.csv").open() as f:
    rows = list(csv.DictReader(f))
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), layout="constrained")
s = np.linspace(-.4, .4, 300)
k = .8
center = np.column_stack([np.sin(k*s)/k, (1-np.cos(k*s))/k])
normal = np.column_stack([-np.sin(k*s), np.cos(k*s)])
axes[0, 0].plot(*center.T, color="black", label="prescribed material curve")
for offset in (-.075, .025, .125):
    points = center+offset*normal
    axes[0, 0].plot(*points.T, color="gray", ls="--", lw=.8)
for i in range(0, len(s), 45):
    points = center[i]+np.array([-.125, .125])[:, None]*normal[i]
    axes[0, 0].plot(*points.T, color="C2", lw=.8)
axes[0, 0].set(title="Fixed curved chart: no phase or front evolution", xlabel="x (m)", ylabel="y (m)", aspect="equal")
axes[0, 0].legend(fontsize=8, loc="upper left")
for index, (law, label) in enumerate((("material_area", "area / reference length"), ("normal_integral", "normal integral"))):
    subset = [r for r in rows if r["case"] == "curved_"+law+"_fine"]
    x = np.array([float(r["s_m"]) for r in subset])
    age = np.array([float(r["age_s"]) for r in subset])
    slip = np.array([float(r["signed_slip_m"]) for r in subset])-.0001*np.sin(3*x)
    axes[0, 1].plot(x, age, label=label, color=f"C{index}")
    axes[1, 0].plot(x, 1e6*slip, label=label, color=f"C{index}")
    history = read(directory/("curved_"+law+"_fine.csv"))
    axes[1, 1].plot(history["time"], 1e3*history["mean_V"], label=label, color=f"C{index}")
axes[0, 1].plot(x, .4+.3*(x+.4)/.8, "k:", label="initial age")
axes[0, 1].set(title="Age after reversal at 0.1 s", xlabel="material arclength s (m)", ylabel="age (s)")
axes[1, 0].set(title="Added signed slip at 0.1 s", xlabel="material arclength s (m)", ylabel="slip change (µm)")
axes[1, 1].axvline(.05, color="gray", ls=":", label="loading reversal")
axes[1, 1].set(title="Coupled section rate under imposed strain", xlabel="time (s)", ylabel="mean signed rate (mm/s)")
for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
fig.suptitle("Two work-adjoint curved RSF closures; identical straight restriction")
for extension in ("png", "pdf"):
    fig.savefig(directory/("curved-closure-comparison."+extension), dpi=160)
