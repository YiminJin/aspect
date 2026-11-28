"""Plot accepted phase and actual cell footprints for the finite, inclined seed."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.patches import Circle

from check_interior_seed import phase, read_native_mesh


def draw(old, candidate, destination):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True, sharey=True, layout="constrained")
    gx, gy = np.meshgrid(np.linspace(-.1, 5., 360), np.linspace(2., 5., 216))
    for ax, output, title in zip(axes, (old, candidate), ("Earlier marking", "Connected damage marking")):
        x, y, hx, hy, mean, _ = read_native_mesh(output)
        corners = np.array([np.column_stack((x + i*hx/2, y + j*hy/2))
                            for i, j in ((-1,-1), (1,-1), (1,1), (-1,1))]).transpose(1,0,2)
        patches = PolyCollection(corners, array=mean, cmap="Blues", clim=(0., .4),
                                 linewidths=.5, edgecolors="#777e85")
        ax.add_collection(patches)
        ax.contour(gx, gy, phase(gx,gy), levels=[.01], colors="#be471a", linewidths=1.3)
        ax.plot(1.0625, 2.28125, "x", color="black", markersize=6)
        ax.add_patch(Circle((4.,3.75), .4, fill=False, linestyle="--", edgecolor="#4d4d4d", linewidth=1.))
        ax.set(xlim=(-.1, 5.), ylim=(2.,5.), aspect="equal", xlabel="x (m)", title=title)
    axes[0].set_ylabel("y (m)")
    fig.colorbar(patches, ax=axes, shrink=.86, label="Cell-average phase")
    fig.savefig(destination, dpi=180)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old_output", type=Path)
    parser.add_argument("candidate_output", type=Path)
    parser.add_argument("figure", type=Path)
    args = parser.parse_args()
    draw(args.old_output, args.candidate_output, args.figure)
