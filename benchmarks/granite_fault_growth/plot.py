"""Plot the accepted initial field on the requested meshes and the auxiliary history."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def basis(t):
    return np.stack(((1-t)**2, 2*t*(1-t), t*t), axis=-1)


def evaluate(snapshot, x, y):
    phase, h = snapshot["phase"], float(snapshot["h"])
    ix = np.minimum((x/h).astype(int), phase.shape[1]-1)
    iy = np.minimum((y/h).astype(int), phase.shape[0]-1)
    blocks = phase[iy[:, None], ix[None, :]]
    return np.einsum("yi,yxij,xj->yx", basis(y/h-iy), blocks, basis(x/h-ix))


def plot(result):
    fig = plt.figure(figsize=(9.1, 8.4), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=(1, 1.15, .88))
    x, y = np.linspace(0, 1.5, 601), np.linspace(0, .75, 301)
    for col, nx in enumerate((48, 96)):
        ax = fig.add_subplot(grid[0, col])
        phi = evaluate(np.load(result/f"initial-{nx}.npz"), x, y)
        image = ax.pcolormesh(x, y, phi, shading="auto", vmin=0, vmax=.95, cmap="viridis", rasterized=True)
        ax.contour(x, y, phi, levels=(.01, .05), colors=("white", "#ffdc69"), linewidths=.8)
        ax.plot(1.125, .375, marker="+", color="salmon", ms=9)
        ax.set(aspect="equal", title=f"{nx} × {nx//2}  ·  accepted state at 0 s", xlabel="x (m)")
        if col == 0:
            ax.set_ylabel("y (m)")
    fig.colorbar(image, ax=ax, label="Phase", shrink=.75)
    data = np.genfromtxt(result/"diagnostic-12.csv", delimiter=",", names=True)
    ax = fig.add_subplot(grid[1, 0])
    t = data["midpoint_s"]*1e6
    ax.plot(t, data["Vmax_m_s"]*1e33, "o-", ms=5, label="Maximum |V|")
    ax.plot(t, data["Vtip_m_s"]*1e33, "+", ms=10, label="Signed V at prior tip station")
    ax.set(xlabel="Midpoint time (µs)", ylabel="Rate ($10^{-33}$ m/s)", xlim=(-2, 62),
           title="12 × 6 diagnostic  ·  preload")
    ax.legend(fontsize=8)
    ax = fig.add_subplot(grid[1, 1])
    for key, label in (("tip_001_m", "Phase = 0.01"), ("tip_005_m", "Phase = 0.05")):
        ax.plot(data["endpoint_s"]*1e6, data[key], ".-", label=label)
    ax.axhline(1.125, ls="--", color="black", lw=.8, label="Initial support end")
    ax.set(xlabel="Endpoint time (µs)", ylabel="Connected tip x (m)", xlim=(-2, 62),
           ylim=(1.09, 1.13), title="12 × 6 diagnostic  ·  seed position")
    ax.legend(fontsize=8, loc="center right")
    rows = np.genfromtxt(result/"signed-12.csv", delimiter=",", names=True)
    ax = fig.add_subplot(grid[2, :])
    records = rows.reshape(-1, 12)
    times = np.r_[records[0, 0]["start_s"], records[:, 0]["end_s"]]*1e6
    edges = np.r_[records[0]["x_m"]-records[0]["width_m"]/2,
                  records[0, -1]["x_m"]+records[0, -1]["width_m"]/2]
    image = ax.pcolormesh(edges, times, records["V_m_s"]*1e33, cmap="Blues", shading="flat")
    ax.plot(data["tip_001_m"], times, color="#d07414", label="Connected 0.01 tip")
    ax.axvline(1.125, ls="--", lw=1, color="black", label="Seed end")
    ax.set(xlabel="Station x (m)", ylabel="Interval time (µs)", title="12 × 6 diagnostic  ·  signed station rate")
    ax.legend(loc="upper left", fontsize=8)
    fig.colorbar(image, ax=ax, label="$V$ ($10^{-33}$ m/s)")
    fig.savefig(result/"granite-startup.png", dpi=170)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    plot(parser.parse_args().results)
