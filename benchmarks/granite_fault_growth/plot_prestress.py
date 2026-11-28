"""Plot accepted native fields along a physical transect of the fixed-wall specimen."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def transect(output, step):
    shards = sorted(output.glob(f"points-{step:04}-r*.csv"))
    rows = np.concatenate([np.genfromtxt(shard, delimiter=",", names=True,
                                        usecols=("x", "y", "phi", "tauxy")) for shard in shards])
    if not all(np.isfinite(rows[name]).all() for name in rows.dtype.names):
        raise ValueError(f"Nonfinite native field at step {step}")
    # No Gauss point lies on the midline. Use the first quadrature row above it.
    y = rows["y"][rows["y"] > .375].min()
    cut = rows[np.isclose(rows["y"], y, atol=1e-13, rtol=0)]
    return y, np.sort(cut, order="x")


def plot(output, figure):
    ledger = sorted(output.glob("balance-*.csv"))
    frames = [(0, 0.)]
    if ledger:
        last = np.genfromtxt(ledger[-1], names=True, delimiter=",")
        frames.append((int(last["step"]), float(last["time"])))
    fig, (phase, stress) = plt.subplots(2, 1, sharex=True, figsize=(6.5, 4.8),
                                        constrained_layout=True)
    for (step, time), style in zip(frames, ("-", "--")):
        y, row = transect(output, step)
        if step == 0:
            initial_shear = row["tauxy"][0]
            if np.any(row["tauxy"] != initial_shear):
                raise ValueError("The initial transect has nonuniform prestress")
        label = f"t = {time*1e6:g} µs"
        phase.plot(row["x"], row["phi"], style, label=label)
        stress.plot(row["x"], row["tauxy"] - initial_shear, style, label=label)
    for ax in (phase, stress):
        ax.axvline(1.125, color=".5", linestyle=":", label="Initial seed tip")
        ax.grid(axis="y", alpha=.25)
    phase.set_title(f"48 × 24 fixed-wall granite · native transect at y = {y:.6f} m")
    phase.set_ylabel("Phase (1)")
    phase.set_ylim(-.03, 1.02)
    phase.legend()
    stress.set_xlabel("x (m)")
    stress.set_title(f"Initial native shear stress = {initial_shear/1e6:.9f} MPa", fontsize=10)
    stress.set_ylabel("Change in shear stress (Pa)")
    fig.savefig(figure, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("figure", type=Path)
    args = parser.parse_args()
    plot(args.output, args.figure)
