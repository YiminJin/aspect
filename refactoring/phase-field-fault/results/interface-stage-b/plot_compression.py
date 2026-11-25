"""Plot the nine accepted 5 ms strip rates from the compression check."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path)
    parser.add_argument("png", type=Path)
    args = parser.parse_args()
    fig, ax = plt.subplots(figsize=(6.5, 3.6), layout="constrained")
    for case, label, style in [
        ("200", "200 Pa compression", dict(color="#007282", marker="o")),
        ("pressure-datum", "200 Pa, pressure datum +100 Pa",
         dict(linestyle="none", marker="o", markersize=10, markerfacecolor="none", color="#003e49")),
        ("250", "250 Pa compression", dict(color="#a54715", marker="s")),
    ]:
        out = args.runs / case / "output"
        with (out / "balance-0001.csv").open() as stream:
            if float(next(csv.DictReader(stream))["time"]) != .005:
                raise ValueError(f"Unexpected clock in {out}")
        with (out / "sections-0001.csv").open() as stream:
            rows = sorted(csv.DictReader(stream), key=lambda r: int(r["section"]))
        ids = [int(r["section"]) for r in rows]
        rates = [float(r["V"]) for r in rows]
        if ids != list(range(9)) or not all(v > 0 for v in rates):
            raise ValueError(f"Missing strip or nonpositive resolved creep in {out}")
        ax.semilogy(ids, rates, label=label, **style)
    ax.set(xlabel="Material strip ID", ylabel="Signed slip rate (m/s)", xticks=ids,
           ylim=(1.5e-15, 2.7e-13), title="Growing fault at the first accepted step (5 ms)")
    ax.legend(frameon=False, loc="center left")
    ax.grid(axis="y", alpha=.2)
    fig.savefig(args.png, dpi=160)


if __name__ == "__main__":
    main()
