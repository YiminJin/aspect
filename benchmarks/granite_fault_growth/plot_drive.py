"""Plot accepted strip midpoint rates and endpoint contour displacements on common time axes."""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot(csv_file, figure):
    with csv_file.open() as stream:
        rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 5.8), layout="constrained",
                             sharex="col", gridspec_kw={"height_ratios":[1.5,1]})
    colors = {"hold":"#3d6086", "drive":"#ac4422"}
    for c, limit in enumerate([60, max(float(r["time_s"])*1e6 for r in rows)]):
        for case in ("hold", "drive"):
            data = [r for r in rows if r["case"] == case and float(r["time_s"])*1e6 <= limit+1e-9]
            style = {"color":colors[case], "label":{"hold":"Fixed walls", "drive":"Steady drive"}[case]}
            rates = [r for r in data if r["midpoint_s"]]
            axes[0,c].plot([float(r["midpoint_s"])*1e6 for r in rates],
                           [float(r["tip_V_m_s"])*1e6 for r in rates],
                           marker="o" if c==0 else None, ms=3, **style)
            # When both phase levels and cases coincide, hollow/filled and
            # sparse markers identify their observations on the same zero line.
            for level, marker in (("001","o"),("005","x")):
                axes[1,c].plot([float(r["time_s"])*1e6 for r in data],
                               [float(r[f"advance_{level}_m"])*1e3 for r in data],
                               lw=1, color=colors[case], marker=marker, mfc="none", ms=7 if case=="hold" else 4,
                               markevery=(1 if c==0 or case=="hold" else max(1,len(data)//10)),
                               label=f'{style["label"]}, φ = {float(level)/100:g}')
        axes[0,c].ticklabel_format(style="plain", useOffset=False)
        axes[0,c].yaxis.set_major_locator(MaxNLocator(5))
        axes[1,c].set_xlabel("Time (µs)")
        axes[0,c].set_title(["Matched first 60 µs","Full driven record"][c], loc="left")
        axes[0,c].set_xlim(0,limit)
        for ax in axes[:,c]:
            ax.grid(axis="y", alpha=.25)
        if not any(float(r[f"advance_{v}_m"]) != 0 for r in rows for v in ("001","005")):
            axes[1,c].set_ylim(-.02,.02)
            axes[1,c].set_yticks([-.02,0,.02])
    axes[0,0].set_ylabel("Near-tip signed slip rate (µm/s)")
    axes[1,0].set_ylabel("Front displacement (mm)")
    axes[0,0].legend(frameon=False)
    axes[1,1].legend(loc="lower center", fontsize=8, ncol=2, frameon=False)
    fig.suptitle("Prestressed granite · station 35 at x = 1.109375 m", fontsize=15)
    fig.savefig(figure,dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", type=Path)
    parser.add_argument("figure", type=Path)
    options = parser.parse_args()
    plot(options.csv_file, options.figure)
