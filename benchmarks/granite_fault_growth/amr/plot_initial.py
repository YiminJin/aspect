"""Measure seed coverage and plot computed granite phase on the actual meshes."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from sector_check import audit


def extract(folder, mode, dest):
    info, cells = audit(folder, mode)
    # Read the accepted Bernstein coefficients from their field, preserving
    # CellId and local index across independent MPI partitions.
    parameters = json.loads((folder/"parameters.json").read_text())
    names = parameters["Compositional fields"]["Names of fields"]["value"].split(",")
    component = 4+[s.strip() for s in names].index("phase_field")
    patches = {cid:np.full(9, np.nan) for cid in cells}
    for shard in sorted(folder.glob("fields-0000-r*.csv")):
        with shard.open() as stream:
            for row in csv.DictReader(stream):
                if int(row["component"]) == component:
                    patches[row["cell"]][int(row["local"])] = float(row["value"])
    # Sort by physical location so that the reduced figure data are
    # independent of the order in which rank shards were written.
    ids = sorted(cells, key=cells.get)
    phase = np.array([patches[cid] for cid in ids]).reshape(-1, 3, 3)
    assert np.isfinite(phase).all()
    np.savez_compressed(dest/f"granite-{mode}.npz", box=[cells[cid] for cid in ids], phase=phase)
    return info


def draw(dest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.0), sharey=True, layout="constrained")
    nx, ny = 768, 384
    dx = 1.5/nx
    samples = (np.arange(nx)+.5)*dx
    for ax,mode,title in zip(axes, ("uniform","sector","disk"), ("Uniform", "Forward sector", "Tip disk")):
        data = np.load(dest/f"granite-{mode}.npz")
        image = np.full((ny, nx), np.nan)
        segments = []
        for (x, y, h), patch in zip(data["box"], data["phase"]):
            i, j = np.rint([x/dx, y/dx]).astype(int)
            n = int(round(h/dx))
            t = (np.arange(n)+.5)/n
            b = np.column_stack(((1-t)**2, 2*t*(1-t), t*t))
            image[j:j+n, i:i+n] = b@patch@b.T
            segments.append([[x,y], [x+h,y], [x+h,y+h], [x,y+h], [x,y]])
        assert np.isfinite(image).all()
        colors = ax.imshow(image, origin="lower", extent=[0,1.5,0,.75], vmin=0., vmax=.95, cmap="Blues",
                           interpolation="nearest", aspect="equal")
        ax.add_collection(LineCollection(segments, colors="black", alpha=.23, linewidths=.24))
        ax.contour(samples, samples[:ny], image, levels=[.01, .05], colors=["#7c245e","#d47100"], linewidths=1.05)
        ax.plot(1.125, .375, "+", c="black", ms=7)
        ax.set(xlabel="x (m)", title=f"{title} · {len(segments):,} cells", xlim=(0,1.5), ylim=(0,.75),
               xticks=[0,.375,.75,1.125,1.5], yticks=[0,.25,.5,.75])
    axes[0].set_ylabel("y (m)")
    fig.colorbar(colors, ax=axes, shrink=.72, label="Phase")
    fig.savefig(dest/"granite-initial-mesh.png", dpi=180)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("destination", type=Path)
    for option in ("uniform","sector","disk"):
        p.add_argument("--"+option, type=Path)
    args = p.parse_args()
    if any((args.uniform, args.sector, args.disk)):
        assert args.uniform and args.sector and args.disk
        report = [extract(getattr(args, mode), mode, args.destination) for mode in ("uniform","sector","disk")]
        (args.destination/"coverage.json").write_text(json.dumps(report, indent=2)+"\n")
    draw(args.destination)
