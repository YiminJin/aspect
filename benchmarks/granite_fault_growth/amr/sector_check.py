"""Verify physical coverage and initialized DG phase on the granite AMR meshes."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def footprint_distances(x0, y0, h):
    """Analytic 2-D distance to the seed, disk center and forward triangle."""
    x = x0+h-1.125
    d = max(0., y0-.375, .375-y0-h)
    seed = np.hypot(max(x0-1.125, 0.), d)
    tip = np.hypot(max(x0-1.125, -x, 0.), d)
    proj = np.clip((x+d)/2, 0, .375)
    sector = 0 if x >= d else np.hypot(x-proj, d-proj)
    return seed, tip, sector


def audit(folder, mode):
    n0 = 96 if mode == "uniform" else 12
    cells, samples = {}, 0
    max_error = 0.
    area = 0.
    for filename in sorted(folder.glob("points-0000-r*.csv")):
        with filename.open() as stream:
            for r in csv.DictReader(stream):
                cid = r["cell"]
                x, y, phi, w = map(float, (r["x"], r["y"], r["phi"], r["weight"]))
                assert np.isfinite([x, y, phi, w]).all() and w > 0
                if cid not in cells:
                    level = int(cid.split("_")[1].split(":")[0])
                    h = 1.5/n0/2**level
                    cells[cid] = (np.floor(x/h)*h, np.floor(y/h)*h, h)
                expected = .95*max(1-abs(y-.375)/.125, 0)**2*np.clip((1.125-x)/.125, 0, 1)**2
                max_error = max(max_error, abs(phi-expected))
                area += w
                samples += 1
    assert abs(area-1.125) < 1e-10 and max_error < 1e-12
    missing = 0
    for x, y, h in cells.values():
        seed, tip, sector = footprint_distances(x, y, h)
        required = seed <= .125 or (tip <= .25 if mode == "disk" else sector <= .125)
        missing += int(required and h > .015625)
    assert min(v[2] for v in cells.values()) == .015625 and missing == 0
    sections = np.genfromtxt(folder/"sections-0000.csv", delimiter=",", names=True)
    assert len(sections) == 96
    np.testing.assert_allclose(sections["xi"], (np.arange(96)+.5)*.015625, atol=1e-14, rtol=0)
    np.testing.assert_array_equal(sections["formed"], 0)
    return dict(mode=mode, cells=len(cells), native_samples=samples, finest_m=.015625,
                uncovered_cells=missing, phase_error=max_error, area_m2=area), cells


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("mode", choices=["sector","disk","uniform"])
    parser.add_argument("--mesh", type=Path)
    args = parser.parse_args()
    result, cells = audit(args.folder, args.mode)
    print(json.dumps(result, indent=2))
    if args.mesh:
        np.savez_compressed(args.mesh, box=np.array(list(cells.values())))
