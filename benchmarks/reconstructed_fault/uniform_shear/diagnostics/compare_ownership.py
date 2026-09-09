#!/usr/bin/env python3
"""Compare production exports; never replace the baseline or any simulated field."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np


def read(path):
    return np.genfromtxt(path, delimiter=",", names=True)


def checks(rows):
    ids = rows["id"].astype(int)
    sums = np.bincount(ids, weights=rows["w"])
    gradient = np.hypot(np.bincount(ids, weights=rows["gx"]),
                        np.bincount(ids, weights=rows["gy"]))
    return sums, gradient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old_output", type=Path)
    parser.add_argument("corrected_packet", type=Path)
    args = parser.parse_args()
    root = Path(__file__).parent
    baseline = root/"results/raw"
    corrected = args.corrected_packet/"raw"
    old_saved = root/"results-old-dedup"
    old_saved.mkdir(exist_ok=True)
    for name in ("before_phase_cpdi.csv", "before_phase_particles.csv"):
        if (args.old_output/name).resolve() != (old_saved/name).resolve():
            shutil.copy2(args.old_output/name, old_saved/name)
        assert (baseline/name).read_bytes() == (old_saved/name).read_bytes(), name
    for name in ("before_phase_particles.csv", "before_phase_nodes.csv", "before_phase_cells.csv"):
        assert (baseline/name).read_bytes() == (corrected/name).read_bytes(), name
    def physical_parameters(directory):
        return [line for line in (directory/"parameters.prm").read_text().splitlines()
                if not line.startswith("set Output directory ")]
    assert physical_parameters(baseline) == physical_parameters(corrected)
    assert (corrected/"before_phase_particles.csv").read_bytes() == (corrected/"pre_mechanics_particles.csv").read_bytes()
    assert (corrected/"before_phase_cpdi.csv").read_bytes() == (corrected/"pre_mechanics_cpdi.csv").read_bytes()
    old, new = read(baseline/"before_phase_cpdi.csv"), read(corrected/"before_phase_cpdi.csv")
    old_sums, old_grad = checks(old)
    sums, grad = checks(new)
    bad = np.flatnonzero(abs(old_sums-1)>1e-12)
    assert len(bad) == 68
    assert np.max(abs(sums-1)) < 1e-12
    assert np.max(grad) < 1e-10
    unaffected_old = old[~np.isin(old["id"], bad)]
    unaffected_new = new[~np.isin(new["id"], bad)]
    assert np.array_equal(unaffected_old, unaffected_new)
    report = dict(old_tolerance_stencils_and_particles_byte_identical=True,
                  unchanged_initial_H_positions_volumes_mesh_and_phi=True,
                  unaffected_stencil_rows_exactly_unchanged=len(unaffected_old),
                  formerly_bad_particles=len(bad), corrected_bad_particles=0,
                  sum_w_max_error=float(np.max(abs(sums-1))),
                  summed_gradient_max_norm=float(np.max(grad)),
                  captured_particle_sums={str(i):[float(old_sums[i]),float(sums[i])] for i in [3836,3983,3986]})
    for label, packet in (("baseline",root/"results"),("corrected",args.corrected_packet)):
        center = read(packet/"centerline.csv")["FE_phi_y_zero"]
        values = json.loads((packet/"measurements.json").read_text())
        report[label] = dict(center_min=float(min(center)), center_max=float(max(center)),
                             center_range=float(np.ptp(center)), measurements=values)
        if label == "corrected":
            assert values["phase_max_along_x_range"] < 1e-10
    (args.corrected_packet/"ownership_comparison.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
