"""Export accepted startup diagnostics on the auxiliary 12-by-6 mesh."""
import csv
import sys
from pathlib import Path

import numpy as np

from initial_field import measure

folder, result = map(Path, sys.argv[1:])
initial = measure(folder, 12, result/"initial-12.npz")
clocks = np.loadtxt(folder/"statistics")[:, 1]
previous = None
rate_rows = []
with (result/"diagnostic-12.csv").open("w") as stream:
    out = csv.writer(stream, lineterminator="\n")
    out.writerow(("endpoint_s", "midpoint_s", "Vmax_m_s", "previous_tip_id", "Vtip_m_s",
                  "tip_001_m", "tip_005_m", "formed", "formed_ahead", "max_phase_ahead",
                  "peak_slip_m", "phase_work_J_m"))
    for step, time in enumerate(clocks):
        def read(table):
            return np.genfromtxt(folder/f"{table}-{step:04d}.csv", names=True, delimiter=",")
        sections = read("sections")
        ids = np.where(sections["formed"] > 0)[0]
        # Every accepted DG coefficient must be identical to its initialized
        # value for reusing either initial contour location in this history.
        fields = np.genfromtxt(folder/f"fields-{step:04d}-r0.csv", names=True, delimiter=",",
                               dtype=None, encoding="utf8")
        phase = fields[fields["component"] == 4]
        if step == 0:
            initial_phase = phase
        np.testing.assert_array_equal(phase, initial_phase)
        midpoint = (clocks[step-1]+time)/2 if step else ""
        tip_id = previous[-1] if previous is not None and len(previous) else None
        work = float(read("balance")["phase_work"]) if step else ""
        out.writerow((time, midpoint, max(abs(sections["V"])) if step else "",
                      tip_id, sections["V"][tip_id] if tip_id is not None else "",
                      *initial["connected_tip_m"].values(), len(ids),
                      sum(sections["xi"][ids] >= 1.125), initial["max_phase_ahead"],
                      max(abs(sections["slip"])), work))
        if step:
            rate_rows.extend((clocks[step-1], time, midpoint, int(row["section"]), row["xi"],
                              row["ds"], int(row["formed"]), row["V"]) for row in sections)
        previous = ids
with (result/"signed-12.csv").open("w") as stream:
    out = csv.writer(stream, lineterminator="\n")
    out.writerow(("start_s", "end_s", "midpoint_s", "id", "x_m", "width_m", "formed", "V_m_s"))
    out.writerows(rate_rows)
