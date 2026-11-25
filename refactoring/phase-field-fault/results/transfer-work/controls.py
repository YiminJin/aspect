"""Report the fitted 32 x 32 trajectory's shifts from the frozen labels and pre-repair DG stress."""
import argparse
import csv
from pathlib import Path

import numpy as np


def load_sorted_bulk(path):
    rows = np.genfromtxt(path, names=True, dtype=None, encoding="utf8")
    return np.sort(rows, order=["cell", "q"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path, help="Baseline free-32/output")
    parser.add_argument("after", type=Path, help="Repaired free-32/output")
    parser.add_argument("reference", type=Path, help="Unmodified corrected 3A label CSV")
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    source = np.loadtxt(args.reference, delimiter=",", skiprows=1)
    with args.report.open("w") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["clock_s", "field", "worst_id", "reference", "current", "signed_change", "outside_frozen_budget"])
        for step, clock in enumerate([0., .0125, .025]):
            ref = source[source[:, 7] == clock]
            old, new = (np.loadtxt(path / f"labels-{step}-0.txt", skiprows=1)
                        for path in (args.before, args.after))
            np.testing.assert_array_equal(ref[:, 0], np.arange(38))
            np.testing.assert_array_equal(old[:, [0, 7]], ref[:, [0, 7]])
            np.testing.assert_allclose(old[:, 1:7], ref[:, 1:7], rtol=1.e-8, atol=1.e-10)
            np.testing.assert_array_equal(new[:, [0, 7]], ref[:, [0, 7]])
            np.testing.assert_array_equal(old[:, 1:4], new[:, 1:4])
            np.testing.assert_array_equal(np.isfinite(new), True)
            for col, name in enumerate(("age_s", "slip_m", "rate_m_per_s"), 4):
                delta = new[:, col] - ref[:, col]
                index = np.argmax(abs(delta))
                outside_budget = np.count_nonzero(abs(delta) > 1.e-10 + 1.e-8 * abs(ref[:, col]))
                writer.writerow([clock, name, int(ref[index, 0]), ref[index, col], new[index, col],
                                 delta[index], outside_budget])

            filename = f"bulk-{step}-0.txt"
            a, b = (load_sorted_bulk(path / filename) for path in (args.before, args.after))
            for key in ("cell", "q", "x", "y", "phi", "chi"):
                np.testing.assert_array_equal(a[key], b[key])
            for key in ("xx", "yy", "xy", "p"):
                delta = b[key] - a[key]
                np.testing.assert_array_equal(np.isfinite(delta), True)
                index = np.argmax(abs(delta))
                writer.writerow([clock, key + "_Pa", f"{a['cell'][index]}/q{a['q'][index]}",
                                 a[key][index], b[key][index], delta[index], ""])


if __name__ == "__main__":
    main()
