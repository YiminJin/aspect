"""Compare accepted labels by material index and bulk samples by cell and quadrature index."""
import argparse
import math
from itertools import chain
from pathlib import Path


def read_snapshot(paths, points):
    header = None
    samples = {}
    for path in paths:
        rows = path.read_text().splitlines()
        if not rows:
            raise ValueError(f"{path}: empty snapshot")
        labels = tuple(map(float, rows[0].split()))
        if len(labels) < 12 or (len(labels) - 2) % 5:
            raise ValueError(f"{path}: invalid time and label record")
        if header is not None and labels != header:
            raise ValueError(f"{path}: ranks disagree on clock or accepted labels")
        header = labels
        for row in rows[1:]:
            cell, q, *fields = row.split()
            key = (cell, int(q))
            if key in samples or len(fields) != 10:
                raise ValueError(f"{path}: duplicate or incomplete sample {key}")
            samples[key] = tuple(map(float, fields))
    if len(samples) != points or header is None:
        raise ValueError(f"Expected {points} bulk points, found {len(samples)} in {paths}")
    return header, samples


def compare_step(reference, output, step, *, rel_tol=1.e-11, abs_tol=1.e-12, points=36):
    def load(directory):
        return read_snapshot(sorted(directory.glob(f"fault-{step}-*.txt")), points)

    a, u = load(reference)
    b, v = load(output)
    if len(a) != len(b) or u.keys() != v.keys():
        raise ValueError(f"Step {step}: label count or cell/quadrature identities differ")
    maximum = 0.
    for key, lhs, rhs in chain([("labels", a, b)], ((k, u[k], v[k]) for k in u)):
        for column, (x, y) in enumerate(zip(lhs, rhs)):
            if not (math.isfinite(x) and math.isfinite(y) and math.isclose(x, y, rel_tol=rel_tol, abs_tol=abs_tol)):
                raise ValueError(f"Step {step}, {key}, column {column}: expected {x}, found {y}")
            maximum = max(maximum, abs(x-y))
    return maximum


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--points", type=int, default=36)
    args = parser.parse_args()
    for step in range(3):
        difference = compare_step(args.reference, args.output, step, rel_tol=1.e-8, abs_tol=1.e-10, points=args.points)
        print(f"Step {step}: maximum absolute difference {difference:.3e}")
