"""Compare label histories and bulk fields in the fixed fault test probe format."""
import math
from pathlib import Path
import sys


def compare(path, other, *, rel_tol=1.e-11, abs_tol=1.e-12):
    """Compare identities, clock, section history and unisolvent Q2 samples."""
    rows = path.read_text().splitlines()
    other_rows = other.read_text().splitlines()
    assert len(rows) == len(other_rows) == 37
    largest = 0.
    for i, (lhs, rhs) in enumerate(zip(rows, other_rows)):
        a, b = lhs.split(), rhs.split()
        first_value = 2 if i else 0
        assert a[:first_value] == b[:first_value] and len(a) == len(b)
        for x, y in zip(map(float, a[first_value:]), map(float, b[first_value:])):
            assert math.isclose(x, y, rel_tol=rel_tol, abs_tol=abs_tol), (path, i, x, y)
            largest = max(largest, abs(x-y))
    return largest


if __name__ == "__main__":
    reference, output = map(Path, sys.argv[1:])
    for step in range(3):
        name = f"fault-{step}-0.txt"
        difference = compare(reference / name, output / name, rel_tol=1.e-8, abs_tol=1.e-10)
        print(f"Stationary donor, step {step}: maximum absolute difference {difference:.3e}")
