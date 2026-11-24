#!/usr/bin/env python3
"""One finite material label enters the adjacent cell with its old age."""
from csv import DictReader
from math import exp, expm1, floor, isclose, tanh
from pathlib import Path

directory = Path("output-layered_moving_fault")
def labels(step):
    with (directory / f"material-{step}-surface-0.csv").open() as stream:
        return [{k: float(v) for k, v in row.items()} for row in DictReader(stream)]
old, new = labels(1), labels(2)
if len(new) != 20 or len(old) != 20:
    raise ValueError("Missing material label")
crossings = 0
for i, (a, b) in enumerate(zip(old, new)):
    if a["i"] != i or b["i"] != i:
        raise ValueError("Persistent IDs changed")
    crossings += floor(2*a["x"]) != floor(2*b["x"])
    v = b["V_m_s"]
    age = a["Theta_s"]*exp(-v*.0125/.008)-.008/v*expm1(-v*.0125/.008)
    eta = 5.5+4.5*tanh(b["y"]/.35)
    for actual, expected in ((b["Theta_s"], age), (b["slip"], a["slip"]+.0125*v),
                             (b["eta"], eta), (b["Z_Pa_m"], exp(-.125/eta)*a["Z_Pa_m"]-eta*expm1(-.125/eta)*v)):
        if not isclose(actual, expected, rel_tol=2e-13, abs_tol=1e-14):
            raise ValueError(f"Label {i} failed its material history")
# These donor rates are solved on the two sides of the cell crossing.
for row, expected in ((old[9], .16239432820602381), (new[9], .15601092584590529)):
    if not isclose(row["V_m_s"], expected, rel_tol=0., abs_tol=1e-8):
        raise ValueError("The crossing label differs from the frozen solved rate")
if not crossings:
    raise ValueError("No material label crossed a cell")
print("The finite fault crossed a cell and published its layered material history.")
