#!/usr/bin/env python3
"""One finite material label enters the adjacent cell with its old age."""
from csv import DictReader
from math import exp, expm1, floor, isclose, tanh
from pathlib import Path

directory = Path("output-layered_moving_fault")
def labels(step):
    with (directory / f"material-{step}-surface-0.csv").open() as stream:
        return [{k: float(v) for k, v in row.items()} for row in DictReader(stream)]
incoming, accepted = labels(1), labels(2)
if len(accepted) != 20 or len(incoming) != 20:
    raise ValueError("Missing material label")
crossings = 0
for i, (parent, endpoint) in enumerate(zip(incoming, accepted)):
    if parent["i"] != i or endpoint["i"] != i:
        raise ValueError("Persistent IDs changed")
    crossings += floor(2*parent["x"]) != floor(2*endpoint["x"])
    rate = endpoint["V_m_s"]
    # Integrate the aging law for 12.5 ms at the returned rate (Dc = 8 mm).
    age = parent["Theta_s"]*exp(-rate*.0125/.008)-.008/rate*expm1(-rate*.0125/.008)
    viscosity = 5.5+4.5*tanh(endpoint["y"]/.35)
    for actual, expected in ((endpoint["Theta_s"], age),
                             (endpoint["slip"], parent["slip"]+.0125*rate),
                             (endpoint["eta"], viscosity),
                             (endpoint["Z_Pa_m"], exp(-.125/viscosity)*parent["Z_Pa_m"]-viscosity*expm1(-.125/viscosity)*rate)):
        if not isclose(actual, expected, rel_tol=2e-13, abs_tol=1e-14):
            raise ValueError(f"Label {i} failed its material history")
# These donor rates are solved on the two sides of the cell crossing.
for row, expected in ((incoming[9], .16239432820602381), (accepted[9], .15601092584590529)):
    if not isclose(row["V_m_s"], expected, rel_tol=0., abs_tol=1e-8):
        raise ValueError("The crossing label differs from the frozen solved rate")
if not crossings:
    raise ValueError("No material label crossed a cell")
print("The finite fault crossed a cell and published its layered material history.")
