#!/usr/bin/env python3
"""Check the physical upper-slab moments in the ordinary one-step calculation."""
from math import isclose
from pathlib import Path

records = Path("output-explicit_dg_slabs/coefficients-1-0").read_text().splitlines()
if tuple(map(float, records[0].split())) != (.01, .01):
    raise ValueError("Unexpected SSPRK3 endpoint")
expected = {"damage": .18748775, "tracer": -.06251225}
checked = set()
for record in records[1:]:
    cell, field, fe, x, y, z, xmax, ymax, zmax, *values = record.split()
    if float(z) == .5 and field in expected:
        # Each quadratic Bernstein polynomial integrates to 1/3. The nodal
        # basis integrates to the Simpson weights. The slab volume is 1/2.
        weights = (1/3,)*3 if field == "damage" else (1/6, 4/6, 1/6)
        if len(values) != 27 or field in checked:
            raise ValueError(f"Invalid upper-slab record for {field}")
        mass = .5*sum(float(v)*weights[i%3]*weights[(i//3)%3]*weights[i//9]
                      for i, v in enumerate(values))
        if not isclose(mass, expected[field], abs_tol=1e-12, rel_tol=0):
            raise ValueError(f"Upper-slab integral for {field} is {mass}")
        checked.add(field)
if checked != expected.keys():
    raise ValueError("Missing upper-slab fields")
print("Both upper-slab moments agree within 1e-12.")
