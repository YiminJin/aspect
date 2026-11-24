#!/usr/bin/env python3
"""Check the three clocks, accepted history and the physical work gate without a donor."""
import csv
from math import isclose
from pathlib import Path

output = Path('output-growing_fault')
for step in (1, 2, 3):
    with (output/f'balance-{step:04}.csv').open() as stream:
        balance, = csv.DictReader(stream)
    if (int(balance['step']) != step or not isclose(float(balance['time']), .005*step, abs_tol=1e-15)
        or not abs(float(balance['mechanical_residual'])) <= 1e-9
        or not abs(float(balance['phase_KKT'])) <= 1e-7 or not abs(float(balance['adjoint_error'])) < 1e-12):
        raise ValueError(f'Invalid accepted midpoint step {step}')
with (output/'sections-0003.csv').open() as stream:
    sections = list(csv.DictReader(stream))
if [int(row['section']) for row in sections] != list(range(9)) or any(row['formed'] != '1' for row in sections):
    raise ValueError('Incomplete material plane history')
slip = sum(float(row['ds'])*float(row['slip']) for row in sections)
if not isclose(slip, .000169302306477, rel_tol=1e-7, abs_tol=1e-11):
    raise ValueError(f'Unexpected integrated slip {slip} m^2')
print('Nine material strips passed the coupled operator, returned-state and slip checks through 15 ms.')
