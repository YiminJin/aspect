#!/usr/bin/env python3
"""Audit the unchanged K1 Armijo search from its signed GDB transcript.

The only nonlinear term along this frozen, prescribed-pressure direction is
friction. Subtract its independently evaluated Taylor remainder to check the
recovered Newton equation; do not reinterpret a residual norm as a signed value.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import re

import numpy as np


def scalar_values(text):
    return [float(v) for v in re.findall(r"\$\d+ = ([\d.eE+-]+)\n", text)]


def vector_after(text, marker):
    text = text.split(marker, 1)[1]
    match = re.search(r"= \{std::vector of length \d+, capacity \d+ = \{([^}]+)\}", text)
    return np.array([float(v) for v in match[1].split(",")])


def friction_remainder(V, increment):
    # For these data the regularized asinh argument exceeds 1e12. Its
    # difference from log(2x), including the tangent, is below 1e-23 Pa.
    # log1p avoids subtracting two ~0.7 friction coefficients.
    ratio = increment / V
    return 25.0 * (ratio - np.log1p(ratio))


def strong_rms(weak, mass):
    return math.sqrt(max(0.0, float(weak @ np.linalg.solve(mass, weak) / mass.sum())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = args.trace.read_text()
    base = text.split("SEARCH_BASE\n")[1].split("SEARCH_TRIAL\n")[0]
    iteration, alpha_max, bulk_scale, surface_scale, bulk0, surface0, merit0 = scalar_values(base)[:7]
    vectors = re.findall(r"= \{std::vector of length \d+, capacity \d+ = \{([^}]+)\}", base)
    V, direction = [np.array([float(v) for v in a.split(",")]) for a in vectors[:2]]
    r0 = vector_after(base, "BASE_SURFACE")
    diagonal = vector_after(base, "SURFACE_MASS_DIAGONAL")
    off_diagonal = vector_after(base, "SURFACE_MASS_OFF_DIAGONAL")
    mass = np.diag(diagonal) + np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    if not math.isclose(strong_rms(r0, mass), surface0, rel_tol=1e-10):
        raise ValueError("Signed base residual/mass do not reproduce the production norm")
    rows = []
    for trial in text.split("SEARCH_TRIAL\n")[1:]:
        trial = trial.split("SEARCH_EXHAUSTION")[0]
        alpha, bulk, relative_bulk, relative_surface, merit, _ = scalar_values(trial)[:6]
        residual = vector_after(trial, "TRIAL_SURFACE")
        # M times nodal remainder is exact for uniform V/dV. The recorded
        # direction is almost uniform; its Q1 interpolation error is bounded
        # separately in the report, not mistaken for a Jacobian defect.
        remainder = mass @ friction_remainder(V, alpha * direction)
        predicted = (1-alpha)*r0 + remainder
        bound = (1-1e-4*alpha)*merit0
        rows.append(dict(alpha=alpha, bulk_norm=bulk,
                         surface_strong_rms_Pa=strong_rms(residual, mass),
                         relative_bulk=relative_bulk, relative_surface=relative_surface,
                         merit=merit, armijo_bound=bound,
                         accepted=merit <= bound,
                         rejection="Armijo inequality" if merit > bound else "none",
                         nonlinear_remainder_rms_Pa=strong_rms(remainder, mass),
                         remainder_subtracted_Newton_defect_Pa=strong_rms(residual-predicted, mass)/alpha))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/"candidates.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    summary = dict(nonlinear_iteration=int(iteration), maximum_step=alpha_max,
                   bulk_scale=bulk_scale, surface_scale_Pa=surface_scale,
                   initial_bulk_norm=bulk0, initial_surface_rms_Pa=surface0,
                   initial_merit=merit0, initial_mean_V_m_s=float(V.mean()),
                   direction_min_m_s=float(direction.min()), direction_max_m_s=float(direction.max()),
                   signed_initial_surface_mean_Pa=float(r0.sum()/mass.sum()),
                   characteristic_surface_scale_Pa=surface_scale/math.sqrt(np.finfo(float).eps),
                   implied_surface_convergence_target_Pa=1e-8*surface_scale,
                   max_remainder_subtracted_Newton_defect_Pa=max(r["remainder_subtracted_Newton_defect_Pa"] for r in rows),
                   rejected_candidates=sum(not r["accepted"] for r in rows))
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
