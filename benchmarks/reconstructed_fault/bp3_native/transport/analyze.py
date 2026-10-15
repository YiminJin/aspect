#!/usr/bin/env python3
"""Assess native prescribed-flow results; this script launches no simulations."""

import argparse
import csv
import json
import math
from pathlib import Path


def translation_weak_reference(n):
    """Independent Gaussian antiderivative times the periodic Q1 hats at T=1."""
    sigma, center, h = .0125, .613, 1 / n
    weights = [0.] * n
    for i in range(n):
        a, b = i * h, (i + 1) * h
        mass = sigma * math.sqrt(math.pi / 2) * (
            math.erf((b - center) / (math.sqrt(2) * sigma))
            - math.erf((a - center) / (math.sqrt(2) * sigma)))
        moment = center * mass + sigma**2 * (
            math.exp(-.5 * ((a - center) / sigma)**2)
            - math.exp(-.5 * ((b - center) / sigma)**2))
        weights[i] += (b * mass - moment) / h
        weights[(i + 1) % n] += (moment - a * mass) / h
    return [weights[i] * h * (.5 if j in (0, n) else 1)
            for j in range(n + 1) for i in range(n)]


def inspect_failed_domain(path):
    """Analysis-only Voronoi half-planes from saved actual neighboring sites."""
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    parent = next(row for row in rows if row["kind"] == "parent")
    px, py = float(parent["x"]), float(parent["y"])
    h = 1 / 128  # The selected 8-cell, 16-particle-per-direction control.
    polygon = [(-4*h, max(-py, -4*h)), (4*h, max(-py, -4*h)),
               (4*h, min(1-py, 4*h)), (-4*h, min(1-py, 4*h))]
    neighbors = [(float(row["x"])-px, float(row["y"])-py)
                 for row in rows if row["kind"] == "neighbor"]
    neighbors = [(dx, dy) for dx, dy in neighbors if math.hypot(dx, dy) > 1e-12]
    for dx, dy in neighbors:
        rhs = (dx*dx + dy*dy) / 2
        clipped = []
        for a, b in zip(polygon, polygon[1:] + polygon[:1]):
            va, vb = a[0]*dx + a[1]*dy - rhs, b[0]*dx + b[1]*dy - rhs
            if va <= 0:
                clipped.append(a)
            if (va < 0 < vb) or (vb < 0 < va):
                fraction = va / (va-vb)
                clipped.append((a[0]+fraction*(b[0]-a[0]), a[1]+fraction*(b[1]-a[1])))
        polygon = clipped
    independent_area = abs(sum(a[0]*b[1]-b[0]*a[1]
                               for a, b in zip(polygon, polygon[1:] + polygon[:1]))) / 2
    stored = [(float(row["x"])-px, float(row["y"])-py)
              for row in rows if row["kind"] == "vertex"]
    violation = max(x*dx + y*dy - (dx*dx + dy*dy)/2
                    for x, y in stored for dx, dy in neighbors)
    return {"parent": [px, py], "native_area": float(parent["area"]),
            "independent_area": independent_area, "lattice_area": h*h,
            "relative_area_error": (float(parent["area"])-independent_area)/independent_area,
            "maximum_half_plane_violation": violation}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dense-shear-run", type=Path,
                        help="Completed fresh dense shear; retained failed run remains untouched.")
    parser.add_argument("--coarse-shear-run", type=Path,
                        help="Matching corrected 8-particle-per-direction control.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    runs = root / "build-bp3-native/transport-checks/runs"
    names = ["translation-8-p4-v2", "translation-8-p8", "translation-8-p16",
             "translation-16-p8", "shear-dt125", "shear-dt0625", "shear-dt03125",
             "shear-dt03125-p16"]
    cases = {}
    histories = {}
    incomplete = []
    for name in names:
        run = args.dense_shear_run if name == "shear-dt03125-p16" and args.dense_shear_run else runs / name
        if name == "shear-dt03125" and args.coarse_shear_run:
            run = args.coarse_shear_run
        with (run / "transport.csv").open() as stream:
            rows = [{key: float(value) for key, value in row.items()}
                    for row in csv.DictReader(stream)]
        histories[name] = rows
        final = rows[-1]
        if final["time"] != 1:
            incomplete.append(name)
            cases[name] = {"complete": False, "last_published_row": final}
            continue
        assert rows[0]["step"] == 0 and rows[0]["time"] == 0
        assert final["time"] == 1 and len(rows) == round(1 / final["dt"]) + 1
        assert all(row["step"] == step for step, row in enumerate(rows))
        assert all(row["dt"] == final["dt"] for row in rows[1:])
        assert all(row["particles"] == final["particles"] for row in rows)
        assert final["min_crossings"] >= 2
        assert max(row["pulse_particle_error"] for row in rows) == 0
        assert max(row["one_step_defect"] for row in rows) < 1e-12
        assert all(abs(row["pulse_mass"] - rows[0]["pulse_mass"]) < 2e-13 for row in rows)
        assert all(row["pulse_peak"] == rows[0]["pulse_peak"] for row in rows)
        if name.startswith("translation"):
            assert max(row["particle_stress_error"] for row in rows) < 1e-12
            assert max(row["position_error"] for row in rows) < 1e-12
            n = 16 if name.startswith("translation-16") else 8
            exact_weak = translation_weak_reference(n)
            with (run / "weak_pulse.csv").open() as stream:
                actual_weak = list(csv.DictReader(stream))
            quadrature_error = sum(abs(float(row["reference"]) - value)
                                   for row, value in zip(actual_weak, exact_weak)) / sum(exact_weak)
            assert len(actual_weak) == len(exact_weak) and quadrature_error < 1e-10
        else:
            quadrature_error = None
        with (run / "pulse_particles.csv").open() as stream:
            samples = list(csv.DictReader(stream))
        assert samples and all(float(p["pulse"]) == float(p["reference"]) for p in samples)
        cases[name] = {
            "complete": True,
            "final": final,
            "maximum_errors": {
                key: max(row[key] for row in rows)
                for key in ("position_error", "particle_stress_error", "pulse_particle_error",
                            "one_step_defect", "FE_total_rms", "LLS_total_rms",
                            "FE_LLS_difference", "weak_relative_l1", "mass_relative_error")
            },
            "positive_steps": len(rows) - 1,
            "pulse_sample_rows": len(samples),
            "independent_reference_quadrature_error": quadrature_error,
        }

    density = names[:3]
    for coarse, fine in zip(density, density[1:]):
        assert cases[fine]["final"]["weak_relative_l1"] < cases[coarse]["final"]["weak_relative_l1"]
        assert cases[fine]["maximum_errors"]["weak_relative_l1"] < cases[coarse]["maximum_errors"]["weak_relative_l1"]
    coarse, fine = cases["translation-8-p8"], cases["translation-16-p8"]
    assert fine["final"]["FE_total_rms"] < coarse["final"]["FE_total_rms"]
    assert fine["final"]["LLS_total_rms"] < coarse["final"]["LLS_total_rms"]

    temporal = names[4:7]
    ratios = []
    common_times = [row["time"] for row in histories[temporal[0]]]
    common = {}
    for name in temporal:
        indexed = {row["time"]: row for row in histories[name]}
        common[name] = [{key: indexed[time][key]
                         for key in ("time", "position_error", "particle_stress_error", "FE_total_rms", "weak_relative_l1")}
                        for time in common_times]
    for coarse, fine in zip(temporal, temporal[1:]):
        c, f = cases[coarse]["final"], cases[fine]["final"]
        assert f["particle_stress_error"] < c["particle_stress_error"]
        assert f["position_error"] < c["position_error"]
        assert all(b["particle_stress_error"] < a["particle_stress_error"]
                   for a, b in zip(common[coarse][1:], common[fine][1:]))
        ratios.append({"coarse": coarse, "fine": fine,
                       "stress_error_ratio": c["particle_stress_error"] / f["particle_stress_error"],
                       "position_error_ratio": c["position_error"] / f["position_error"]})
    if cases["shear-dt03125-p16"]["complete"]:
        coarse, fine = cases["shear-dt03125"], cases["shear-dt03125-p16"]
        assert fine["final"]["weak_relative_l1"] < coarse["final"]["weak_relative_l1"]
        assert fine["maximum_errors"]["weak_relative_l1"] < coarse["maximum_errors"]["weak_relative_l1"]
    result = {
        "pass": not incomplete,
        "completed_family_refinement_gates": True,
        "incomplete_cases": incomplete,
        "cases": cases,
        "temporal_error_ratios": ratios,
        "temporal_common_times": common,
        "failed_domain": inspect_failed_domain(runs / "shear-dt03125-p16-domain-v2/failed_domain.csv"),
        "scope": "Native RK2/particle transfer/CPDI and native Maxwell kernel in a prescribed-flow harness; no coupled mechanics or objective rotation.",
    }
    if args.dense_shear_run:
        with (args.dense_shear_run / "geometry.csv").open() as stream:
            geometry = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
        assert len(geometry) == 33 and geometry[-1]["time"] == 1
        keys = [k for k in geometry[0] if k not in ("step", "time", "native_area", "independent_area",
                                                   "min_y_derivative", "max_y_derivative")]
        maxima = {k: max(row[k] for row in geometry) for k in keys}
        assert max(abs(row["native_area"] - 1) for row in geometry) < 1e-10
        assert max(abs(row["independent_area"] - 1) for row in geometry) < 1e-10
        assert all(maxima[k] < 1e-10 for k in keys)
        result["geometry"] = {"run": str(args.dense_shear_run), "maxima": maxima,
                              "minimum_y_derivative": min(row["min_y_derivative"] for row in geometry),
                              "maximum_y_derivative": max(row["max_y_derivative"] for row in geometry),
                              "steps": len(geometry) - 1}
        evidence = args.dense_shear_run.parent.parent
        result["coordinate_probes"] = {}
        for name in ("fullpatch", "local"):
            with (evidence / f"coordinate-probe-{name}.csv").open() as stream:
                rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
            assert len(rows) == 48
            keys = [k for k in rows[0] if k not in ("conditioned", "scale", "shift_x", "shift_y", "vertices")]
            groups = {str(mode): {k: max(abs(row[k]) for row in rows if row["conditioned"] == mode)
                                 for k in keys} for mode in (0, 1)}
            assert all(value < 1e-10 for value in groups["1"].values())
            result["coordinate_probes"][name] = {"maxima": groups, "cases": rows}
        areas, y_derivatives = {}, {}
        with (evidence / "runs/shear-original/domains-step2.csv").open() as stream:
            for row in csv.DictReader(stream):
                i = int(row["id"])
                if row["kind"] == "parent":
                    areas[i] = float(row["a"])
                if row["kind"] == "operator":
                    node = int(row["x"])
                    y_derivatives[i] = y_derivatives.get(i, 0) + (node // 8) / 8 * float(row["c"])
        values = [y_derivatives[i] / area for i, area in areas.items()]
        result["original_linear_gradient"] = {"domains": len(values),
                                               "negative_y_derivatives": sum(v < 0 for v in values),
                                               "minimum": min(values), "maximum": max(values)}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"pass": not incomplete, "cases": len(cases), "output": str(args.output)}, indent=2))
    if incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
