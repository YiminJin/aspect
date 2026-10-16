#!/usr/bin/env python3
"""PF2 gates: fresh baseline, same-build controls, full fields and operators."""
import argparse
import json
from pathlib import Path

import numpy as np


def table(path):
    return np.atleast_1d(np.genfromtxt(path, names=True, delimiter=","))


def settings(path, prefix=""):
    result = {}
    for key, value in json.loads(path.read_text()).items():
        label = prefix+key
        if isinstance(value, dict) and "value" in value:
            result[label] = value["value"]
        elif isinstance(value, dict):
            # Parameter JSON has arbitrary subsection nesting.
            result.update(flatten(value, label+"/"))
    return result


def flatten(data, prefix):
    result = {}
    for key, value in data.items():
        if isinstance(value, dict) and "value" in value:
            result[prefix+key] = value["value"]
        elif isinstance(value, dict):
            result.update(flatten(value, prefix+key+"/"))
    return result


def compare(first, second):
    rows = []
    for name in [f"bulk-{i}.csv" for i in range(4)]+[f"surface-{i}.csv" for i in range(4)]+["storage.csv"]:
        a, b = table(first/"run"/name), table(second/"run"/name)
        assert a.shape == b.shape, name
        for column in sorted(set(a.dtype.names) & set(b.dtype.names)):
            error = float(np.max(np.abs(a[column]-b[column])))
            scale = max(1., float(np.max(np.abs(a[column]))))
            # Unit SI scales are fixed; stress/storage tolerances are scaled,
            # while velocity and displacement use 1 m/s and 1 m respectively.
            tolerance = 1.e-12 if column in ("ux", "uy", "V", "slip", "s_elas") else 1.e-10
            if column in ("x", "y", "weight", "phi", "I", "step"):
                tolerance = 0.
            rows.append(dict(file=name, column=column, absolute=error, scaled=error/scale,
                             scale=scale, tolerance=tolerance, passed=error/scale <= tolerance))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--fe", type=Path, required=True)
    parser.add_argument("--dg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite "+str(args.output))
    runs = {key: getattr(args, key) for key in ("baseline", "fe", "dg")}
    records = {key: json.loads((path/"record.json").read_text()) for key, path in runs.items()}
    assert all(r["exit_code"] == 0 for r in records.values())
    assert records["fe"]["executable_sha256"] == records["dg"]["executable_sha256"]
    assert records["fe"]["plugin_sha256"] == records["dg"]["plugin_sha256"]
    assert records["fe"]["source_hashes"] == records["dg"]["source_hashes"]
    assert (args.fe/"run/accepted-phase.csv").read_bytes() == (args.dg/"run/accepted-phase.csv").read_bytes(), "Accepted phase coefficients differ"
    ignored = {"Additional shared libraries", "Output directory", "Material model/Phase field fault/Bulk history representation"}
    resolved = {key: {k: v for k, v in settings(path/"run/parameters.json").items() if k not in ignored}
                for key, path in runs.items()}
    assert resolved["baseline"] == resolved["fe"] == resolved["dg"], "Physical/control inputs changed"

    baseline = compare(args.baseline, args.fe)
    coupled = compare(args.fe, args.dg)
    tensors = []
    for step in range(4):
        a, b = table(args.fe/"run"/f"bulk-{step}.csv"), table(args.dg/"run"/f"bulk-{step}.csv")
        assert np.array_equal(a["weight"], b["weight"])
        difference = ((a["tau_xx"]-b["tau_xx"])**2+(a["tau_yy"]-b["tau_yy"])**2
                      +(a["tau_zz"]-b["tau_zz"])**2+2*(a["tau_xy"]-b["tau_xy"])**2)
        reference = a["tau_xx"]**2+a["tau_yy"]**2+a["tau_zz"]**2+2*a["tau_xy"]**2
        volume = float(np.sum(a["weight"]))
        rms = float(np.sqrt(np.sum(a["weight"]*difference)/volume))
        scaled = rms/max(1., float(np.sqrt(np.sum(a["weight"]*reference)/volume)))
        assert scaled <= 1.e-10
        tensors.append(dict(step=step, max_full3D_error_Pa=float(np.sqrt(np.max(difference))),
                            volume_weighted_RMS_error_Pa=rms, scaled_RMS=scaled, volume_m2=volume))
    loads = []
    for step in (1, 2, 3):
        for name in (f"loads-{step}.csv", f"shear-load-{step}.csv"):
            a, b = table(args.fe/"run"/name), table(args.dg/"run"/name)
            assert a.shape == b.shape
            for column in a.dtype.names:
                error = float(np.max(np.abs(a[column]-b[column])))
                scale = max(1., float(np.max(np.abs(a[column]))))
                loads.append(dict(file=name, column=column, absolute=error, scaled=error/scale,
                                  tolerance=1.e-10, passed=error/scale <= 1.e-10))
    local = {}
    for key in ("fe", "dg"):
        path = runs[key]/"run"
        steps = table(path/"steps.csv")
        assert len(steps) == 3 and np.all(steps["max_q"] > 0.) and np.all(steps["max_s_elas"] > 0.)
        assert np.max(steps["Z_error"]) <= 1.e-10
        assert np.all(steps["phase_change"] == 0.) and np.all(steps["I_change"] == 0.) and np.all(steps["repeat_error"] == 0.)
        operators = table(path/"operator-checks.csv")
        assert np.all(operators["poison_load_error"] == 0.) and np.all(operators["poison_solve_error"] == 0.)
        assert np.max(operators["adjoint_error"]) <= 1.e-10
        assert np.max(np.abs(operators["B_work"]-operators["independent_work"])/np.maximum(1.,np.abs(operators["independent_work"]))) <= 1.e-10
        assert (path/"rejection.txt").exists()
        source = np.concatenate([table(path/f"source-checks-{i}.csv") for i in (0, 3)])
        source_error = source["max_stress_error_Pa"]/source["scale_Pa"]
        assert np.max(source_error) <= 1.e-12
        representation = table(path/"representation.csv")
        assert representation["mass_rank"].tolist()[:2] == [4., 9.]
        assert representation["DGQ1_gradient_L2"][-1] > 1.e-10
        text = (runs[key]/"run.log").read_text()
        publication_label = "DG Maxwell publication:" if key == "dg" else "FE Maxwell publication:"
        assert text.count(publication_label) == 3, "History did not publish exactly once per accepted step"
        local[key] = dict(max_source_error_Pa=float(np.max(source["max_stress_error_Pa"])),
                          max_scaled_source_error=float(np.max(source_error)),
                          max_poison_load_error=float(np.max(operators["poison_load_error"])),
                          max_poison_solve_error=float(np.max(operators["poison_solve_error"])),
                          max_adjoint_error=float(np.max(operators["adjoint_error"])),
                          independent_work_error=float(np.max(np.abs(operators["B_work"]-operators["independent_work"]))),
                          max_q_Pa=float(np.max(steps["max_q"])), max_elastic_slip_m=float(np.max(steps["max_s_elas"])),
                          max_Z_error_Pa_m=float(np.max(steps["Z_error"])),
                          source_cases=len(source), actual_publications=3,
                          mass_ranks=[4,9], DGQ1_source_L2_m_per_s=float(representation["DGQ1_gradient_L2"][-1]))
    result = dict(passed=all(r["passed"] for r in baseline+coupled+loads),
                  same_build=True, resolved_inputs_match=True, baseline_control=baseline,
                  fe_dg_fields=coupled, fe_dg_full3D_stress=tensors, fe_dg_loads=loads, local_checks=local,
                  records={k: {s: r[s] for s in ("label", "head", "executable_sha256", "plugin_sha256", "seconds", "max_rss_KiB")}
                           for k, r in records.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(passed=result["passed"], baseline_max_scaled=max(r["scaled"] for r in baseline),
                         fields_max_scaled=max(r["scaled"] for r in coupled),
                         loads_max_scaled=max(r["scaled"] for r in loads), local=local)))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
