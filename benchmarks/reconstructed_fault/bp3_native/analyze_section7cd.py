#!/usr/bin/env python3
"""Assess the selected 7C--7D controls using retained physical-coordinate data."""
import csv
import json
import os

import numpy as np

from analyze_section7b import ROOT, RUNS, FIELDS, SCALES, read, rms, oscillations, cut_jumps


OUTPUT = ROOT / "build-bp3-native/section7cd"
CASES = {"baseline": "7b-fe-perturbed", "time": "7cd-time", "q6": "7cd-q6",
         "surface": "7cd-surface", "bulk": "7cd-bulk"}
PAIRS = [("temporal", "baseline", "time"), ("quadrature", "baseline", "q6"),
         ("surface", "q6", "surface"), ("bulk", "surface", "bulk")]


def metric(s, a, b, scale):
    e = b-a
    mean = float(np.sum(np.diff(s)*(e[:-1]+e[1:])/2)/(s[-1]-s[0]))
    return dict(max_abs=float(np.max(np.abs(e))), rms_abs=rms(s, e), scale=scale,
                max_relative=float(np.max(np.abs(e)))/scale,
                rms_relative=rms(s, e)/scale, signed_mean=mean,
                oscillatory_rms=rms(s, e-mean))


def profiles(directory):
    result = {}
    for path in directory.glob("coupled-profile-*.csv"):
        data = read(path)
        result[int(data["time_s"][0])] = (int(path.stem.split("-")[-1]), data)
    return result


def normal_diagnostics(p):
    result = {}
    for name, selected in {"full": np.ones(len(p["s_m"]), dtype=bool),
                           "lower_200m": p["s_m"] >= p["s_m"][-1]-200}.items():
        s = p["s_m"][selected]
        pressure, tensor = p["pressure_Pa"][selected], p["deviatoric_Pa"][selected]
        raw, filtered = p["raw_normal_Pa"][selected]-50e6, p["filtered_normal_Pa"][selected]-50e6
        result[name] = dict(raw_extrema=[float(min(raw)), float(max(raw))],
                            filtered_extrema=[float(min(filtered)), float(max(filtered))],
                            pressure_max=float(np.max(np.abs(pressure))),
                            tensor_max=float(np.max(np.abs(tensor))),
                            cancellation_ratio=(rms(s, pressure)+rms(s, tensor))/max(rms(s, pressure+tensor), 1e-20))
    return result


def phase_cut(directory):
    paths = list(directory.glob("*bulk-normal-cuts-0.csv"))
    data = {}
    for path in paths:
        with path.open() as stream:
            for row in csv.DictReader(stream):
                key = (int(row["station"]), round(float(row["r_m"]), 8))
                data.setdefault(key, []).append(float(row["phi"]))
    grid = np.arange(-150., 151., 2.)
    values = []
    for station in range(1, 4):
        samples = sorted((r, np.mean(v)) for (j, r), v in data.items() if j == station)
        values.extend(np.interp(grid, [x[0] for x in samples], [x[1] for x in samples]))
    return np.array(values)


def mpi_assessment(runs):
    cases = {name: RUNS/("7cd-"+name) for name in ["serial", "mpi", "retry"]}
    result = dict(cases={}, comparisons={})
    for name, directory in cases.items():
        if "bp3-native-7cd-"+name not in runs:
            continue
        assert runs["bp3-native-7cd-"+name]["exit_code"] == 0, name
        journals = [read(p) for p in directory.glob("*coupled-checks.csv")]
        expected_ranks = 1 if name == "serial" else 2
        assert len(journals) == expected_ranks
        for journal in journals:
            assert journal["step"].tolist() == list(range(4))
            assert journal["time_s"].tolist() == [0, 200000, 400000, 600000]
            assert np.all(journal["publications"] == 1)
        assert all(sum(j["particles"][i] for j in journals) == 184448 for i in range(4))
        accepted = read(directory/"accepted_steps.csv")
        result["cases"][name] = dict(ranks=expected_ranks,
                                    max_stress_defect_Pa=max(float(max(j["stress_error_Pa"])) for j in journals),
                                    stress_scale_Pa=max(float(max(j["stress_scale_Pa"])) for j in journals),
                                    max_residual=float(max(accepted["normalized_nonlinear_residual"])),
                                    newton=accepted["newton_updates"].tolist(),
                                    krylov=accepted["krylov_iterations"].tolist())
        if name == "retry":
            failures = [read(p) for p in directory.glob("*candidate-rejection.csv")]
            assert len(failures) == 2
            for failure in failures:
                for field, expected in dict(step=2, time_s=400000, dt_s=200000, owner=1,
                                            caught=1, accepted_histories_unchanged=1,
                                            published_before_retry=0).items():
                    assert failure[field].tolist() == [expected]
            result["cases"][name]["coordinated_rejection"] = "pass on both ranks, no publication before identical-dt retry"
    for first, second in [("serial", "mpi"), ("mpi", "retry")]:
        if first not in result["cases"] or second not in result["cases"]:
            continue
        stress_scale = max(result["cases"][first]["stress_scale_Pa"], result["cases"][second]["stress_scale_Pa"], 1.)
        scales = dict(SCALES, slip_m=1e-9*600000)
        for field in ["shear_Pa", "raw_normal_Pa", "filtered_normal_Pa", "pressure_Pa", "deviatoric_Pa"]:
            scales[field] = stress_scale
        entries = []
        for step in range(4):
            a, b = (read(cases[n]/f"coupled-profile-{step}.csv") for n in [first, second])
            assert np.array_equal(a["s_m"], b["s_m"])
            s = a["s_m"]
            fields = {}
            for field in FIELDS+["I_h_m", "Z_Pa_m"]:
                scale = scales.get(field, max(float(max(np.abs(a[field]))), float(max(np.abs(b[field]))), 1.))
                allowance = (1e-10 if field == "I_h_m" else 1e-7)*scale
                fields[field] = dict(metric(s, a[field], b[field], scale), allowance=allowance)
                assert fields[field]["max_abs"] <= allowance, (first, second, step, field, fields[field])
            physical = []
            for name in [first, second]:
                values = {}
                for path in cases[name].glob(f"*physical-fe-{step}.csv"):
                    with path.open() as stream:
                        for row in csv.DictReader(stream):
                            key = (int(row["component"]), round(float(row["x"]), 10), round(float(row["y"]), 10))
                            entry = (float(row["accepted"]), float(row["history"]))
                            # A coarse Q2 edge midpoint and a fine-cell vertex
                            # can have distinct constrained DoFs at the same
                            # physical point. Check their coincident trace, then
                            # compare that physical value once across partitions.
                            if key in values:
                                assert values[key] == entry, (name, step, key, values[key], entry)
                            else:
                                values[key] = entry
                physical.append(values)
            assert physical[0] and physical[0].keys() == physical[1].keys()
            phase_component = max(key[0] for key in physical[0])
            maximum = dict(velocity=0., pressure=0., phase=0., history=0.)
            for key, values in physical[0].items():
                other = physical[1][key]
                component = key[0]
                field = "velocity" if component < 2 else ("phase" if component == phase_component else "pressure")
                maximum[field] = max(maximum[field], abs(values[0]-other[0]))
                maximum["history"] = max(maximum["history"], abs(values[1]-other[1]))
            allowances = dict(velocity=1e-7*1e-9, pressure=1e-7*stress_scale,
                              phase=1e-13, history=1e-7*1e-9*600000)
            assert all(maximum[f] <= allowances[f] for f in maximum), (first, second, step, maximum, allowances)
            entries.append(dict(step=step, fields=fields, physical_FE_maximum=maximum,
                                physical_FE_allowances=allowances, physical_coefficients=len(physical[0])))
        result["comparisons"][first+"_vs_"+second] = entries
    return result


def plot_profiles(loaded):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/bp3-section7cd-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    series = [("V_m_s", 1e-9, 0., "V / Vp"), ("theta_s", 8e6, 0., "theta / theta0"),
              ("Omega", 1., 0., "Omega"), ("slip_m", .002, 0., "slip / (Vp T)"),
              ("shear_Pa", 1000., SCALES["shear_Pa"], "shear perturbation [kPa]"),
              ("raw_normal_Pa", 1000., 50e6, "raw normal perturbation [kPa]"),
              ("pressure_Pa", 1000., 0., "pressure [kPa]"),
              ("deviatoric_Pa", 1000., 0., "tensor contribution [kPa]")]
    with PdfPages(OUTPUT/"profiles.pdf") as pdf:
        fig, axes = plt.subplots(4, 2, figsize=(10, 11), sharex=True)
        for name, history in loaded.items():
            p = history[2000000][1]
            for ax, (field, scale, background, label) in zip(axes.flat, series):
                ax.plot(p["s_m"], (p[field]-background)/scale, label=name, linewidth=1.1)
                ax.set_ylabel(label)
                ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=8)
        for ax in axes[-1]:
            ax.set_xlabel("down-dip distance [m]")
        fig.suptitle("7C: final accepted response, T = 2e6 s; native surface meshes\nSensitivity comparisons, no convergence-order or error-bound claim", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, .95))
        pdf.savefig(fig)
        plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
        for name, history in loaded.items():
            p = history[2000000][1]
            for ax, field in zip(axes.flat, ["raw_normal_Pa", "filtered_normal_Pa", "pressure_Pa", "deviatoric_Pa"]):
                background = 50e6 if "normal" in field else 0.
                ax.plot(p["s_m"][-1]-p["s_m"], (p[field]-background)/1000., ".-", label=name, markersize=3, linewidth=1.)
                ax.set_ylabel(field+" / kPa (background removed)")
                ax.set_xlim(200, 0)
                ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=8)
        for ax in axes[-1]:
            ax.set_xlabel("distance above lower endpoint [m]")
        fig.suptitle("Lower endpoint: dots are actual surface nodes; filter remains 20 m")
        fig.tight_layout(rect=(0, 0, 1, .95))
        pdf.savefig(fig)
        plt.close(fig)


def assess():
    loaded, checks = {}, {}
    evidence = ROOT / "benchmarks/reconstructed_fault/refactoring_boundary/evidence"
    runs = {p.stem: json.loads(p.read_text()) for p in evidence.glob("bp3-native-7cd-*.json")}
    for name, path in CASES.items():
        directory = RUNS / path
        if name != "baseline":
            result = runs.get("bp3-native-7cd-"+name)
            if result is None:
                continue
            assert result["exit_code"] == 0, name
        loaded[name] = profiles(directory)
        expected = 20 if name == "time" else 10
        dt = 100000 if name == "time" else 200000
        journal, accepted = read(directory/"coupled-checks.csv"), read(directory/"accepted_steps.csv")
        assert journal["step"].tolist() == list(range(expected+1)), name
        assert journal["time_s"].tolist() == [i*dt for i in range(expected+1)], name
        assert np.all(journal["publications"] == 1), name
        p = loaded[name][2000000][1]
        checks[name] = dict(positive_steps=expected,
                            max_stress_error_Pa=float(max(journal["stress_error_Pa"])),
                            max_allowance_fraction=float(max(journal["error_fraction_of_allowance"])),
                            max_normalized_residual=float(max(accepted["normalized_nonlinear_residual"])),
                            max_surface_RMS_Pa=float(max(accepted["surface_RMS_Pa"])),
                            max_theta_error=float(max(accepted["Theta_relative_error"])),
                            quadrature_points=int(journal["mechanical_qps"][0]),
                            surface_nodes=len(p["s_m"]), min_I=float(min(p["I_h_m"])),
                            final_normal=normal_diagnostics(p),
                            roughness={field: oscillations(p["s_m"], p[field], SCALES.get(field, max(rms(p["s_m"], p[field]), 1.)))
                                       for field in ["V_m_s", "raw_normal_Pa", "filtered_normal_Pa", "pressure_Pa", "deviatoric_Pa"]},
                            cut_jumps=cut_jumps(directory, expected))
    comparisons = {}
    for label, first, second in PAIRS:
        if first not in loaded or second not in loaded:
            continue
        entries = []
        for time in sorted(set(loaded[first]) & set(loaded[second])):
            a, b = loaded[first][time][1], loaded[second][time][1]
            s = np.unique(np.r_[a["s_m"], b["s_m"]])
            fields = {}
            for field in FIELDS+["I_h_m", "Z_Pa_m"]:
                aa, bb = np.interp(s, a["s_m"], a[field]), np.interp(s, b["s_m"], b[field])
                scale = SCALES.get(field, max(rms(s, aa), rms(s, bb), 1.))
                fields[field] = metric(s, aa, bb, scale)
                if field in {"raw_normal_Pa", "filtered_normal_Pa", "shear_Pa"}:
                    background = 50e6 if "normal" in field else SCALES["shear_Pa"]
                    pert = max(rms(s, aa-background), rms(s, bb-background), 1.)
                    fields[field]["perturbation_scale"] = pert
                    fields[field]["rms_relative_to_perturbation"] = fields[field]["rms_abs"]/pert
            entries.append(dict(time_s=time, fields=fields))
        comparisons[label] = dict(first=first, second=second, common_times=entries)
        if label in {"temporal", "quadrature"}:
            for file in ["initial-mesh.csv", "initial-phase.csv", "initial-particle-identity.csv"]:
                assert (RUNS/CASES[first]/file).read_bytes() == (RUNS/CASES[second]/file).read_bytes()
            assert entries[0]["fields"]["I_h_m"]["max_abs"] == 0.
        a, b = phase_cut(RUNS/CASES[first]), phase_cut(RUNS/CASES[second])
        comparisons[label]["phase_on_three_common_normal_cuts"] = dict(max_abs=float(max(np.abs(b-a))),
                                                                      RMS=float(np.sqrt(np.mean((b-a)**2))))
    if "time" in loaded:
        assert (RUNS/CASES["baseline"]/"coupled-profile-0.csv").read_bytes() == (RUNS/CASES["time"]/"coupled-profile-0.csv").read_bytes()
    report = dict(checks=checks, comparisons=comparisons, runs=runs,
                  mpi=mpi_assessment(runs),
                  simulation_seconds=sum(v["seconds"] for v in runs.values()),
                  conventions=["Differences are second minus first at common physical arc lengths/times.",
                               "Exact Q1 norms on the union of actual surface meshes; Omega is the exported nodal product.",
                               "Roughness uses each native surface mesh; interpolated output is not refinement.",
                               "No order or spatial-error bound follows from these sensitivity pairs."])
    (OUTPUT/"comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    if set(loaded) == set(CASES):
        plot_profiles(loaded)
    with (OUTPUT/"common-time-differences.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["comparison", "time_s", "field", "max_abs", "rms_abs", "scale", "max_relative", "rms_relative"])
        for name, comparison in comparisons.items():
            for entry in comparison["common_times"]:
                for field, values in entry["fields"].items():
                    writer.writerow([name, entry["time_s"], field,
                                     *[values[k] for k in ["max_abs", "rms_abs", "scale", "max_relative", "rms_relative"]]])
    print(json.dumps({name: dict(final=values["common_times"][-1], phase=values["phase_on_three_common_normal_cuts"])
                      for name, values in comparisons.items()}, indent=2))


if __name__ == "__main__":
    assess()
