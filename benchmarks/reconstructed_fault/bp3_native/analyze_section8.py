#!/usr/bin/env python3
"""Bounded section-8 assessment using the existing physical-profile diagnostics."""
import csv
import hashlib
import json
import os

import numpy as np

from analyze_section7b import ROOT, RUNS, FIELDS, SCALES, read, rms, oscillations
from analyze_section7cd import metric, profiles, normal_diagnostics

OUTPUT = ROOT / "build-bp3-native/section8"


def profile_check():
    cases = {}
    values = []
    for name in ["base", "fine"]:
        directory = RUNS / ("8-profile-base-v2" if name == "base" else "8-profile-fine")
        groups = {}
        with (directory/"initial-profile-cuts.csv").open() as stream:
            for row in csv.DictReader(stream):
                key = (round(float(row["s_m"]), 8), float(row["r_m"]), int(row["exterior"]))
                groups.setdefault(key, []).append([float(row[k]) for k in ["phi", "physical_phi", "I_h_m", "h", "chi_per_m"]])
        # Duplicate traces are continuous Q1 values; check before grouping.
        for group in groups.values():
            assert np.max(np.ptp(group, axis=0)/np.array([1., 1., 13000., 1000., .1])) < 1e-10
        data = {key: np.mean(group, axis=0) for key, group in groups.items()}
        values.append(data)
        cases[name] = dict(summary={k: float(v[0]) for k, v in read(directory/"initial-profile-summary.csv").items()},
                           points=len(data), exterior_points=sum(k[2] for k in data),
                           max_physical_phase_error=max(abs(v[0]-v[1]) for v in data.values()))
    assert values[0].keys() == values[1].keys(), "Profile probes must use identical physical points."
    regions = {}
    length = max(key[0] for key in values[0])
    for name, predicate in {"all": lambda k: True, "endpoint_50m": lambda k: min(k[0], length-k[0]) <= 50.,
                            "exterior_completion": lambda k: k[2] == 1,
                            "interior": lambda k: min(k[0], length-k[0]) > 50.}.items():
        keys = [k for k in sorted(values[0]) if predicate(k)]
        a, b = np.array([values[0][k] for k in keys]), np.array([values[1][k] for k in keys])
        fields = {}
        for j, field in enumerate(["phi", "physical_phi", "I_h_m", "h", "chi_per_m"]):
            scale = max(float(np.max(np.abs(a[:, j]))), float(np.max(np.abs(b[:, j]))), 1e-20)
            e = b[:, j]-a[:, j]
            fields[field] = dict(max_abs=float(max(abs(e))), RMS=float(np.sqrt(np.mean(e*e))),
                                 scale=scale, max_relative=float(max(abs(e)))/scale)
        regions[name] = dict(points=len(keys), fields=fields)
    ia, ib = [read(RUNS/("8-profile-base-v2" if n == "base" else "8-profile-fine")/"initial-normalization.csv") for n in ["base", "fine"]]
    assert np.array_equal(ia["s_m"], ib["s_m"])
    return dict(cases=cases, common_points=regions,
                nodal_I=metric(ia["s_m"], ia["I_h_m"], ib["I_h_m"], float(max(ib["I_h_m"]))))


def assess():
    evidence = ROOT/"benchmarks/reconstructed_fault/refactoring_boundary/evidence"
    runs = {p.stem: json.loads(p.read_text()) for p in evidence.glob("bp3-native-8-*.json")}
    loaded = {"filtered": profiles(RUNS/"7cd-bulk"), "raw": profiles(RUNS/"8-raw-v3")}
    checks, comparisons = {}, []
    for name, directory in {"filtered": RUNS/"7cd-bulk", "raw": RUNS/"8-raw-v3"}.items():
        journal, accepted = read(directory/"coupled-checks.csv"), read(directory/"accepted_steps.csv")
        assert journal["time_s"].tolist() == [i*200000 for i in range(11)]
        assert np.all(journal["publications"] == 1) and np.all(accepted["fresh_linear_checks_passed"] == 1)
        assert max(accepted["normalized_nonlinear_residual"]) < 1e-8
        p = loaded[name][2000000][1]
        checks[name] = dict(max_stress_defect_Pa=float(max(journal["stress_error_Pa"])),
                            max_allowance_fraction=float(max(journal["error_fraction_of_allowance"])),
                            max_residual=float(max(accepted["normalized_nonlinear_residual"])),
                            max_surface_RMS_Pa=float(max(accepted["surface_RMS_Pa"])),
                            max_theta_relative_error=float(max(accepted["Theta_relative_error"])),
                            Newton_updates=accepted["newton_updates"].tolist(),
                            Krylov_iterations=accepted["krylov_iterations"].tolist(),
                            normal=normal_diagnostics(p),
                            roughness={f: oscillations(p["s_m"], p[f], SCALES.get(f, max(rms(p["s_m"], p[f]), 1.)))
                                       for f in ["V_m_s", "raw_normal_Pa", "filtered_normal_Pa", "pressure_Pa", "deviatoric_Pa"]})
    identities = {}
    for f in ["initial-mesh.csv", "initial-phase.csv", "initial-particle-identity.csv", "mechanical-quadrature.csv"]:
        a, b = [(RUNS/n/f).read_bytes() for n in ["7cd-bulk", "8-raw-v3"]]
        assert a == b, f
        identities[f] = hashlib.sha256(a).hexdigest()
    for time in sorted(loaded["filtered"]):
        a, b = loaded["filtered"][time][1], loaded["raw"][time][1]
        s = a["s_m"]
        assert np.array_equal(s, b["s_m"])
        fields = {}
        for f in FIELDS+["I_h_m", "Z_Pa_m"]:
            scale = SCALES.get(f, max(rms(s, a[f]), rms(s, b[f]), 1.))
            if f in {"pressure_Pa", "deviatoric_Pa"}:
                scale = max(float(max(abs(a[f]))), float(max(abs(b[f]))), 1.)
            fields[f] = metric(s, a[f], b[f], scale)
            if f in {"raw_normal_Pa", "filtered_normal_Pa", "shear_Pa"}:
                bg = 50e6 if "normal" in f else SCALES["shear_Pa"]
                pert = max(rms(s, a[f]-bg), rms(s, b[f]-bg), 1.)
                fields[f].update(perturbation_scale=pert, rms_relative_to_perturbation=fields[f]["rms_abs"]/pert)
        assert fields["I_h_m"]["max_abs"] == 0.
        comparisons.append(dict(time_s=time, fields=fields))
    report = dict(checks=checks, differences_raw_minus_filtered=comparisons, identical_initial_data=identities,
                  runs=runs, simulation_seconds=sum(v["seconds"] for v in runs.values()))
    if (RUNS/"8-profile-fine/initial-profile-summary.csv").exists():
        report["profile_refinement"] = profile_check()
    directory = RUNS/"8-full-startup"
    if (directory/"coupled-checks.csv").exists():
        j, a = read(directory/"coupled-checks.csv"), read(directory/"accepted_steps.csv")
        assert np.all(j["publications"] == 1) and np.all(a["fresh_linear_checks_passed"] == 1)
        assert max(a["normalized_nonlinear_residual"]) < 1e-8
        h = profiles(directory)
        initial = h[0][1]
        fraction = np.clip((initial["s_m"]-15000.)/3000., 0., 1.)
        direct = .010+.015*fraction
        expected = .008/1e-6*np.exp(direct/.015*np.log(2e-6/1e-9*np.sinh(
            (SCALES["shear_Pa"]-4624440e-9)/(direct*50e6)))-.6/.015)
        theta_error = float(max(abs(initial["theta_s"]/expected-1)))
        assert theta_error < 1e-12 and np.all(initial["Z_Pa_m"] == 0) and np.all(initial["slip_m"] == 0)
        state_predictions = []
        times = sorted(h)
        for i in range(1, len(times)):
            before = h[times[i-1]][1]
            dt = float(h[times[i]][1]["time_s"][0]-before["time_s"][0])
            V, theta = before["V_m_s"], before["theta_s"]
            x = V*dt/.008
            predicted = theta*np.exp(-x)-.008/V*np.expm1(-x)
            change = float(max(abs(np.log(predicted/theta))))
            assert abs(change-.2) < 1e-12, "Ordinary BP3 startup no longer selects its state bound."
            state_predictions.append(dict(dt_s=dt, max_predicted_log_change=change))
        columns, statistics = {}, []
        for line in (directory/"statistics").read_text().splitlines():
            if line.startswith("# "):
                number, label = line[2:].split(": ", 1)
                columns[label] = int(number)-1
            elif line.strip():
                statistics.append([float(v) for v in line.split()])
        report["coarse_full_startup"] = dict(steps=j["step"].tolist(), times=j["time_s"].tolist(),
            dt=j["dt_s"].tolist(), min_I=float(min(j["min_I_m"])),
            max_stress_defect_Pa=float(max(j["stress_error_Pa"])), max_allowance_fraction=float(max(j["error_fraction_of_allowance"])),
            max_residual=float(max(a["normalized_nonlinear_residual"])), max_theta_error=float(max(a["Theta_relative_error"])),
            initial_theta_relative_error=theta_error, state_predictor=state_predictions,
            forced_local_dt_max_predicted_log_change=float(max(abs(np.log((initial["theta_s"]*np.exp(-initial["V_m_s"]*200000/.008)
                -.008/initial["V_m_s"]*np.expm1(-initial["V_m_s"]*200000/.008))/initial["theta_s"])))),
            publications=j["publications"].tolist(), Newton_updates=a["newton_updates"].tolist(),
            Krylov_iterations=a["krylov_iterations"].tolist(),
            peak_virtual_memory_MiB=max(row[columns["Peak virtual memory usage (VmPeak) (MB)"]] for row in statistics),
            system_matrix_MiB=max(row[columns["System matrix memory consumption (MB)"]] for row in statistics),
            surface_vertices=len(initial["s_m"]), actual_mean_surface_spacing_m=float(np.mean(abs(np.diff(initial["s_m"])))) )
    (OUTPUT/"comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    with (OUTPUT/"common-time-differences.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "field", "max_abs", "rms_abs", "scale", "max_relative", "rms_relative"])
        for entry in comparisons:
            for f, data in entry["fields"].items():
                writer.writerow([entry["time_s"], f, *[data[k] for k in ["max_abs", "rms_abs", "scale", "max_relative", "rms_relative"]]])
    plot(loaded)
    print(json.dumps({k: v for k, v in report.items() if k not in {"differences_raw_minus_filtered", "runs"}}, indent=2))


def plot(loaded):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/bp3-section8-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    fields = FIELDS+["Z_Pa_m"]
    labels = dict(V_m_s="V / Vp", theta_s="theta / (8e6 s)", Omega="Omega", slip_m="slip / (Vp T)",
                  shear_Pa="shear [kPa]", raw_normal_Pa="raw normal [kPa]", filtered_normal_Pa="effective normal [kPa]",
                  pressure_Pa="pressure [kPa]", deviatoric_Pa="tensor normal [kPa]", Z_Pa_m="Z [Pa m]")
    with PdfPages(OUTPUT/"profiles.pdf") as pdf:
        for time in [0, 1000000, 2000000]:
            fig, axes = plt.subplots(5, 2, figsize=(10, 13))
            for name, history in loaded.items():
                p = history[time][1]
                for ax, f in zip(axes.flat, fields):
                    bg = 50e6 if "normal_Pa" in f else (SCALES["shear_Pa"] if f == "shear_Pa" else 0.)
                    scale = 1000. if f.endswith("_Pa") else SCALES.get(f, 1.)
                    ax.plot(p["s_m"], (p[f]-bg)/scale, label=name, linewidth=1.)
                    ax.set_ylabel(labels[f])
                    ax.grid(alpha=.2)
            axes[0, 0].legend()
            fig.suptitle(f"Section 8 native profiles, t={time:g} s; stationary FE\nShear/normal backgrounds subtracted")
            fig.tight_layout(rect=(0, 0, 1, .97))
            pdf.savefig(fig)
            plt.close(fig)
        fig, axes = plt.subplots(5, 2, figsize=(10, 12))
        for name, history in loaded.items():
            p = history[2000000][1]
            for row, f in enumerate(["V_m_s", "raw_normal_Pa", "filtered_normal_Pa", "pressure_Pa", "deviatoric_Pa"]):
                scale = 1e-9 if f == "V_m_s" else 1000.
                bg = 50e6 if "normal_Pa" in f else 0.
                for column, x in enumerate([p["s_m"], p["s_m"][-1]-p["s_m"]]):
                    axes[row, column].plot(x, (p[f]-bg)/scale, ".-", label=name, markersize=3, linewidth=1.)
                    axes[row, column].set_xlim(0., 200.)
                    axes[row, column].set_ylabel(labels[f])
                    axes[row, column].grid(alpha=.2)
        axes[0, 0].legend()
        axes[-1, 0].set_xlabel("distance from upper endpoint [m]")
        axes[-1, 1].set_xlabel("distance from lower endpoint [m]")
        fig.suptitle("Final endpoint response; dots are actual surface nodes\nNormal background subtracted")
        fig.tight_layout(rect=(0, 0, 1, .97))
        pdf.savefig(fig)
        plt.close(fig)


if __name__ == "__main__":
    assess()
