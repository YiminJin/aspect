#!/usr/bin/env python3
"""Assess the bounded native managed-particle case; reuse BP3 profile metrics."""
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_section7b import FIELDS, read, rms, oscillations, cut_jumps
from analyze_section7cd import normal_diagnostics

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "build-bp3-native/managed-coupled"
EVIDENCE = ROOT / "benchmarks/reconstructed_fault/refactoring_boundary/evidence"


def check(directory):
    history = read(directory / "coupled-checks.csv")
    accepted = read(directory / "accepted_steps.csv")
    moving = read(directory / "moving-particles.csv")
    sampling = read(directory / "moving-sampling.csv")
    steps = history["step"]
    assert steps.tolist() == list(range(len(steps)))
    assert np.array_equal(steps, accepted["step"])
    assert np.array_equal(steps, moving["step"])
    assert np.array_equal(history["time_s"], moving["time_s"])
    assert np.all(history["publications"] == 1)
    assert np.all(moving["survivor_history_error"] == 0)
    assert np.all(accepted["fresh_linear_checks_passed"] == 1)
    assert np.max(accepted["normalized_nonlinear_residual"]) < 1e-8
    last = int(steps[-1])
    profiles = {step: read(directory / f"coupled-profile-{step}.csv")
                for step in (0, 1, last)}
    final = profiles[last]
    result = dict(positive_steps=last, endpoint_s=float(history["time_s"][-1]),
                  final_particles=int(moving["after"][-1]),
                  births=int(np.sum(moving["births"])), removals=int(np.sum(moving["removed"])),
                  outflow=int(np.sum(moving["outflow"])),
                  crossing_events=int(np.sum(moving["crossings"])),
                  first_crossing_step=int(steps[moving["crossings"] > 0][0]) if np.any(moving["crossings"]) else None,
                  first_birth_step=int(steps[moving["births"] > 0][0]) if np.any(moving["births"]) else None,
                  first_removal_step=int(steps[moving["removed"] > 0][0]) if np.any(moving["removed"]) else None,
                  min_per_cell=int(np.min(moving["min_per_cell"])), max_per_cell=int(np.max(moving["max_per_cell"])),
                  max_motion_per_step_m=float(np.max(moving["max_motion_m"])),
                  max_birth_stress_Pa=float(np.max(moving["max_birth_stress_Pa"])),
                  max_coverage_error=float(np.max(moving["coverage_relative"])),
                  max_partition_error=float(np.max(moving["partition_error"])),
                  max_linear_gradient_error=float(np.max(moving["linear_gradient_error"])),
                  max_stress_defect_Pa=float(np.max(history["stress_error_Pa"])),
                  max_stress_allowance_fraction=float(np.max(history["error_fraction_of_allowance"])),
                  max_nonlinear_residual=float(np.max(accepted["normalized_nonlinear_residual"])),
                  max_surface_RMS_Pa=float(np.max(accepted["surface_RMS_Pa"])),
                  max_theta_relative_defect=float(np.max(accepted["Theta_relative_error"])),
                  max_normal_split_defect_Pa=float(np.max(history["normal_split_error_Pa"])),
                  minimum_I_m=float(np.min(history["min_I_m"])),
                  minimum_theta_s=float(np.min(history["min_theta_s"])),
                  total_Newton_updates=int(np.sum(accepted["newton_updates"])),
                  total_Krylov_iterations=int(np.sum(accepted["krylov_iterations"])),
                  max_step_slip_over_Dc=float(np.max(accepted["max_step_slip_over_Dc"])),
                  multiple_trace_events=int(np.sum(sampling["multiple_traces"])),
                  max_host_sampling_difference_Pa=float(np.max(sampling["max_host_difference_Pa"])),
                  max_mean_host_sampling_difference_Pa=float(np.max(sampling["mean_host_difference_Pa"])),
                  final_ranges={name: [float(np.min(final[name])), float(np.max(final[name]))] for name in FIELDS},
                  final_roughness={name: oscillations(final["s_m"], final[name], 1e-9 if name == "V_m_s" else 50e6)
                                   for name in ("V_m_s", "raw_normal_Pa", "filtered_normal_Pa")},
                  final_normal_decomposition=normal_diagnostics(final),
                  final_cut_jumps=cut_jumps(directory, last))
    tangent = read(directory / "moving-tangent.csv")
    result["tangent"] = {name: values.tolist() for name, values in tangent.items()}
    return result


def compare(left, right):
    asteps, bsteps = read(left / "coupled-checks.csv"), read(right / "coupled-checks.csv")
    common = min(len(asteps["step"]), len(bsteps["step"]))
    assert np.array_equal(asteps["time_s"][:common], bsteps["time_s"][:common])
    assert np.array_equal(asteps["dt_s"][:common], bsteps["dt_s"][:common])
    for name in ("initial-mesh.csv", "initial-phase.csv", "mechanical-quadrature.csv"):
        assert (left/name).read_bytes() == (right/name).read_bytes(), name
    results = []
    for step in asteps["step"][:common].astype(int):
        a, b = [read(path / f"coupled-profile-{step}.csv") for path in (left, right)]
        assert np.array_equal(a["s_m"], b["s_m"])
        assert np.array_equal(a["I_h_m"], b["I_h_m"])
        assert np.array_equal(a["background_Pa"], b["background_Pa"])
        time = float(a["time_s"][0])
        scales = dict(V_m_s=1e-9, theta_s=8e6, Omega=1., slip_m=max(.008, time*1e-9))
        fields = {}
        for name in FIELDS:
            difference = a[name]-b[name]
            scale = scales.get(name, 50e6)
            fields[name] = dict(max_abs=float(np.max(np.abs(difference))), rms_abs=rms(a["s_m"], difference),
                                scale=scale, max_scaled=float(np.max(np.abs(difference)))/scale,
                                rms_scaled=rms(a["s_m"], difference)/scale)
        results.append(dict(step=int(step), time_s=time, fields=fields))
    return results


def plot(cases, comparisons):
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/bp3-managed-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    step, time = comparisons[-1]["step"], comparisons[-1]["time_s"]
    reference = read(cases["p4"]/"coupled-profile-0.csv")
    labels = ["V / Vp", "theta / theta0", "Omega", "slip / (Vp T)",
              "shear change [MPa]", "raw normal change [MPa]", "filtered normal change [MPa]",
              "pressure contribution [MPa]", "tensor contribution [MPa]"]
    with PdfPages(BASE/"managed-comparison.pdf") as pdf:
        fig, axes = plt.subplots(3, 3, figsize=(11, 8), sharex=True, constrained_layout=True)
        for (name, path), style in zip(cases.items(), [".-", ".--"]):
            data = read(path/f"coupled-profile-{step}.csv")
            for field, label, ax in zip(FIELDS, labels, axes.flat):
                scale = dict(V_m_s=1e-9, theta_s=8e6, Omega=1., slip_m=max(.008, time*1e-9)).get(field, 1e6)
                offset = reference[field] if field == "shear_Pa" else (50e6 if "normal_Pa" in field else 0.)
                ax.plot(data["s_m"], (data[field]-offset)/scale, style, label={"p4": "4x4 (12-24)", "p5": "5x5 (19-38)"}[name])
                ax.set_ylabel(label)
                ax.grid(alpha=.25)
                if field == "Omega": ax.set_ylim(.95, 1.05)
        for ax in axes[-1]: ax.set_xlabel("down dip [m]")
        axes[0, 0].legend()
        fig.suptitle(f"Managed particles: common step {step}, t={time:.9g} s; fixed 59-node surface")
        pdf.savefig(fig); plt.close(fig)
        fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, constrained_layout=True)
        times = [c["time_s"] for c in comparisons]
        for field, ax in zip(["V_m_s", "theta_s", "slip_m", "raw_normal_Pa", "filtered_normal_Pa", "pressure_Pa"], axes.flat):
            ax.plot(times, [100*c["fields"][field]["max_scaled"] for c in comparisons])
            ax.set_ylabel(field+"\nmaximum [% of scale]")
            ax.grid(alpha=.25)
        for ax in axes[-1]: ax.set_xlabel("physical time [s]")
        fig.suptitle("4x4 versus 5x5 particles/cell: observed sensitivity, not an error bound")
        pdf.savefig(fig); plt.close(fig)


def main():
    cases = {"p4": BASE/"runs/managed-v3", "p5": BASE/"runs/managed-p5"}
    report = dict(checks={name: check(path) for name, path in cases.items()},
                  density_comparison=compare(*cases.values()))
    report["density_maxima"] = {}
    for field in FIELDS:
        entry = max(report["density_comparison"], key=lambda c: c["fields"][field]["max_scaled"])
        report["density_maxima"][field] = dict(step=entry["step"], time_s=entry["time_s"], **entry["fields"][field])
    report["retained_stationary_step1_limit"] = {}
    current = read(cases["p4"]/"coupled-profile-1.csv")
    for backend in ["fe", "particle"]:
        old = read(ROOT/f"build-bp3-native/runs/7b-{backend}-creep/coupled-profile-1.csv")
        assert np.array_equal(current["s_m"], old["s_m"])
        assert np.array_equal(current["I_h_m"], old["I_h_m"])
        report["retained_stationary_step1_limit"][backend] = {
            field: float(np.max(np.abs(current[field]-old[field]))) for field in FIELDS}
    report["run_results"] = {p.stem: json.loads(p.read_text()) for p in sorted(EVIDENCE.glob("bp3-managed-*.json"))}
    report["simulation_seconds"] = sum(r["seconds"] for r in report["run_results"].values())
    report["common_comparison_endpoint_s"] = report["density_comparison"][-1]["time_s"]
    report["common_comparison_positive_steps"] = report["density_comparison"][-1]["step"]
    report["identities"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for directory in cases.values() for p in directory.iterdir()
                             if p.is_file() and p.suffix in (".prm", ".json")}
    report["limits"] = ["Serial smooth strengthening case; no MPI/restart/cutback qualification.",
                        "Particle-density sensitivity combines seeding, transfer and management; it does not isolate one operator.",
                        "Large diagnostic timesteps do not establish physical temporal accuracy.",
                        "Roughness measures are diagnostics, not independent error estimates.",
                        "No arbitrary subcell pulse deletion/conservation guarantee follows from smooth-case birth checks."]
    plot(cases, report["density_comparison"])
    (BASE/"assessment.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(checks=report["checks"], initial_comparison=report["density_comparison"][0],
                          final_comparison=report["density_comparison"][-1], simulation_seconds=report["simulation_seconds"]),
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
