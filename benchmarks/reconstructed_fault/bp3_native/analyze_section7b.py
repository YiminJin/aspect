#!/usr/bin/env python3
"""Compare only the fresh serial 7B histories on common physical times."""
import csv
import hashlib
import json
import pathlib

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "build-bp3-native/section7b"
RUNS = ROOT / "build-bp3-native/runs"
FIELDS = ["V_m_s", "theta_s", "Omega", "slip_m", "shear_Pa", "raw_normal_Pa",
          "filtered_normal_Pa", "pressure_Pa", "deviatoric_Pa"]
SCALES = dict(V_m_s=1e-9, theta_s=8e6, Omega=1., slip_m=.002,
              shear_Pa=50e6*.025*np.arcsinh(1e-9/(2e-6)*np.exp((.6+.015*np.log(1e-6/1e-9))/.025))+4624440e-9, raw_normal_Pa=50e6,
              filtered_normal_Pa=50e6)


def read(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    return {name: np.array([float(row[name]) for row in rows])
            for name in rows[0] if name not in {"cell"}}


def rms(s, values):
    lengths = np.diff(s)
    return float(np.sqrt(np.sum(lengths * (values[:-1]**2
                                           + values[:-1]*values[1:]
                                           + values[1:]**2)) / (3*(s[-1]-s[0]))))


def oscillations(s, values, scale):
    left, right = np.diff(s)[:-1], np.diff(s)[1:]
    chord = values[1:-1] - (right*values[:-2]+left*values[2:])/(left+right)
    result = {}
    for name, selected in {"upper_200m": s[1:-1] <= 200,
                           "interior": (s[1:-1] > 200) & (s[1:-1] < s[-1]-200),
                           "lower_200m": s[1:-1] >= s[-1]-200}.items():
        result[name] = float(np.max(np.abs(chord[selected])))
    # Observational short-wave spectral content, not a numerical-error estimate.
    residual = values - np.polyval(np.polyfit(s, values, 1), s)
    frequency = np.fft.rfftfreq(len(s), d=float(np.mean(np.diff(s))))
    amplitude = 2*np.abs(np.fft.rfft(residual*np.hanning(len(s))))/np.sum(np.hanning(len(s)))
    short = frequency >= 1/200.
    index = np.flatnonzero(short)[np.argmax(amplitude[short])]
    result["short_wave_amplitude"] = float(amplitude[index])
    result["short_wave_wavelength_m"] = (float(1/frequency[index])
                                           if amplitude[index] > 1e-12*scale else None)
    result["minimum_resolvable_wavelength_m"] = float(2*np.mean(np.diff(s)))
    return result


def cut_jumps(directory, step):
    with (directory/f"bulk-normal-cuts-{step}.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    # Exact cell entry/exit coordinates can differ by a few ulps. The grouping
    # precision is far below the two-metre observation spacing or mesh scale.
    groups = {}
    for row in rows:
        key = (int(row["station"]), round(float(row["r_m"]), 8))
        groups.setdefault(key, []).append(row)
    maxima = {name: 0. for name in ["pressure_Pa", "deviatoric_Pa", "normal_Pa"]}
    interfaces = []
    for (station, radius), group in groups.items():
        if len({float(row["edge_m"]) for row in group}) < 2:
            continue
        interfaces.append([station, radius, sorted({float(row["edge_m"]) for row in group})])
        for name in maxima:
            maxima[name] = max(maxima[name], max(float(row[name]) for row in group)
                               - min(float(row[name]) for row in group))
    return dict(graded_interfaces=len(interfaces), maximum_jumps=maxima,
                interface_positions=interfaces)


def assess():
    comparisons, checks, identities, roughness, cuts = {}, {}, {}, {}, {}
    for variant in ["creep", "perturbed"]:
        pair = {backend: RUNS/f"7b-{backend}-{variant}" for backend in ["fe", "particle"]}
        resolved = {}
        for backend, directory in pair.items():
            resolved[backend] = [line for line in (directory/"parameters.prm").read_text().splitlines()
                                 if not line.strip().startswith(("set Output directory ",
                                                                 "set Bulk history representation "))]
        assert resolved["fe"] == resolved["particle"], variant
        identities[variant] = {}
        for name in ["initial-mesh.csv", "initial-phase.csv", "mechanical-quadrature.csv",
                     "initial-particle-identity.csv"]:
            hashes = {key: hashlib.sha256((path/name).read_bytes()).hexdigest()
                      for key, path in pair.items()}
            assert len(set(hashes.values())) == 1, (variant, name, hashes)
            identities[variant][name] = hashes["fe"]
        roughness[variant], cuts[variant], checks[variant] = {}, {}, {}
        for backend, path in pair.items():
            journal = read(path/"coupled-checks.csv")
            accepted = read(path/"accepted_steps.csv")
            assert journal["step"].tolist() == list(range(11))
            assert accepted["step"].tolist() == list(range(11))
            assert journal["time_s"].tolist() == [200000*i for i in range(11)]
            assert np.all(journal["publications"] == 1)
            checks[variant][backend] = {
                "positive_steps": 10,
                "maximum_stress_error_Pa": float(np.max(journal["stress_error_Pa"])),
                "maximum_stress_error_fraction_of_allowance": float(np.max(journal["error_fraction_of_allowance"])),
                "maximum_sample_stress_scale_Pa": float(np.max(journal["stress_scale_Pa"])),
                "maximum_instant_tensor_normal_difference_Pa": float(np.max(journal["instant_tensor_normal_difference_Pa"])),
                "maximum_sample_divergence_per_s": float(np.max(journal["max_divergence_per_s"])),
                "maximum_normal_split_error_Pa": float(np.max(journal["normal_split_error_Pa"])),
                "minimum_I_m": float(np.min(journal["min_I_m"])),
                "minimum_theta_s": float(np.min(journal["min_theta_s"])),
                "final_Z_max_Pa_m": float(journal["Z_max_Pa_m"][-1]),
                "maximum_normalized_residual": float(np.max(accepted["normalized_nonlinear_residual"])),
                "positive_newton_updates": accepted["newton_updates"][1:].tolist(),
                "positive_krylov_iterations": accepted["krylov_iterations"][1:].tolist(),
                "maximum_surface_RMS_Pa": float(np.max(accepted["surface_RMS_Pa"])),
                "maximum_Theta_relative_error": float(np.max(accepted["Theta_relative_error"])),
            }
            roughness[variant][backend] = {}
            cuts[variant][backend] = {}
            for step in [0, 10]:
                profile = read(path/f"coupled-profile-{step}.csv")
                roughness[variant][backend][str(step)] = {
                    field: oscillations(profile["s_m"], profile[field], SCALES[field])
                    for field in ["V_m_s", "raw_normal_Pa", "filtered_normal_Pa"]}
                cuts[variant][backend][str(step)] = cut_jumps(path, step)
        comparisons[variant] = []
        for step in range(11):
            profiles = {key: read(path/f"coupled-profile-{step}.csv") for key, path in pair.items()}
            a, b = profiles["fe"], profiles["particle"]
            assert np.array_equal(a["s_m"], b["s_m"])
            assert np.array_equal(a["I_h_m"], b["I_h_m"])
            if step == 0:
                assert np.array_equal(a["theta_s"], b["theta_s"])
                assert np.array_equal(a["background_Pa"], b["background_Pa"])
            s = a["s_m"]
            entry = dict(step=step, time_s=float(a["time_s"][0]), fields={})
            for field in FIELDS:
                error = a[field]-b[field]
                reference = max(rms(s,a[field]), rms(s,b[field]), 1.)
                scale = SCALES.get(field, reference)
                mean = float(np.sum(np.diff(s)*(error[:-1]+error[1:])/2)/(s[-1]-s[0]))
                values = dict(max_abs=float(np.max(np.abs(error))), rms_abs=rms(s,error),
                              scale=scale, rms_relative_to_scale=rms(s,error)/scale,
                              maximum_relative_to_scale=float(np.max(np.abs(error)))/scale,
                              signed_mean=mean, oscillatory_rms=rms(s,error-mean))
                if field in {"raw_normal_Pa", "filtered_normal_Pa", "shear_Pa"}:
                    background = 50e6 if "normal" in field else SCALES["shear_Pa"]
                    perturbation_scale = max(rms(s,a[field]-background), rms(s,b[field]-background), 1.)
                    values.update(perturbation_scale=perturbation_scale,
                                  rms_relative_to_perturbation=rms(s,error)/perturbation_scale)
                entry["fields"][field] = values
            comparisons[variant].append(entry)
    evidence = ROOT / "benchmarks/reconstructed_fault/refactoring_boundary/evidence"
    runs = {path.stem: json.loads(path.read_text())
            for path in sorted(evidence.glob("bp3-native-7b-*.json"))}
    for variant in ["creep", "perturbed"]:
        for backend in ["fe", "particle"]:
            result = runs[f"bp3-native-7b-{backend}-{variant}"]
            assert result["exit_code"] == 0
            checks[variant][backend]["wall_seconds"] = result["seconds"]
    report = dict(identity=identities, checks=checks, comparisons=comparisons,
                  oscillations=roughness, bulk_cut_jumps=cuts,
                  run_results=runs,
                  total_simulation_seconds=sum(run["seconds"] for run in runs.values()),
                  qualifications=["Fixed spatial rule and timestep only; no accuracy/refinement qualification.",
                                  "RMS norms use Q1 interpolation of exported nodal fields, including nodal Omega.",
                                  "Instant tensor shift is evaluated on each own velocity path, not a unique causal decomposition.",
                                  "Legacy tensor-load column is named deviatoric by the API but uses the retained full-gradient law."])
    (OUTPUT/"comparison.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    with (OUTPUT/"common-time-differences.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["variant","step","time_s","field","max_abs","rms_abs","scale","rms_relative_to_scale","signed_mean","oscillatory_rms"])
        for variant, entries in comparisons.items():
            for entry in entries:
                for field, values in entry["fields"].items():
                    writer.writerow([variant,entry["step"],entry["time_s"],field,
                                     *[values[name] for name in ["max_abs","rms_abs","scale","rms_relative_to_scale","signed_mean","oscillatory_rms"]]])
    print(json.dumps({variant: dict(checks=checks[variant], final=comparisons[variant][-1])
                      for variant in comparisons},indent=2))


if __name__ == "__main__":
    assess()
