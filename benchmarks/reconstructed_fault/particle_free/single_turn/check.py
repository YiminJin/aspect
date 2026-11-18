#!/usr/bin/env python3
"""Run the bounded constitutive comparison and write compact reproducible evidence."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from reference import (Parameters, Chart, History, initial, aging, advance,
                       history_run, shear_tensor, rotation, select_virgin_plane,
                       resistance, energy, composite_gauss)


def norm(x, weights):
    while weights.ndim < x.ndim:
        weights = weights[..., None]
    return float(np.sqrt(np.sum(weights*x*x)))


def difference(a, b, chart):
    return dict(stress_Pa_m=norm(a.stress-b.stress, chart.area),
                cohesive_strain_m=norm(a.A-b.A, chart.area),
                age_s_sqrt_m=norm(a.age-b.age, chart.ds),
                slip_m_sqrt_m=norm(a.slip-b.slip, chart.ds))


def write_csv(path, rows):
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    p = Parameters()
    checks, trajectories, states, charts, all_samples = {}, {}, {}, {}, []
    # No large/native trajectory: six independent fixed-phase quadrature paths.
    for geometry, curvature, angle in (("straight", 0., 0.), ("rotated", 0., .37), ("curved", .8, 0.)):
        for law in ("material_area", "normal_integral"):
            name = geometry+"_"+law
            chart = Chart.create(curvature, angle, law)
            state, rows, samples = history_run(chart, rotation_angle=angle)
            states[name], trajectories[name], charts[name] = state, rows, chart
            all_samples.extend((chart, *sample) for sample in samples)
            write_csv(out/(name+".csv"), rows)
    straight = charts["straight_material_area"]
    checks["straight_laws"] = difference(states["straight_material_area"], states["straight_normal_integral"], straight)
    assert max(checks["straight_laws"].values()) == 0.
    R = rotation(.37)
    a = states["straight_material_area"]
    expected = History(R@a.stress@R.T, R@a.A@R.T, a.age, a.slip)
    checks["rotated_covariance"] = difference(expected, states["rotated_material_area"], straight)
    assert checks["rotated_covariance"]["stress_Pa_m"] < 1e-10
    assert checks["rotated_covariance"]["age_s_sqrt_m"] < 1e-12
    curved = charts["curved_material_area"]
    checks["curved_laws"] = difference(states["curved_material_area"], states["curved_normal_integral"], curved)
    for law in ("material_area", "normal_integral"):
        name = "curved_"+law
        fine, fine_rows, _ = history_run(charts[name], dt=.000625)
        checks[name+"_time_halving"] = difference(states[name], fine, curved)
        states[name+"_fine"] = fine
        write_csv(out/(name+"_fine.csv"), fine_rows)
    checks["curved_laws_fine"] = difference(states["curved_material_area_fine"], states["curved_normal_integral_fine"], curved)
    assert checks["curved_laws"]["age_s_sqrt_m"] > 0.
    fine = states["curved_material_area_fine"]
    checks["curved_law_relative_difference"] = dict(
        stress=checks["curved_laws_fine"]["stress_Pa_m"]/norm(fine.stress, curved.area),
        age=checks["curved_laws_fine"]["age_s_sqrt_m"]/norm(fine.age, curved.ds),
        added_slip=checks["curved_laws_fine"]["slip_m_sqrt_m"]/norm(fine.slip-initial(curved).slip, curved.ds))

    # Actual Cartesian Q6 JxW, overlapping section interpolation and its
    # transpose. This exercises nonaligned membership, unlike a fitted tube.
    x, wx = composite_gauss(-.4, .4, 4)
    y, wy = composite_gauss(-.125, .125, 4)
    X, Y = np.meshgrid(x, y, indexing="ij")
    weights = np.outer(wx, wy).ravel()
    labels = np.linspace(-.5, .5, 9)
    coordinate = (X+.15*Y).ravel()
    interpolation = np.maximum(0., 1-abs(labels[:, None]-coordinate)/.125)
    assert np.max(abs(np.sum(interpolation, axis=0)-1)) < 1e-14
    ds = np.full(len(labels), .125)
    L = interpolation*weights/ds[:, None]
    # Wb^-1 L^T Ws, not the unweighted transpose.
    adjoint = L.T*ds/weights[:, None]
    w = .02*np.sin(9*X.ravel())+.01*Y.ravel()
    f = np.linspace(-120., 130., len(labels))
    lhs, rhs = np.dot(weights*w, adjoint@f), np.dot(ds*f, L@w)
    checks["native_cartesian_overlap_power_error"] = float(lhs-rhs)
    checks["wrong_unweighted_transpose_power_error"] = float(np.dot(weights*w, L.T@f)-rhs)
    assert abs(lhs-rhs) < 1e-14
    assert abs(checks["wrong_unweighted_transpose_power_error"]) > 1e-4
    mobility = .2+.1*np.cos(X.ravel())
    condensed = ds[:, None]*((L*mobility)@adjoint)
    checks["condensed_symmetry_error"] = float(np.max(abs(condensed-condensed.T)))
    checks["condensed_minimum_eigenvalue"] = float(np.min(np.linalg.eigvalsh(condensed)))
    assert checks["condensed_symmetry_error"] < 1e-14
    assert checks["condensed_minimum_eigenvalue"] > -1e-14

    # Reflections change tensor components, not material age. Build the reflected
    # physical chart explicitly; do not regenerate a different initial state.
    chart = curved
    parent = initial(chart)
    D = .04*shear_tensor(0.)
    original_parent = parent.copy()
    candidate, obs = advance(parent, chart, D, .00125)
    mirror = np.diag([1., -1., 1.])
    reflected = copy.deepcopy(chart)
    reflected.S = mirror@chart.S@mirror.T
    reflected_parent = History(mirror@parent.stress@mirror.T, mirror@parent.A@mirror.T,
                               parent.age.copy(), parent.slip.copy())
    reflected_next, _ = advance(reflected_parent, reflected, mirror@D@mirror.T, .00125)
    mirror_expected = History(mirror@candidate.stress@mirror.T, mirror@candidate.A@mirror.T,
                              candidate.age, candidate.slip)
    checks["reflected_tensor_chart"] = difference(mirror_expected, reflected_next, chart)
    assert max(checks["reflected_tensor_chart"].values()) < 1e-10

    # Reverse one frame vector: S,w,V,f,slip reverse; tau,A,age stay physical.
    signed = copy.deepcopy(chart)
    signed.S *= -1
    signed_parent = parent.copy()
    signed_parent.slip *= -1
    signed_next, signed_obs = advance(signed_parent, signed, D, .00125)
    sign_expected = History(candidate.stress, candidate.A, candidate.age, -candidate.slip)
    checks["signed_frame"] = difference(sign_expected, signed_next, chart)
    checks["signed_frame"]["V_error"] = float(np.max(abs(obs["V"]+signed_obs["V"])))
    assert max(checks["signed_frame"].values()) < 1e-10

    # Pure record renumbering, fixed physical points and weights, is invariant.
    permutation = np.arange(len(chart.s))[::-1]
    renamed = copy.deepcopy(chart)
    for field in ("s", "ds", "angle", "J", "S", "area", "g", "phi"):
        setattr(renamed, field, getattr(renamed, field)[permutation].copy())
    reordered = History(*(v[permutation].copy() for v in (parent.stress, parent.A, parent.age, parent.slip)))
    renamed_next, _ = advance(reordered, renamed, D, .00125)
    renamed_expected = History(*(v[permutation] for v in (candidate.stress, candidate.A, candidate.age, candidate.slip)))
    checks["record_permutation"] = difference(renamed_expected, renamed_next, renamed)
    assert max(checks["record_permutation"].values()) == 0.

    # Discard a candidate, serialize accepted reference state, retry: never MPI
    # or native checkpoint qualification, and never a second simulator pipeline.
    assert all(np.array_equal(getattr(parent, k), getattr(original_parent, k)) for k in vars(parent))
    np.savez(out/"reference-parent.npz", **vars(parent))
    with np.load(out/"reference-parent.npz") as saved:
        restored = History(**{k: saved[k].copy() for k in vars(parent)})
    retry, _ = advance(restored, chart, D, .00125)
    checks["reference_discard_restore_retry_bitwise"] = all(np.array_equal(getattr(candidate, k), getattr(retry, k)) for k in vars(parent))
    assert checks["reference_discard_restore_retry_bitwise"]

    # Prospective material has age from t=0 even before any assigned plane.
    intact = copy.deepcopy(chart)
    intact.phi[:] = 0.
    virgin = parent.copy()
    virgin.A[:] = 0.
    virgin.slip[:] = 0.
    virgin.age[:] = .5
    for _ in range(8):
        virgin, response = advance(virgin, intact, D, .01)
        assert np.max(abs(response["w"])) == 0.
    checks["prospective_age_error"] = float(np.max(abs(virgin.age-.58)))
    assert checks["prospective_age_error"] < 1e-14

    # Native source oracle at actual points from each geometry/law/history.
    inputs, expected = [], []
    for chart, old, D, dt, response in all_samples:
        endpoint_tau = 2*response["stage_stress"]-old.stress
        endpoint_A = 2*response["stage_A"]-old.A
        for i in (0, 11, 23):
            for j in (0, 12, 22, 35, 47):
                tau = old.stress[i, j]
                old_e = 2*np.sum(old.A[i, j]*chart.S[i, j])
                inputs.append([p.G, p.eta, p.m, chart.phi[i, j], dt, chart.angle[i], old_e,
                               tau[0, 0], tau[1, 1], tau[0, 1], tau[2, 2],
                               dt*D[0, 0], dt*D[1, 1], dt*D[0, 1], dt*D[2, 2], response["force"][i, j]])
                tau = endpoint_tau[i, j]
                expected.append([tau[0, 0], tau[1, 1], tau[0, 1], tau[2, 2],
                                 2*np.sum(endpoint_A[i, j]*chart.S[i, j]), response["w"][i, j],
                                 response["Y"][i, j], response["eb"][i, j], response["ec"][i, j], 0.])
    np.savetxt(out/"oracle-input.txt", inputs, fmt="%.17g")
    result = subprocess.run([str(args.oracle.resolve())], input=(out/"oracle-input.txt").read_text(),
                            capture_output=True, text=True, check=True)
    (out/"oracle-output.txt").write_text(result.stdout)
    (out/"oracle-stderr.txt").write_text(result.stderr)
    actual = np.loadtxt(out/"oracle-output.txt")
    error = np.max(abs(actual-np.asarray(expected)), axis=0)
    checks["native_oracle_max_absolute_errors"] = error.tolist()
    checks["native_oracle_rows"] = len(inputs)
    assert max(error[:4]) < 1e-9 and max(error[4:6]) < 1e-12
    assert max(error[6:9]) < 1e-10 and error[9] < 1e-11

    # Point angle selection is not phase-front selection. Test both conjugates
    # against an independent angular scan, including reflected off-axis loads.
    selections = []
    for angle in (0., np.deg2rad(12), -np.deg2rad(12)):
        tau = 1000*2*shear_tensor(angle)
        candidates, chosen = select_virgin_plane(tau, 0.)
        grid = np.linspace(-np.pi/2, np.pi/2, 36001)
        force = np.sum(shear_tensor(grid)*tau, axis=(-2, -1))
        Y = p.m*force**2/(2*p.G)
        chosen_Y = p.m*np.sum(shear_tensor(chosen)*tau)**2/(2*p.G)
        assert abs(chosen-angle) < 1e-14 and abs(chosen_Y-np.max(Y)) < 1e-10
        selections.append(dict(load_angle_deg=float(np.rad2deg(angle)),
                               selected_plane_deg=float(np.rad2deg(chosen)),
                               conjugate_deg=float(np.rad2deg(candidates[1])),
                               intact_Y_Pa=float(chosen_Y)))
    assert select_virgin_plane(1000*2*shear_tensor(0.), np.pi/4)[1] is None
    assert select_virgin_plane(np.zeros((3, 3)), 0.)[1] is None
    write_csv(out/"virgin-selection.csv", selections)

    # Work equality alone cannot license changing material membership or
    # projecting cohesive memory onto a new tangent.
    q = 2*np.sum(parent.A*shear_tensor(.3), axis=(-2, -1))
    projected_A = q[..., None, None]*shear_tensor(.3)
    old_ec = energy(parent.stress, parent.A, chart.phi, p)[1]
    projected_ec = energy(parent.stress, projected_A, chart.phi, p)[1]
    checks["forbidden_projection_energy_change_J_per_m"] = float(np.sum(chart.area*(projected_ec-old_ec)))
    cohesive = p.m*chart.phi*(1+chart.phi)/(1-chart.phi)**2
    e = 2*np.sum(parent.A*chart.S, axis=(-2, -1))
    moment = np.sum(chart.dn*chart.J*e, axis=1)
    integrated_h = np.sum(chart.dn*chart.J*cohesive, axis=1)
    collapsed_energy = np.sum(chart.ds*p.G*moment**2/(2*integrated_h))
    checks["bulk_cohesive_energy_J_per_m"] = float(np.sum(chart.area*old_ec))
    checks["variance_energy_above_section_moment_J_per_m"] = float(np.sum(chart.area*old_ec)-collapsed_energy)
    assert checks["variance_energy_above_section_moment_J_per_m"] > 0.
    # A material label is not a physical arclength. Under xi=1.4*s its
    # quadrature weight changes, while the arclength metric compensates.
    checks["coordinate_relabelling_power_error"] = float(np.sum((1.4*chart.ds)/1.4*obs["f"]*obs["V"])-np.sum(chart.ds*obs["f"]*obs["V"]))
    f0 = resistance(.001, .2, .000625, p)
    f1 = resistance(.001, .8, .000625, p)
    fm = resistance(.001, .5, .000625, p)
    checks["mixed_age_force_defect_Pa"] = float(fm-.5*(f0+f1))

    for name, rows in trajectories.items():
        assert max(abs(r["error"]) for r in rows) < 1e-11
        assert max(abs(r["power_error"]) for r in rows) < 1e-12
        assert max(r["rate_error"] for r in rows) < 1e-12
        assert min(r["mean_V"] for r in rows) < 0. < max(r["mean_V"] for r in rows)
        assert min(r["fault_power"] for r in rows) >= -1e-14
    checks["maximum_power_error_W_per_m"] = max(abs(r["power_error"]) for rows in trajectories.values() for r in rows)
    checks["maximum_step_energy_error_J_per_m"] = max(abs(r["error"]) for rows in trajectories.values() for r in rows)
    checks["minimum_local_negative_power_W_per_m"] = min(r["negative_power"] for rows in trajectories.values() for r in rows)
    checks["maximum_opposite_sign_sections_of_24"] = max(r["opposite_sections"] for rows in trajectories.values() for r in rows)
    checks["maximum_active_components_per_section"] = max(r["max_active_components"] for rows in trajectories.values() for r in rows)
    checks["summed_absolute_energy_error_J_per_m"] = {k: sum(abs(r["error"]) for r in rows) for k, rows in trajectories.items()}
    checks["signed_cumulative_energy_error_J_per_m"] = {k: sum(r["error"] for r in rows) for k, rows in trajectories.items()}
    checks["seconds"] = time.monotonic()-start
    checks["oracle_sha256"] = hashlib.file_digest(args.oracle.open("rb"), "sha256").hexdigest()
    checks["parameters"] = vars(p)
    checks["scope"] = "Fixed prescribed charts/phase, imposed affine strain, coupled section roots; no momentum or phase solve."
    checks["all_checks_passed"] = True
    (out/"checks.json").write_text(json.dumps(checks, indent=2)+"\n")
    rows = []
    for name, state in states.items():
        chart = charts[name.removesuffix("_fine")]
        for i, s in enumerate(chart.s):
            rows.append(dict(case=name, s_m=s, angle_deg=np.rad2deg(chart.angle[i]),
                             age_s=state.age[i], signed_slip_m=state.slip[i],
                             initial_age_s=initial(chart).age[i]))
    write_csv(out/"endpoint-sections.csv", rows)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
