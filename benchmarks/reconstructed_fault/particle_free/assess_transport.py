#!/usr/bin/env python3
"""Independent PF4 characteristic, DG weak-operator and convergence checks."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss


def table(path):
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")


def gauss(n):
    x, w = leggauss(n)
    return (x+1)/2, w/2


def shape(r, s, support):
    r, s = np.asarray(r), np.asarray(s)
    lr = np.array([(2*r-1)*(r-1), 4*r*(1-r), r*(2*r-1)]).T
    ls = np.array([(2*s-1)*(s-1), 4*s*(1-s), s*(2*s-1)]).T
    dr = np.array([4*r-3, 4-8*r, 4*r-1]).T
    ds = np.array([4*s-3, 4-8*s, 4*s-1]).T
    ri, si = np.rint(support["r"]*2).astype(int), np.rint(support["s"]*2).astype(int)
    return lr[:, ri]*ls[:, si], dr[:, ri]*ls[:, si], lr[:, ri]*ds[:, si]


def norm2(a):
    return a[..., 0]**2+a[..., 1]**2+(a[..., 0]+a[..., 1])**2+2*a[..., 2]**2


def exact(case, x, y, t, rate):
    out = np.zeros(x.shape+(3,))
    def basis(f):
        return f[..., None]*np.array([1000., -400., 700.])
    def pulse(a, b):
        return np.exp(-((a-.3)/.13)**2-((b-.5)/.16)**2)
    def chi(a):
        return np.exp(-((a-.5)/.12)**2)/(.12*np.sqrt(np.pi)*math.erf(.5/.12))
    if case.startswith("pulse_translation") or case.startswith("temporal_"):
        return basis(pulse(x-.4*t, y-.2*t))
    if case == "pulse_divergence":
        return basis(pulse((x+2/3)*np.exp(-.3*t)-2/3, .5+(y-.5)*np.exp(.1*t)))
    if case in ("constant_divergence", "density_negative", "rotation_nonobjective"):
        return basis(np.ones_like(x))
    if case == "frozen":
        return basis(np.full_like(x, np.exp(-rate*t)))
    if case == "signed_jump":
        xx = x-.4*t
        return basis(np.where((xx >= .25) & (xx < .5), 1., np.where((xx >= .5) & (xx < .75), -1., 0.)))
    if case.startswith("normal"):
        offset = .25*t if case in ("normal_mature", "normal_residual") else 0.
        out[..., 2] = -5000*np.exp(-rate*t)*chi(y-offset)
        return out
    if case.startswith("tangent"):
        speed = .35 if case in ("tangent_mature", "tangent_residual") else .1
        out[..., 2] = -5000*np.exp(-rate*t)*chi(y)*(1+.25*np.sin(2*np.pi*(x-speed*t)))
        return out
    if case == "comoving":
        out[..., 2] = -5000*np.exp(-rate*t)*chi(y-.25*t)*(1+.25*np.sin(2*np.pi*(x-.2*t)))
        return out
    if case == "rotation_objective_forced":
        out[..., 0] = 300+700*np.cos(2*t)-700*np.sin(2*t)
        out[..., 1] = 600-out[..., 0]
        out[..., 2] = 700*(np.sin(2*t)+np.cos(2*t))
        return out
    if case.startswith("rotation_profile_"):
        profile = chi(.5-np.sin(t)*(x-.5)+np.cos(t)*(y-.5))
        out[..., 2] = 1000*profile
        if case != "rotation_profile_nonobjective":
            out[..., 0] = -1000*profile*np.sin(2*t)
            out[..., 1] = -out[..., 0]
            out[..., 2] *= np.cos(2*t)
        return out
    raise ValueError(case)


class Entry:
    def __init__(self, directory):
        self.path = directory/"run"
        self.cells, self.basis = table(self.path/"cells.csv"), table(self.path/"basis.csv")
        self.constants = table(self.path/"constants.csv")
        self.nx, self.ny = int(self.constants["nx"]), int(self.constants["ny"])
        self.rate, self.time = float(self.constants["relaxation_rate"]), float(self.constants["time"])
        self.indices = np.stack([self.cells[f"i{i}"] for i in range(9)], axis=1)
        self.metrics = {str(r["case"]): r for r in np.atleast_1d(table(self.path/"metrics.csv"))}
        self.coefficients = {}
        self.cache = {}

    def coefficients_for(self, case):
        if case not in self.coefficients:
            data = table(self.path/f"field-{case}.csv")
            assert np.array_equal(data["i"], np.arange(len(data)))
            self.coefficients[case] = np.stack([data["xx"], data["yy"], data["xy"]], axis=1)
        return self.coefficients[case]

    def samples(self, order):
        if order not in self.cache:
            a, w = gauss(order)
            r, s = np.repeat(a, order), np.tile(a, order)
            weights = np.repeat(w, order)*np.tile(w, order)
            N, _, _ = shape(r, s, self.basis)
            x = self.cells["x0"][:, None]+self.cells["dx"][:, None]*r
            y = self.cells["y0"][:, None]+self.cells["dy"][:, None]*s
            Jw = self.cells["dx"][:, None]*self.cells["dy"][:, None]*weights
            self.cache[order] = N, x, y, Jw
        return self.cache[order]

    def measure(self, case, order=12):
        N, x, y, w = self.samples(order)
        coefficients = self.coefficients_for(case)[self.indices]
        value = np.einsum("qj,cjk->cqk", N, coefficients)
        reference = exact(case, x, y, self.time, self.rate)
        error = np.sqrt(np.sum(w*norm2(value-reference)))
        reference_norm = np.sqrt(np.sum(w*norm2(reference)))
        out = dict(error_L2=float(error), reference_L2=float(reference_norm),
                   relative_L2=float(error/reference_norm), norm_L2=float(np.sqrt(np.sum(w*norm2(value)))),
                   mean_xx=float(np.sum(w*value[..., 0])))
        if case == "density_negative":
            target = reference*np.exp(-.2*self.time)
            out["density_characteristic_relative_error"] = float(np.sqrt(np.sum(w*norm2(value-target)))/np.sqrt(np.sum(w*norm2(target))))
        return out

    def jump_resolved(self, order=24):
        # Split at the three exact translated jumps before error integration;
        # otherwise a fixed Gauss rule samples a discontinuous reference.
        p = self.coefficients_for("signed_jump")
        a, w = gauss(order)
        jumps = np.array([.25, .5, .75])+.4*self.time
        error2, error1, norm, mean = 0., 0., 0., 0.
        for c, cell in enumerate(self.cells):
            cuts = [0., *[(z-cell["x0"])/cell["dx"] for z in jumps if cell["x0"] < z < cell["x0"]+cell["dx"]], 1.]
            for lo, hi in zip(cuts[:-1], cuts[1:]):
                r, s = np.repeat(lo+(hi-lo)*a, len(a)), np.tile(a, len(a))
                N, _, _ = shape(r, s, self.basis)
                weights = cell["dx"]*cell["dy"]*(hi-lo)*np.repeat(w, len(a))*np.tile(w, len(a))
                x, y = cell["x0"]+cell["dx"]*r, cell["y0"]+cell["dy"]*s
                value = N @ p[self.indices[c]]
                reference = exact("signed_jump", x, y, self.time, self.rate)
                error2 += np.sum(weights*norm2(value-reference))
                error1 += np.sum(weights*np.sqrt(norm2(value-reference)))
                norm += np.sum(weights*norm2(value))
                mean += np.sum(weights*value[:, 0])
        row = self.metrics["signed_jump"]
        initial_squared_norm = 2.5e6*.5
        return dict(error_L2=float(np.sqrt(error2)), error_L1=float(error1),
                    relative_L2=float(np.sqrt(error2/initial_squared_norm)),
                    quadratic_norm_ratio=float(norm/initial_squared_norm), mean_xx=float(mean),
                    min_xx=float(row["min_xx"]), max_xx=float(row["max_xx"]),
                    overshoot=float(max(0., row["max_xx"]-1000)/1000),
                    undershoot=float(max(0., -1000-row["min_xx"])/1000))

    def difference(self, a, b):
        N, _, _, w = self.samples(3)
        d = self.coefficients_for(a)[self.indices]-self.coefficients_for(b)[self.indices]
        value = np.einsum("qj,cjk->cqk", N, d)
        return float(np.sqrt(np.sum(w*norm2(value))))


def operator_reference(entry):
    cells, support, indices = entry.cells, entry.basis, entry.indices
    n = int(np.max(indices))+1
    matrix, correction = np.zeros((n, n)), np.zeros((n, n))
    a, w = gauss(5)
    r, s = np.repeat(a, len(a)), np.tile(a, len(a))
    w2 = np.repeat(w, len(a))*np.tile(w, len(a))
    N, Nr, Ns = shape(r, s, support)
    lookup = {(round(float(c["x0"])*entry.nx), round(float(c["y0"])*entry.ny)): k for k, c in enumerate(cells)}
    boundary = np.zeros(n)
    polynomial_boundary = np.zeros(n)
    analytic_polynomial = np.zeros(n)
    for c, cell in enumerate(cells):
        x, y = cell["x0"]+cell["dx"]*r, cell["y0"]+cell["dy"]*s
        wx = w2*cell["dx"]*cell["dy"]
        vx, vy, div = .2+.1*x*x*y, -.15+.05*x*y*y, .3*x*y
        gradient_test = Nr/cell["dx"]*vx[:, None]+Ns/cell["dy"]*vy[:, None]
        mass = N.T @ (wx[:, None]*N)
        inverse = np.linalg.inv(mass)
        volume = gradient_test.T @ (wx[:, None]*N)+N.T @ ((wx*div)[:, None]*N)
        divergence = N.T @ ((wx*div)[:, None]*N)
        correction[np.ix_(indices[c], indices[c])] += inverse @ divergence
        analytic_polynomial[indices[c]] = inverse @ (N.T @ (wx*(-2*x*y*y*vx-2*x*x*y*vy)))
        for face in range(4):
            # Face ordering is physical, independently of deal.II's local map.
            if face < 2:
                fr, fs = np.full_like(a, face), a
                xx, yy = cell["x0"]+cell["dx"]*fr, cell["y0"]+cell["dy"]*fs
                vn = (-1 if face == 0 else 1)*(.2+.1*xx*xx*yy)
                fw = w*cell["dy"]
                key = (round(float(cell["x0"])*entry.nx)+(1 if face else -1), round(float(cell["y0"])*entry.ny))
            else:
                fr, fs = a, np.full_like(a, face-2)
                xx, yy = cell["x0"]+cell["dx"]*fr, cell["y0"]+cell["dy"]*fs
                vn = (-1 if face == 2 else 1)*(-.15+.05*xx*yy*yy)
                fw = w*cell["dx"]
                key = (round(float(cell["x0"])*entry.nx), round(float(cell["y0"])*entry.ny)+(1 if face == 3 else -1))
            F, _, _ = shape(fr, fs, support)
            outflow = np.maximum(vn, 0)
            volume -= F.T @ ((fw*outflow)[:, None]*F)
            inflow = np.minimum(vn, 0)
            neighbor = lookup.get(key)
            if neighbor is not None:
                other = cells[neighbor]
                FN, _, _ = shape((xx-other["x0"])/other["dx"], (yy-other["y0"])/other["dy"], support)
                matrix[np.ix_(indices[c], indices[neighbor])] -= inverse @ (F.T @ ((fw*inflow)[:, None]*FN))
            else:
                boundary[indices[c]] -= inverse @ (F.T @ (fw*inflow))
                polynomial_boundary[indices[c]] -= inverse @ (F.T @ (fw*inflow*xx*xx*yy*yy))
        matrix[np.ix_(indices[c], indices[c])] += inverse @ volume
    native = table(entry.path/"operator.csv")
    observed, observed_correction = np.zeros_like(matrix), np.zeros_like(matrix)
    observed[native["i"], native["j"]] = native["value"]
    observed_correction[native["i"], native["j"]] = native["divergence_correction"]
    loads = table(entry.path/"operator-loads.csv")
    inflow = loads[loads["kind"] == 1]
    observed_boundary, observed_poly_boundary = np.zeros(n), np.zeros(n)
    np.add.at(observed_boundary, inflow["i"], inflow["weight"])
    np.add.at(observed_poly_boundary, inflow["i"], inflow["weight"]*inflow["x"]**2*inflow["y"]**2)
    polynomial = np.zeros(n)
    for c, cell in enumerate(cells):
        polynomial[indices[c]] = (cell["x0"]+cell["dx"]*support["r"])**2*(cell["y0"]+cell["dy"]*support["s"])**2
    scale = lambda a, b: float(np.max(np.abs(a-b))/max(1., np.max(np.abs(b))))
    return dict(matrix_scaled_error=scale(observed, matrix), correction_scaled_error=scale(observed_correction, correction),
                boundary_scaled_error=scale(observed_boundary, boundary),
                constant_defect=float(np.max(np.abs(observed @ np.ones(n)+observed_boundary))),
                polynomial_defect=scale(observed @ polynomial+observed_poly_boundary, analytic_polynomial),
                uncorrected_constant_defect=float(np.max(np.abs((observed-observed_correction) @ np.ones(n)+observed_boundary))))


def assess(paths):
    entries = sorted([Entry(p) for p in paths], key=lambda e: e.nx)
    gates, rows, errors, discontinuities = {}, [], {}, []
    def gate(name, value, tolerance):
        gates[name] = dict(value=float(value), tolerance=tolerance, passed=bool(value <= tolerance))
    identities = [json.loads((p/"record.json").read_text()) for p in paths]
    assert len({r["executable_sha256"] for r in identities}) == 1
    assert len({r["plugin_sha256"] for r in identities}) == 1
    assert all(r["gate_passed"] for r in identities)
    for e in entries:
        assert float(np.loadtxt(e.path/"accepted.csv", skiprows=1)) == 1.
        for case, row in e.metrics.items():
            result = e.measure(case)
            native_rule = e.measure(case, order=8)
            gate(f"native_norm_reference_{e.nx}_{case}", abs(native_rule["norm_L2"]-row["norm_L2"])/max(1., native_rule["norm_L2"]), 1e-11)
            gate(f"native_error_reference_{e.nx}_{case}", abs(native_rule["error_L2"]-row["error_L2"])/max(1., native_rule["reference_L2"]), 1e-11)
            if case != "signed_jump":
                resolved_rule = e.measure(case, order=16)
                gate(f"characteristic_quadrature_{e.nx}_{case}", abs(result["error_L2"]-resolved_rule["error_L2"])/max(1., result["reference_L2"]), 1e-8)
            result.update(case=case, nx=e.nx, ny=e.ny, dt=float(row["dt"]), steps=int(row["steps"]))
            rows.append(result)
            errors.setdefault(case, []).append((e.nx, result["relative_L2"]))
            if case == "constant_divergence":
                gate(f"material_constant_{e.nx}", result["relative_L2"], 1e-11)
            if case == "density_negative":
                gates[f"density_negative_detected_{e.nx}"] = dict(value=result["relative_L2"], minimum=.03, passed=result["relative_L2"] > .03)
                gate(f"density_characteristic_{e.nx}", result["density_characteristic_relative_error"], 1e-6)
        # Independently inspect the initial L2 projection, separate from final
        # transport error. Its coefficient file is not a transported state.
        initial = table(e.path/"initial-pulse.csv")
        initial_coefficients = np.stack([initial["xx"], initial["yy"], initial["xy"]], axis=1)
        N, x, y, w = e.samples(16)
        projected = np.einsum("qj,cjk->cqk", N, initial_coefficients[e.indices])
        initial_exact = exact("pulse_translation", x, y, 0., e.rate)
        rows.append(dict(case="initial_projection", nx=e.nx, ny=e.ny,
                         error_L2=float(np.sqrt(np.sum(w*norm2(projected-initial_exact)))),
                         relative_L2=float(np.sqrt(np.sum(w*norm2(projected-initial_exact))/np.sum(w*norm2(initial_exact))))))
        jump = e.jump_resolved()
        jump_refined = e.jump_resolved(order=48)
        jump["L1_quadrature_difference"] = abs(jump_refined["error_L1"]-jump["error_L1"])
        gate(f"jump_L2_quadrature_{e.nx}", abs(jump_refined["error_L2"]-jump["error_L2"])/max(1., jump_refined["error_L2"]), 1e-11)
        jump.update(nx=e.nx, ny=e.ny);discontinuities.append(jump)
        gate(f"quadratic_norm_nonincrease_{e.nx}", jump["quadratic_norm_ratio"], 1+1e-11)
    spatial = {}
    for case in ["pulse_translation", "pulse_divergence", "normal_cohesive", "normal_mature", "normal_residual", "tangent_cohesive", "tangent_mature", "tangent_residual", "comoving", "rotation_profile_nonobjective", "rotation_profile_objective_forced"]:
        values = errors[case]
        orders = [math.log(a[1]/b[1], 2) for a, b in zip(values[:-1], values[1:])]
        spatial[case] = dict(relative_errors=values, orders=orders)
        gates[f"spatial_{case}"] = dict(last_order=orders[-1], minimum=2.5,
                                          passed=bool(orders[-1] >= 2.5 and all(b[1] < a[1] for a, b in zip(values[-3:-1], values[-2:]))))
    for case, threshold in [("normal_omitted", .3), ("tangent_omitted", .04)]:
        gates[f"commutator_negative_{case}"] = dict(values=errors[case], minimum=threshold, passed=errors[case][-1][1] > threshold)
    gates["objective_rotation_omission_detected"] = dict(values=errors["rotation_profile_omitted"], minimum=.3,
                                                        passed=errors["rotation_profile_omitted"][-1][1] > .3)
    small = next(e for e in entries if e.nx == 4)
    operator = operator_reference(small)
    for name in ["matrix_scaled_error", "correction_scaled_error", "boundary_scaled_error", "constant_defect", "polynomial_defect"]:
        gate("operator_"+name, operator[name], 1e-11)
    gates["operator_density_negative"] = dict(value=operator["uncorrected_constant_defect"], minimum=.1, passed=operator["uncorrected_constant_defect"] > .1)
    lifecycle = {str(r["name"]): float(r["value"]) for r in table(small.path/"lifecycle.csv")}
    gate("lifecycle_constant", lifecycle["constant_error"]/np.sqrt(2.5e6), 1e-11)
    gate("lifecycle_repeat", lifecycle["repeat_error"], 0.)
    for name in ["invalid_source_rejected", "incoming_unchanged", "unsafe_dt_rejected"]:
        gate(name, 1-lifecycle[name], 0.)
    medium = next(e for e in entries if e.nx == 8)
    temporal = table(medium.path/"temporal.csv")
    selected = temporal[temporal["steps"] <= 128]
    time_orders = np.log2(selected["difference_from_1024"][:-1]/selected["difference_from_1024"][1:])
    reference_uncertainty = float(temporal[temporal["steps"] == 512]["difference_from_1024"][0])
    for row in temporal:
        independently_measured = medium.difference("temporal_"+str(int(row["steps"])), "temporal_reference")
        gate("temporal_norm_reference_"+str(row["steps"]), abs(independently_measured-row["difference_from_1024"])/max(1., independently_measured), 1e-11)
    gate("temporal_reference_fraction", reference_uncertainty/selected["difference_from_1024"][-1], .05)
    gates["SSPRK3_temporal_order"] = dict(orders=time_orders.tolist(), minimum=2.7, passed=bool(np.min(time_orders[-2:]) >= 2.7))
    q48 = medium.difference("normal_cohesive", "normal_quad8")
    q812 = medium.difference("normal_quad8", "normal_quad12")
    gate("source_quadrature_reference", q812/max(1., medium.measure("normal_quad12")["reference_L2"]), 1e-10)
    for case in ["frozen", "rotation_nonobjective"]:
        gate(case, medium.measure(case)["relative_L2"], 1e-11)
    gate("rotation_objective_forced", medium.measure("rotation_objective_forced")["relative_L2"], 1e-6)
    finest = entries[-1]
    dt_difference = finest.difference("pulse_translation", "pulse_translation_half_dt")
    gate("spatial_pulse_time_contamination", dt_difference/finest.measure("pulse_translation")["error_L2"], .01)
    closures = []
    for e in entries:
        closures.append(dict(nx=e.nx,
                             normal_difference=e.difference("normal_mature", "normal_residual"),
                             tangent_difference=e.difference("tangent_mature", "tangent_residual")))
    return dict(passed=all(r["passed"] for r in gates.values()), gates=gates, characteristic_results=rows,
                operator=operator, lifecycle=lifecycle, spatial=spatial, discontinuities=discontinuities,
                temporal=dict(steps=selected["steps"].tolist(), differences=selected["difference_from_1024"].tolist(),
                              orders=time_orders.tolist(), reference_uncertainty=reference_uncertainty),
                quadrature=dict(normal_Q4_Q8_difference=q48, normal_Q8_Q12_difference=q812),
                finest_dt_half_difference=dt_difference, direct_residual_closures=closures,
                scope="Prescribed noncommitting DG transport only; analytic surface histories/localization; no limiter or coupled publication")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = assess(args.runs)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(passed=result["passed"], gates=len(result["gates"]), failed=[k for k, r in result["gates"].items() if not r["passed"]])))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
