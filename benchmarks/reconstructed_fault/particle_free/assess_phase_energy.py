#!/usr/bin/env python3
"""PF3 independent quadrature/complex-step references for the native DG operator.

No production constitutive, normalization or phase-assembly routine is imported.
Exported native geometry, DG tensor values and incoming Z are the inputs.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import eigvalsh


def csv(path):
    return np.genfromtxt(path, delimiter=",", names=True)


def gauss(n):
    x, w = leggauss(n)
    return (x+1)/2, w/2


def q1(r, s):
    return np.array([(1-r)*(1-s), r*(1-s), (1-r)*s, r*s]).T


def lagrange2(x):
    return np.array([(2*x-1)*(x-1), 4*x*(1-x), x*(2*x-1)]).T


def scaled_error(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a-b))/max(1., float(np.max(np.abs(b)))))


class Reference:
    def __init__(self, directory):
        self.path = directory
        self.nodes = csv(directory/"nodes.csv")
        self.cells = csv(directory/"cells.csv")
        self.tensor = csv(directory/"tensor.csv")
        self.surface = csv(directory/"surface.csv")
        self.profiles = csv(directory/"profiles.csv")
        self.constants = csv(directory/"constants.csv")
        self.G = float(self.constants["G"])
        self.m = float(self.constants["m"])
        self.A = float(self.constants["A"])
        self.ell = float(self.constants["ell"])
        self.n = len(self.nodes)
        self.ns = len(self.surface)
        self.indices = np.stack([self.cells[f"i{i}"] for i in range(4)], axis=1).astype(int)
        self.sf = np.zeros((len(self.profiles), self.ns))
        for k, profile in enumerate(self.profiles):
            segment = int(profile["segment"])
            self.sf[k, segment:segment+2] = [1-profile["xi"], profile["xi"]]
        wf = self.profiles["weight"]
        mass = self.sf.T @ (wf[:, None]*self.sf)
        self.P = np.linalg.solve(mass, self.sf.T*wf)
        # Independent normal integration, with the cell containing each fixed x
        # found from physical coordinates, rather than exported ray jets.
        g, w = gauss(192)
        normal_indices, normal_shapes, normal_weights = [], [], []
        for profile in self.profiles:
            seg, xi = int(profile["segment"]), profile["xi"]
            x = (1-xi)*self.surface["x"][seg]+xi*self.surface["x"][seg+1]
            crossed = np.flatnonzero((x >= self.cells["x0"]) & (x < self.cells["x0"]+self.cells["dx"]))
            ni, nn, nw = [], [], []
            for c in crossed:
                r = (x-self.cells["x0"][c])/self.cells["dx"][c]
                ni.append(np.tile(self.indices[c], (len(g), 1)))
                nn.append(q1(np.full_like(g, r), g))
                nw.append(w*self.cells["dy"][c])
            normal_indices.append(np.concatenate(ni))
            normal_shapes.append(np.concatenate(nn))
            normal_weights.append(np.concatenate(nw))
        self.ni = np.array(normal_indices)
        self.nn = np.array(normal_shapes)
        self.nw = np.array(normal_weights)
        assert np.max(np.abs(self.nw.sum(axis=1)-1)) < 1e-13
        self.cache = {}

    def bulk_rule(self, order=None):
        if order in self.cache:
            return self.cache[order]
        if order is None:
            rule = csv(self.path/"rule.csv")
            r, s, w = rule["r"], rule["s"], rule["weight"]
        else:
            # Same tangential intersection partition, independently generated.
            x, wx = gauss(3)
            x, wx = np.concatenate([x/2, .5+x/2]), np.tile(wx/2, 2)
            y, wy = gauss(order)
            r, s = np.repeat(x, len(y)), np.tile(y, len(x))
            w = np.repeat(wx, len(y))*np.tile(wy, len(x))
        N = q1(r, s)
        ix = np.repeat(self.indices[:, None, :], len(r), axis=1).reshape(-1, 4)
        nn = np.tile(N, (len(self.cells), 1))
        grad = np.empty((len(self.cells), len(r), 4, 2))
        grad[:, :, :, 0] = np.array([-(1-s), 1-s, -s, s]).T[None, :, :]/self.cells["dx"][:, None, None]
        grad[:, :, :, 1] = np.array([-(1-r), -r, 1-r, r]).T[None, :, :]/self.cells["dy"][:, None, None]
        weights = (self.cells["dx"]*self.cells["dy"])[:, None]*w[None, :]
        x = self.cells["x0"][:, None]+self.cells["dx"][:, None]*r[None, :]
        segment = np.minimum(self.ns-2, (x*(self.ns-1)).astype(int)).ravel()
        xi = (x*(self.ns-1)).ravel()-segment
        sf = np.zeros((len(segment), self.ns))
        sf[np.arange(len(segment)), segment] = 1-xi
        sf[np.arange(len(segment)), segment+1] = xi
        T = np.zeros((len(self.cells), len(r), 3))
        lr, ls = lagrange2(r), lagrange2(s)
        for c in range(len(self.cells)):
            data = self.tensor[self.tensor["cell"] == c]
            for sample in data:
                shape = lr[:, round(2*sample["r"])]*ls[:, round(2*sample["s"])]
                T[c] += shape[:, None]*np.array([sample["xx"], sample["yy"], sample["xy"]])
        result = ix, nn, grad.reshape(-1, 4, 2), weights.ravel(), sf, T.reshape(-1, 3)
        self.cache[order] = result
        return result

    def evaluate(self, p, order=None, pullback=True):
        dtype = np.result_type(p)
        f = np.sum(p[self.ni]*self.nn, axis=2)
        h = self.m*f*(1+f)/(1-f)**2
        hp = self.m*(4/(1-f)**3-3/(1-f)**2)
        direct = np.sum(self.nw*h, axis=1)
        I = self.P @ direct
        db = np.zeros((len(self.profiles), self.n), dtype=dtype)
        for k in range(len(self.profiles)):
            for v in range(4):
                np.add.at(db[k], self.ni[k, :, v], self.nw[k]*hp[k]*self.nn[k, :, v])
        DI = self.P @ db
        ix, N, gradN, w, sf, T = self.bulk_rule(order)
        phi = np.sum(p[ix]*N, axis=1)
        grad = np.sum(p[ix, None]*gradN, axis=1)
        ii, z = sf @ I, sf @ self.surface["Z"]
        h = self.m*phi*(1+phi)/(1-phi)**2
        hp = self.m*(4/(1-phi)**3-3/(1-phi)**2)
        chi = h/ii
        shear = T[:, 2]-z*chi
        norm = T[:, 0]**2+T[:, 1]**2+(T[:, 0]+T[:, 1])**2+2*shear**2
        bulk = np.sum(w*norm)/(4*self.G)
        fs = self.sf @ I
        zs = self.sf @ self.surface["Z"]
        coh = np.sum(self.profiles["weight"]*zs**2/fs)/(2*self.G)
        frac = np.sum(w*self.A*(phi+self.ell**2*np.sum(grad*grad, axis=1)))
        c = -z/self.G*shear
        gradient = np.zeros(self.n, dtype=dtype)
        fracture_gradient = np.zeros(self.n, dtype=dtype)
        for v in range(4):
            np.add.at(gradient, ix[:, v], w*c*hp/ii*N[:, v])
            np.add.at(fracture_gradient, ix[:, v], w*self.A*(N[:, v]+2*self.ell**2*np.sum(grad*gradN[:, v], axis=1)))
        forceI = sf.T @ (-w*c*h/ii**2)-self.sf.T @ (self.profiles["weight"]*zs**2/(2*self.G*fs**2))
        if pullback:
            gradient += DI.T @ forceI
        gradient += fracture_gradient
        loads = sf.T @ (w*chi*shear)
        return dict(energy=bulk+coh+frac, bulk=bulk, cohesive=coh, fracture=frac,
                    gradient=gradient, I=I, DI=DI, loads=loads)

    def cpdi(self, p):
        domains = csv(self.path/"domains.csv")
        energy = 0.
        for particle in np.unique(domains["particle"]):
            data = domains[domains["particle"] == particle]
            values = p[data["i"].astype(int)]
            phi = values @ data["w"]
            grad = np.array([values @ data["gx"], values @ data["gy"]])
            energy += data["volume"][0]*self.A*(phi+self.ell**2*(grad @ grad))
        return energy


def representation_study(G, m, A, ell):
    """Separate phase interpolation, DG history projection and CPDI quadrature.

    These are independent manufactured controls, not new native trajectories.
    """
    # Match native material constants. Only the independent manufactured fields
    # change here; no native material/trajectory is optimized.
    Z = 5000.
    def phase(y):
        return .05+.8*np.exp(-((y-.5)/.035)**2)
    def tensor(y):
        return 20000*(1+.2*np.sin(4*np.pi*y))
    def energy(n, phase_degree=None, history_degree=None):
        r, w = gauss(128)
        y = (np.arange(n)[:, None]+r)/n
        weights = np.tile(w/n, (n, 1))
        p = phase(y)
        if phase_degree is not None:
            p0 = phase(np.arange(n+1)/n)
            p = p0[:-1, None]*(1-r)+p0[1:, None]*r
        T = tensor(y)
        if history_degree is not None:
            v = np.polynomial.legendre.legvander(2*r-1, history_degree)
            # Exact projection of smooth source to each independent cell.
            coefficients = (T*w) @ v * (2*np.arange(history_degree+1)+1)
            T = coefficients @ v.T
        h = m*p*(1+p)/(1-p)**2
        I = np.sum(weights*h)
        tau = T-Z*h/I
        return np.sum(weights*tau*tau)/(2*G)+Z*Z/(2*G*I)
    reference = energy(512)
    phase_table, history_table = [], []
    for n in [16, 32, 64, 128, 256, 512, 1024]:
        value = energy(n, phase_degree=1)
        phase_table.append(dict(cells=n, energy=value, abs_error=abs(value-reference)))
    # Exact phase for this history-only control isolates the projected tensor.
    for degree in [1, 2]:
        for n in [8, 16, 32, 64]:
            value = energy(n, history_degree=degree)
            history_table.append(dict(cells=n, degree=degree, energy=value, abs_error=abs(value-reference)))
    # Smooth 2D field; AT1 reaction averages exactly, whereas CPDI squares
    # each subdomain's average gradient. Gauss2 integrates Q1 fracture exactly.
    fracture_table = []
    exact_frac = A*(.15+.4*4/np.pi**2+ell**2*.4**2*np.pi**2/2)
    g, w = gauss(2)
    r, s = np.repeat(g, 2), np.tile(g, 2)
    N, w2 = q1(r, s), np.repeat(w, 2)*np.tile(w, 2)
    for n in [8, 16, 32, 64]:
        fe, cpdi = 0., 0.
        for iy in range(n):
            for ix in range(n):
                corners = np.array([[ix, iy], [ix+1, iy], [ix, iy+1], [ix+1, iy+1]])/n
                p = .15+.4*np.sin(np.pi*corners[:, 0])*np.sin(np.pi*corners[:, 1])
                phi = N @ p
                gx = np.array([-(1-s), 1-s, -s, s]).T @ p*n
                gy = np.array([-(1-r), -r, 1-r, r]).T @ p*n
                fe += np.sum(w2*A*(phi+ell**2*(gx*gx+gy*gy)))/n**2
                for sy in range(3):
                    for sx in range(3):
                        rr, ss = (sx+.5)/3, (sy+.5)/3
                        pp = q1(np.array([rr]), np.array([ss]))[0] @ p
                        dx = np.array([-(1-ss), 1-ss, -ss, ss]) @ p*n
                        dy = np.array([-(1-rr), -rr, 1-rr, rr]) @ p*n
                        cpdi += A*(pp+ell**2*(dx*dx+dy*dy))/(9*n**2)
        fracture_table.append(dict(cells_per_axis=n, fe=fe, cpdi=cpdi, difference=fe-cpdi,
                                   fe_error=abs(fe-exact_frac), cpdi_error=abs(cpdi-exact_frac)))
    return dict(reference_energy=reference, reference_uncertainty=abs(reference-energy(256)),
                constants=dict(G=G, m=m, A=A, ell=ell, Z=Z), phase_Q1=phase_table, history_DG=history_table,
                fracture_reference=exact_frac, fe_cpdi=fracture_table)


def assess(path):
    ref = Reference(path)
    p = ref.nodes["phase"]
    expected = ref.evaluate(p)
    energies = csv(path/"energy.csv")
    native_gradient = csv(path/"gradient.csv")["total"]
    native_hessian = np.loadtxt(path/"hessian.csv", delimiter=",")
    normal = csv(path/"normalization.csv")
    gates = {}
    def gate(name, error, tolerance):
        gates[name] = dict(error=float(error), tolerance=tolerance, passed=bool(error <= tolerance))
    for key, column in [("energy", "total"), ("bulk", "bulk"), ("cohesive", "cohesive"), ("fracture", "fracture")]:
        gate(key, scaled_error(energies[column], expected[key]), 1e-10)
    gate("normalization", scaled_error(normal["I"], expected["I"]), 1e-10)
    gate("gradient", scaled_error(native_gradient, expected["gradient"]), 1e-10)
    gate("hessian_symmetry", scaled_error(native_hessian, native_hessian.T), 1e-12)
    directional = []
    for direction in ["d1", "d2"]:
        d = ref.nodes[direction]
        perturbed = ref.evaluate(p.astype(complex)+1j*1e-30*d)
        independent_gradient_action = perturbed["energy"].imag/1e-30
        independent_hessian_action = perturbed["gradient"].imag/1e-30
        gate(f"complex_energy_{direction}", scaled_error(native_gradient @ d, independent_gradient_action), 1e-10)
        gate(f"complex_hessian_{direction}", scaled_error(native_hessian @ d, independent_hessian_action), 1e-10)
        gate(f"normalization_action_{direction}", scaled_error(normal[direction], expected["DI"] @ d), 1e-10)
        directional.append(dict(direction=direction, independent_gradient_action=independent_gradient_action,
                                hessian_action_inf=float(np.max(np.abs(independent_hessian_action)))))
    differences = csv(path/"differences.csv")
    sweeps = []
    for d in [0, 1]:
        data = differences[differences["direction"] == d]
        e = np.abs(data["energy_difference"]-data["gradient_action"])/np.maximum(1., np.abs(data["gradient_action"]))
        h = data["hessian_error"]/data["hessian_scale"]
        gate(f"energy_difference_direction{d}", np.min(e), 1e-6)
        gate(f"hessian_difference_direction{d}", np.min(h), 1e-6)
        sweeps.append(dict(direction=d, epsilon=data["epsilon"].tolist(), energy_scaled=e.tolist(), hessian_scaled=h.tolist()))
    cpdi = ref.cpdi(p)
    gate("cpdi_reference", scaled_error(energies["cpdi_fracture"], cpdi), 1e-10)
    missing_projection = scaled_error(ref.evaluate(p, pullback=False)["gradient"], expected["gradient"])
    gates["omitted_projection_negative_control"] = dict(error=missing_projection, minimum=1e-3, passed=missing_projection > 1e-3)
    quadrature = csv(path/"quadrature.csv")
    loads = csv(path/"quadrature-loads.csv")
    quadrature_results = []
    sharp = .05+.945*np.maximum(0, 1-np.abs(ref.nodes["y"]-.5)/.0625)
    for profile, nodal in enumerate([p, sharp]):
        resolved = ref.evaluate(nodal, order=192)
        reference2 = ref.evaluate(nodal, order=256)
        uncertainty = {key: scaled_error(resolved[key], reference2[key]) for key in ["energy", "bulk", "I", "loads"]}
        for key, error in uncertainty.items():
            gate(f"quadrature_reference_uncertainty_{profile}_{key}", error, 1e-8)
        for row in quadrature[quadrature["profile"] == profile]:
            order = int(row["order"])
            same = ref.evaluate(nodal, order=order)
            native_load = loads[(loads["profile"] == profile) & (loads["order"] == order)]["load"]
            gate(f"quadrature_same_rule_{profile}_{order}", scaled_error(row["energy"], same["energy"]), 1e-10)
            gate(f"quadrature_load_same_rule_{profile}_{order}", scaled_error(native_load, same["loads"]), 1e-10)
            quadrature_results.append(dict(profile=profile, order=order, energy=float(row["energy"]),
                                           energy_scaled_error=scaled_error(row["energy"], resolved["energy"]),
                                           bulk_scaled_error=scaled_error(row["bulk"], resolved["bulk"]),
                                           normalization_scaled_error=scaled_error([row["I_min"], row["I_max"]], [np.min(resolved["I"]), np.max(resolved["I"])]),
                                           load_scaled_error=scaled_error(native_load, resolved["loads"])))
        last = quadrature_results[-1]
        gate(f"resolved_order96_energy_{profile}", last["energy_scaled_error"], 1e-8)
        gate(f"resolved_order96_load_{profile}", last["load_scaled_error"], 1e-8)
    controls_data = np.genfromtxt(path/"controls.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
    controls = {str(row["name"]): float(row["value"]) for row in controls_data}
    gate("no_healing", controls["bounded_healing"], 0.)
    gate("redistribution", controls["redistribution_error"]/max(1., controls["redistribution_norm"]), 1e-10)
    for name in ["particle_properties_independent", "accepted_history_unchanged"]:
        gate(name, abs(controls[name]-1), 0.)
    gate("repeated_bounded_exact", abs(controls["repeated_bounded_exact"]-1), 0.)
    bounded = csv(path/"bounded-phase.csv")
    bounded_reference = ref.evaluate(bounded["phase"])
    resolved_bounded = ref.evaluate(bounded["phase"], order=192)
    gate("bounded_residual_reference", scaled_error(-bounded["rhs"], bounded_reference["gradient"]), 1e-10)
    gate("bounded_energy_reference", scaled_error(controls["bounded_final_energy"], bounded_reference["energy"]), 1e-10)
    contact = bounded["phase"] == bounded["lower"]
    H = np.loadtxt(path/"bounded-hessian.csv", delimiter=",")
    ix, N, _, w, _, _ = ref.bulk_rule()
    mass = np.zeros_like(H)
    for i in range(4):
        for j in range(4):
            np.add.at(mass, (ix[:, i], ix[:, j]), w*N[:, i]*N[:, j])
    free = ~contact
    curvature = eigvalsh(H[np.ix_(free, free)], mass[np.ix_(free, free)])
    # These eigenvalues concern the fixed-history phase functional alone.
    # They do not qualify the mechanically reduced coupled Hessian.
    bounded_assessment = dict(
        free_dofs=int(np.sum(free)), lower_contacts=int(np.sum(contact)),
        min_free_mass_normalized_curvature=float(curvature[0]),
        minimum_contact_gradient=float(np.min(-bounded["rhs"][contact])),
        resolved_energy_scaled_difference=scaled_error(bounded_reference["energy"], resolved_bounded["energy"]),
        resolved_gradient_scaled_difference=scaled_error(bounded_reference["gradient"], resolved_bounded["gradient"]),
        max_phase=float(np.max(bounded["phase"])),
        max_q_before=float(np.max(ref.surface["Z"]/ref.surface["I_accepted"])),
        max_q_after=float(np.max(ref.surface["Z"]/bounded_reference["I"])),
        max_elastic_slip=float(np.max(ref.surface["Z"])/ref.G))
    study = representation_study(ref.G, ref.m, ref.A, ref.ell)
    gate("representation_reference_uncertainty", study["reference_uncertainty"], 1e-10)
    phase_errors = [r["abs_error"] for r in study["phase_Q1"]]
    # A subcell nonlinear profile need not converge monotonically while its
    # phase representation is unresolved. Retain those errors, then test the
    # independently refined asymptotic sequence (256/512/1024 cells).
    gates["phase_representation_asymptotic_convergence"] = dict(errors=phase_errors, tested_cells=[256, 512, 1024], passed=bool(np.all(np.diff(phase_errors[-3:]) < 0)))
    for degree in [1, 2]:
        errors = [r["abs_error"] for r in study["history_DG"] if r["degree"] == degree]
        gates[f"DG{degree}_representation_convergence"] = dict(errors=errors, passed=bool(np.all(np.diff(errors) < 0)))
    gaps = [r["difference"] for r in study["fe_cpdi"]]
    gates["fe_cpdi_functional_difference_convergence"] = dict(differences=gaps, passed=bool(np.all(np.diff(gaps) < 0) and min(gaps) > 0))
    return dict(passed=all(g["passed"] for g in gates.values()), gates=gates, native_controls=controls,
                bounded_assessment=bounded_assessment,
                directional_reference=directional, perturbation_sweeps=sweeps, quadrature=quadrature_results,
                representation=study, native_fracture=dict(fe=float(energies["fracture"]), cpdi=float(energies["cpdi_fracture"]),
                                                          difference=float(energies["fracture"]-energies["cpdi_fracture"])),
                scope="Fixed-surface phase operator; independent manufactured representation studies; no coupled evolution, transport or stability qualification")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = assess(args.run/"run")
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(passed=result["passed"], gates=len(result["gates"]), failed=[name for name, gate in result["gates"].items() if not gate["passed"]])))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
