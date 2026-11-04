#!/usr/bin/env python3
"""Independent Stage A algebra/kinematics checks; no ASPECT or particle solver.

Exact rational component differentiation checks the separated source; finite
coordinate differences check geometric formulas. The scalar variable-rate
exercise checks the declared coefficient order, not the Stage B tensor PDE.
"""
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import random

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
DEFAULT = ROOT / 'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/stage_a/algebra.json'


def comm(w, a):
    return w @ a - a @ w


def tensor(xx, yy, xy):
    return np.array([[xx, xy, F(0)], [xy, yy, F(0)], [F(0), F(0), -xx-yy]], dtype=object)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    args = parser.parse_args()
    checks = {}
    rng = random.Random(20261102)
    q = lambda: F(rng.randrange(-8, 9), rng.randrange(1, 8))
    for _ in range(512):
        chi, chi_dot, s, s_grad_defect, r_bulk, r_surface, slip = [q() for _ in range(7)]
        orient, omega = q(), q()
        # Oblique rational t=(3/5,4/5), n=(-4/5,3/5), full3D S.
        t = np.array([F(3,5), F(4,5), F(0)], dtype=object)
        n = np.array([-F(4,5), F(3,5), F(0)], dtype=object)
        S = (np.outer(t,n)+np.outer(n,t))/2
        W = np.array([[F(0),-omega,F(0)],[omega,F(0),F(0)],[F(0),F(0),F(0)]], dtype=object)
        rotation = np.array([[F(0),-orient,F(0)],[orient,F(0),F(0)],[F(0),F(0),F(0)]], dtype=object)
        S_dot = comm(rotation,S)
        G = F(7,3)
        T, R, strain = [tensor(q(),q(),q()) for _ in range(3)]
        B = 2*G*chi*s*S
        physical = T-B-R
        # Direct component differentiation of B uses bulk derivative of the
        # extended surface scalar, rather than the operator C formula.
        s_dot_bulk = slip-r_surface*s+s_grad_defect
        B_dot = 2*G*(chi_dot*s*S+chi*s_dot_bulk*S+chi*s*S_dot)
        C = s*(chi_dot*S+chi*(S_dot-comm(W,S)))+chi*S*s_grad_defect
        mismatch = (r_bulk-r_surface)*chi*s*S
        R_dot = comm(W,R)-r_bulk*R-2*G*(C+mismatch)
        T_dot = comm(W,T)-r_bulk*T+2*G*strain
        transformed = T_dot-B_dot-R_dot
        direct = comm(W,physical)-r_bulk*physical+2*G*(strain-chi*slip*S)
        assert np.all(transformed == direct), 'General mature product rule'
        # In the fixed-profile matched-rate special case the sign is +[W,B].
        fixed_C = -chi*s*comm(W,S)
        assert np.all(-2*G*fixed_C == comm(W,B)), 'Fixed-profile residual sign'
        assert sum((physical*comm(W,physical)).flat) == 0, 'Spin work'
    checks['exact_rational_product_rules'] = dict(cases=512, maximum_defect=0,
        included=['spin', 'changing profile', 'orientation', 'surface/bulk transport', 'rate mismatch', 'zz', 'spin work'])

    # General finite-mode lifting with a noncommuting surface rate operator.
    for _ in range(128):
        K = [tensor(q(),q(),q()) for _ in range(2)]
        K_dot = [tensor(q(),q(),q()) for _ in range(2)]
        b = np.array([q(),q()],dtype=object)
        velocity = np.array([q(),q()],dtype=object)
        rate = np.array([[q(),q()],[q(),q()]],dtype=object)
        defect = np.array([q(),q()],dtype=object)
        local = q()
        def apply(basis, values):
            return basis[0]*values[0]+basis[1]*values[1]
        omega = q()
        W = np.array([[F(0),-omega,F(0)],[omega,F(0),F(0)],[F(0),F(0),F(0)]],dtype=object)
        A, R, smooth = [tensor(q(),q(),q()) for _ in range(3)]
        G = F(7,3)
        H = 2*G*apply(K,b)
        physical = A+H-R
        CK = apply(K_dot,b)+apply(K,defect)-comm(W,apply(K,b))
        DK = local*apply(K,b)-apply(K,rate@b)
        b_dot = velocity-rate@b+defect
        H_dot = 2*G*(apply(K_dot,b)+apply(K,b_dot))
        A_dot = comm(W,A)-local*A+2*G*smooth
        R_dot = comm(W,R)-local*R+2*G*(CK+DK)
        direct = comm(W,physical)-local*physical+2*G*(smooth+apply(K,velocity))
        assert np.all(A_dot+H_dot-R_dot==direct), 'Native lifting correction sign'
    checks['exact_native_lifting_product_rules'] = dict(cases=128,maximum_defect=0)

    # Native lifting need not commute with a spatially varying coefficient.
    b, db, rate, dr = F(3,2), F(2,3), F(5,4), F(7,6)
    mismatch = rate*db-(rate*db+dr*b)
    assert mismatch == -dr*b and mismatch != 0
    checks['lifting_rate_product_counterexample'] = dict(defect=str(mismatch))

    # Geometry from independent finite deformation F(t)=I+tL, not spin.
    L = np.array([[.2, 1.3], [-.4, -.2]])
    t = np.array([.6, .8]); n = np.array([-.8, .6])
    eps = 1e-6
    def normal(time):
        out = np.linalg.solve((np.eye(2)+time*L).T,n)
        return out/np.linalg.norm(out)
    derivative = (normal(eps)-normal(-eps))/(2*eps)
    exact = -(np.eye(2)-np.outer(n,n)) @ L.T @ n
    err = float(np.linalg.norm(derivative-exact))
    assert err < 2e-9
    kappa = .7
    def curvature(time):
        deformation = np.eye(2)+time*L
        first, second = deformation@t, deformation@(kappa*n)
        return np.linalg.det(np.column_stack([first,second]))/np.linalg.norm(first)**3
    curvature_dot = n@L@(kappa*n)-2*(t@L@t)*kappa
    curvature_error = float(abs((curvature(eps)-curvature(-eps))/(2*eps)-curvature_dot))
    assert curvature_error < 2e-9
    # Closest-point association on a deforming straight oblique curve.
    X = np.array([.3,-.2]); x = X+.7*t+.4*n
    def projection(time):
        deformation = np.eye(2)+time*L
        centre, tangent, point = deformation@X, deformation@t, deformation@x
        tangent /= np.linalg.norm(tangent)
        return centre+tangent*np.dot(point-centre,tangent)
    pi = projection(0); zeta = np.dot(x-pi,n)
    u = L@pi; v = L@x; b_s = n@L@t
    pi_dot = u+t*(np.dot(v-u,t)+zeta*b_s)
    pi_error = float(np.linalg.norm((projection(eps)-projection(-eps))/(2*eps)-pi_dot))
    assert pi_error < 2e-9
    simple = np.array([[0.,1.7],[0.,0.]])
    normal0 = np.array([0.,1.])
    geom_dot = -(np.eye(2)-np.outer(normal0,normal0))@simple.T@normal0
    spin_dot = (simple-simple.T)@normal0/2
    assert np.linalg.norm(geom_dot)==0 and np.linalg.norm(spin_dot)==.85
    checks['geometry'] = dict(normal_derivative_error=err, curvature_derivative_error=curvature_error, association_derivative_error=pi_error,
                              tolerance=2e-9, shear_normal_vs_spin_defect=float(np.linalg.norm(geom_dot-spin_dot)))

    # Scalar ALE relabelling and changing surface mass are distinct.
    c, k = F(2,5), F(7,3)
    assert c*k != 0
    # Projection Mz=f, exact inverse/differentiation in a two-mode example.
    M = np.array([[F(2),F(1)],[F(1),F(3)]],dtype=object)
    inv = np.array([[F(3,5),-F(1,5)],[-F(1,5),F(2,5)]],dtype=object)
    Md = np.array([[F(1),F(2)],[F(2),F(4)]],dtype=object)
    z = np.array([F(1,3),F(5,4)],dtype=object)
    zd = np.array([F(7,5),-F(2,3)],dtype=object)
    fd = Md@z+M@zd
    assert np.all(inv@(fd-Md@z)==zd)
    assert np.any(inv@fd!=zd)
    checks['surface_ALE_and_projection'] = dict(relabel_state_error_rate=str(c*k), mass_solve_derivative_defect=0)

    # Translating finite-support material damage must vacate old cells.
    grid = np.linspace(-2,2,101)
    old = np.maximum(0.,1-4*np.abs(grid))
    transported = np.maximum(0.,1-4*np.abs(grid-.8))
    trail = float(np.max(np.maximum(old,transported)-transported))
    assert trail == 1.
    checks['Eulerian_max_negative_control'] = dict(spurious_trailing_damage=trail)

    # Exact scalar rate r(t)=r0+k t, over a unit time. Endpoint-frozen update.
    r0, slope, duration = .4, .8, 1.
    exact = np.exp(-r0*duration-.5*slope*duration**2)
    errors = []
    for steps in (20,40,80):
        dt=duration/steps
        rate=r0+slope*np.arange(1,steps+1)*dt
        value=np.exp(-dt*np.sum(rate))
        errors.append(float(abs(value-exact)))
    ratios=[errors[i]/errors[i+1] for i in (0,1)]
    assert all(1.9 < r < 2.1 for r in ratios)
    checks['variable_rate_scalar'] = dict(steps=[20,40,80],errors=errors,ratios=ratios,
                                        acceptance='Both error ratios in (1.9,2.1), declared before run')

    W = np.array([[0.,-.6,0.],[.6,0.,0.],[0.,0.,0.]])
    hidden = np.diag([2.,0.,-2.])
    assert hidden[1,1]==hidden[0,1]==0
    prestress = float(np.linalg.norm(comm(W,hidden)))
    assert prestress>0
    checks['prestress_nonuniqueness'] = dict(equal_fault_shear_and_normal=True, stress_rate_difference=prestress)
    Gc, length, stretch_rate=2.,3.,.4
    shape_work = Gc*length*stretch_rate
    assert shape_work>0
    checks['cohesive_measure_work_negative_control'] = dict(omitted_work_J_m_s=shape_work,
        meaning='Current-arclength fully formed crack stretches at zero da and zero slip')

    args.output.parent.mkdir(parents=True,exist_ok=True)
    result=dict(scope='Stage A algebra and kinematics only; not Stage B representation/transport qualification',
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                seed=20261102,checks=checks,status='passed')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
