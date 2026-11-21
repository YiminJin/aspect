"""Conditional endpoint equilibrium, like-point comparisons, and two independent midpoint controls."""
import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np

from midpoint_audit import (ANGLE, CONTRACTION, DC, G, KF, M, fields,
                            inverse_compliance, phase_audit, point_audit, read, tensor)


def endpoint(points, folder, step):
    planes = read(folder / f'tip-planes-{step:04}.csv')
    rates = read(folder / f'tip-sections-{step:04}.csv')
    k = points['section']
    angle = ANGLE + planes['beta'][k]
    S = np.column_stack((-.5*np.sin(2*angle), .5*np.sin(2*angle),
                         np.zeros_like(angle), .5*np.cos(2*angle)))
    tau_s = np.sum(tensor(points, 'tau') * S * CONTRACTION, axis=1)
    A = tensor(points, 'A')
    phi = points['phi']
    # In damaged material the derivative at fixed published A is unique.
    # Only intact points need a conditional endpoint intact reaction. Use the
    # retained section root, endpoint age and published stress on its plane.
    v = rates['resolved_V']
    f = 3*np.arcsinh(v/.002*np.exp((.6+.02*np.log(.001*planes['age']/DC))/.015)) + 1000*v
    Y = M/(2*G)*(tau_s-f[k])**2
    active = phi > 0
    p = phi[active]
    Y[active] = (G*np.sum(A[active]**2*CONTRACTION, axis=1)*M*(1+3*p)*(1-p)
                 / (M*p*(1+p))**2)
    np.testing.assert_allclose(Y[active], points['Y_endpoint'][active], rtol=5e-15, atol=1e-16)
    assert np.all(A[~active] == 0)
    return tau_s, 2*np.sum(A*S*CONTRACTION, axis=1), Y


def projected_load(points, rows, key_values):
    result = np.empty(len(rows))
    for cell in np.unique(rows['cell']):
        group = rows['cell'] == cell
        sample = points['cell'] == cell
        np.testing.assert_array_equal(rows['i'][group], np.arange(9))
        def basis(axis):
            factor = .5 + 8*(points[axis][sample]-rows[axis][group][0])
            return np.array([(1-factor)**2, 2*factor*(1-factor), factor**2])
        B = (basis('y')[:, None, :]*basis('x')[None, :, :]).reshape(9, -1)
        np.testing.assert_allclose(B@points['weight'][sample], rows['mass'][group], atol=1e-16, rtol=1e-13)
        result[group] = B@(points['weight'][sample]*key_values[sample])
    return result


def balance(b, fracture, load, mass):
    g = fracture-load
    projected = np.clip(b-g/(KF*mass), b, 1)-b
    return dict(fracture_only_downhill=int(sum((b<1)&(fracture<0))),
                downhill=int(sum((b<1)&(g<0))), above_stage_tolerance=int(sum(projected>1e-7)),
                increment_only_KKT=float(max(projected)), max_load=float(max(load)),
                min_fracture_gradient=float(min(fracture)), min_total_gradient=float(min(g)))


def reduced_obstacle():
    """Strictly convex two-row energy with an analytically enumerable box VI."""
    H = np.array([[2., -1.], [-1., 2.]])
    load = np.array([.2, -.15])

    def step(parent, midpoint):
        upper = (1+parent)/2 if midpoint else np.ones_like(parent)
        for pair in product((-1, 0, 1), repeat=2):
            status = np.array(pair)
            free = status == 0
            b = np.where(status == 1, upper, parent)
            b[free] = np.linalg.solve(H[np.ix_(free,free)], (load-H[:,~free]@b[~free])[free])
            g = H@b-load
            if np.allclose(b, np.clip(b-g, parent, upper), atol=1e-14, rtol=0):
                return 2*b-parent if midpoint else b
        raise AssertionError('Missing box minimizer')

    out = {}
    for name, midpoint, expected in [('midpoint', True, [[.2,0], [.2,.05]]),
                                     ('endpoint', False, [[.1,0], [.1,0]])]:
        first = step(np.zeros(2), midpoint)
        second = step(first, midpoint)
        np.testing.assert_allclose([first, second], expected, atol=1e-15)
        out[name] = dict(one=first.tolist(), two=second.tolist(), first_endpoint_gradient=(H@first-load).tolist())
    return out


def smooth_work():
    """Fixed-time signed and absolute defects for a smooth strain/phase chord."""
    records = []
    for count in (8,16,32,64):
        t = np.linspace(0, 1, count+1)
        phi, e = .19+.03*t, .001+.0006*t
        pm, em = (phi[1:]+phi[:-1])/2, (e[1:]+e[:-1])/2
        q = G*inverse_compliance(pm)*em
        Y = M*(1+3*pm)/(1-pm)**3*q*q/(2*G)
        psi = .5*G*inverse_compliance(phi)*e**2
        error = q*np.diff(e)-Y*np.diff(phi)-np.diff(psi)
        records.append(dict(steps=count, signed=float(sum(error)), absolute=float(sum(abs(error)))))
    ratios = np.array([r['absolute'] for r in records])
    np.testing.assert_allclose(ratios[:-1]/ratios[1:], 4, rtol=.001)
    return records


def frozen_fracture(points, rows, states):
    """Independent rectangular AT1/SIPG form and frozen-load obstacle solves."""
    from scipy.sparse import coo_matrix
    from scipy.optimize import minimize

    dx, ell, penalty = .125, .25, 10.
    k = 2*KF*ell**2
    nodes, weights = np.polynomial.legendre.leggauss(5)
    nodes, weights = (nodes+1)/2, weights/2

    def shapes(x, y):
        def basis(t):
            return np.array([(1-t)**2, 2*t*(1-t), t*t])
        def deriv(t):
            return np.array([2*(t-1), 2-4*t, 2*t])/dx
        def product(a, b):
            return (a[:,None,:]*b[None,:,:]).reshape(9,-1).T
        return product(basis(y),basis(x)), product(basis(y),deriv(x)), product(deriv(y),basis(x))

    x, y = np.meshgrid(nodes, nodes)
    B, Bx, By = shapes(x.ravel(), y.ravel())
    w = dx**2*np.outer(weights,weights).ravel()
    cell_mass = B.T@(w[:,None]*B)
    cell_stiffness = k*(Bx.T@(w[:,None]*Bx)+By.T@(w[:,None]*By))
    irows, cols, values = [], [], []

    def add(ids, a):
        irows.extend(np.repeat(ids,len(ids)))
        cols.extend(np.tile(ids,len(ids)))
        values.extend(a.ravel())

    cells = {}
    coefficients = {name:np.empty(len(rows)) for name in points}
    for cell in np.unique(rows['cell']):
        ids = np.flatnonzero(rows['cell']==cell)
        ij = tuple(int(rows[t][ids[0]]/dx) for t in ('x','y'))
        cells[ij] = ids
        add(ids,cell_stiffness)
        use = points['parent']['cell']==cell
        basis = shapes(points['parent']['x'][use]/dx-ij[0],points['parent']['y'][use]/dx-ij[1])[0]
        for name, p in points.items():
            solution, _, rank, _ = np.linalg.lstsq(basis,p['phi'][use],rcond=None)
            assert rank == 9
            np.testing.assert_allclose(basis@solution,p['phi'][use],atol=2e-15,rtol=0)
            coefficients[name][ids] = solution
    # Each interior face visited once. Outward normal of the lower-index
    # neighbor is +x or +y. Natural exterior contributes no face energy.
    for axis in (0,1):
        edge = lambda s: shapes(np.full_like(nodes,s),nodes) if axis==0 else shapes(nodes,np.full_like(nodes,s))
        high, low = edge(1.),edge(0.)
        jump = np.column_stack((high[0],-low[0]))
        avg = .5*k*np.column_stack((high[axis+1],low[axis+1]))
        face = dx*(jump.T@(weights[:,None]*(9*penalty*k/dx*jump-avg))-avg.T@(weights[:,None]*jump))
        for (ix,iy), ids in cells.items():
            other = cells.get((ix+(axis==0),iy+(axis==1)))
            if other is not None:
                add(np.r_[ids,other],face)
    H = coo_matrix((values,(irows,cols)),shape=(len(rows),len(rows))).tocsr()
    lin = KF*rows['mass']
    def energy(b):
        return .5*b@(H@b)+lin@b

    # Independent controls for the assembled face signs, penalty and row order.
    np.testing.assert_allclose(H@coefficients['parent']+lin,
                               2*rows['stage_fracture_gradient']-rows['fracture_gradient'],atol=1e-12,rtol=0)
    def norm(b):
        return float(np.sqrt(sum(b[ids]@cell_mass@b[ids] for ids in cells.values())))

    def vi(parent, midpoint):
        upper = (parent+1)/2 if midpoint else np.ones_like(parent)
        def objective(b):
            grad = H@b
            return .5*b@grad+lin@b, grad+lin
        res = minimize(objective,parent,jac=True,method='L-BFGS-B',bounds=list(zip(parent,upper)),
                       options=dict(ftol=1e-15,gtol=1e-10,maxiter=2000,maxcor=20))
        assert res.success, res.message
        # Polish the identified free rows by the exact quadratic restricted
        # equation. Verify every multiplier and row with the projected residual.
        from scipy.sparse.linalg import spsolve
        b = res.x.copy()
        free = (b>parent+1e-10)&(b<upper-1e-10)
        b[free] -= spsolve(H[free][:,free],(H@b+lin)[free])
        kkt = max(abs(b-np.clip(b-(H@b+lin)/lin,parent,upper)))
        assert kkt < 1e-10
        return (2*b-parent if midpoint else b), float(kkt)

    parent_coefficients = coefficients['parent']
    # Same parent and fracture law. Suppress mechanical driving only in these
    # conditional phase solves to measure its role in the observed changes.
    a, r1 = vi(parent_coefficients,True)
    c, r2 = vi(a,True)
    ep, r3 = vi(parent_coefficients,False)
    again, r4 = vi(ep,False)
    final = 'second_half'
    final_balance = balance(coefficients[final],H@coefficients[final]+lin,
                            projected_load(points[final],rows,states[final][2]),rows['mass'])
    # Restrict each quadratic to the 16 one-dimensional intervals of the
    # native depth-four enclosure. Its Bernstein hull bounds the whole cell.
    start = np.arange(16)/16
    def basis_1d(x):
        return np.column_stack(((1-x)**2,2*x*(1-x),x*x))
    derivative = np.column_stack((2*(start-1),2-4*start,2*start))
    split = np.vstack((basis_1d(start),basis_1d(start)+derivative/32,basis_1d(start+1/16)))
    hulls = {}
    for name, b in coefficients.items():
        pieces = np.array([split@b[ids].reshape(3,3)@split.T for ids in cells.values()])
        hulls[name] = [float(pieces.min()),float(pieces.max())]
        assert pieces.min() >= -1e-14 and pieces.max() <= 1
    return dict(phase_enclosures=hulls,fracture={n:float(energy(b)) for n,b in coefficients.items()},
                minimum_coefficients={n:float(min(b)) for n,b in coefficients.items()},
                refined_final_balance=final_balance,
                frozen_midpoint_diff_L2={n:norm(f-coefficients[n]) for n,f in
                                         [('coarse',a),('first_half',a),(final,c)]},
                frozen_first_to_second_L2=norm(c-a),
                frozen_second_increment_error_L2=norm((c-a)-(coefficients[final]-coefficients['first_half'])),
                endpoint_second_increment_L2=norm(again-ep),
                endpoint_first_increment_L2=norm(ep-parent_coefficients), projected_residuals=[r1,r2,r3,r4])


def audit(runs):
    labels = [('parent','localized-oneo1',1), ('coarse','localized-oneo2',2),
              ('first_half','strip18-midpoint-half',2), ('second_half','strip18-midpoint-half',3)]
    points, states = {}, {}
    for name, case, step in labels:
        folder = runs/case/'output'
        points[name] = fields(folder,step)
        states[name] = endpoint(points[name],folder,step)
        for key in ('cell','q','section','x','y','weight'):
            np.testing.assert_array_equal(points[name][key], points['parent'][key])
    balances = {}
    for label, observer in [('coarse','strip18-audit-full'), ('first_half','strip18-audit-half1')]:
        rows, _, _ = phase_audit(runs/observer/'output')
        # Validate the integration/ordering against the saved native stage load.
        stage_load = projected_load(points[label], rows, points[label]['Y'])
        np.testing.assert_allclose(stage_load, rows['driving_load'], rtol=1e-12, atol=2e-16)
        for state, b, grad in [('parent',rows['lower'],2*rows['stage_fracture_gradient']-rows['fracture_gradient']),
                               (label,rows['B'],rows['fracture_gradient'])]:
            values = balance(b,grad,projected_load(points[state],rows,states[state][2]),rows['mass'])
            if state in balances:
                assert values['downhill'] == balances[state]['downhill']
                np.testing.assert_allclose(values['increment_only_KKT'],balances[state]['increment_only_KKT'],atol=1e-14)
            balances[state] = values
    profiles = {}
    for label, left, right in [('largest_first_increment','parent','coarse'),
                               ('largest_second_increment','first_half','second_half')]:
        i = np.argmax(points[right]['phi']-points[left]['phi'])
        point = points['parent'][i]
        profiles[label] = dict(cell=str(point['cell']),q=int(point['q']),x=float(point['x']),y=float(point['y']),
                               states={name:dict(phi=float(p['phi'][i]),resolved_stress_Pa=float(states[name][0][i]),
                                                 cohesive_shear_strain=float(states[name][1][i]),
                                                 endpoint_Y_Pa=float(states[name][2][i]),stage_Y_Pa=float(p['Y'][i]))
                                       for name,p in points.items()})
    frozen = frozen_fracture(points,rows,states)
    ledgers, work = {}, {}
    for name, case, step in labels:
        ledger = read(runs/case/'output'/'bulk-accepted.csv')
        recorded = ledger[ledger['step']==step]['fracture'][0]
        np.testing.assert_allclose(frozen['fracture'][name],recorded,atol=1e-12,rtol=0)
        row = ledger[ledger['step']==step][0].copy()
        row['fracture'] = frozen['fracture'][name]
        ledgers[name] = row
        if name != 'parent':
            parent = 'first_half' if name=='second_half' else 'parent'
            terms = point_audit(points[parent],points[name],runs/case/'output',step,row,ledgers[parent],[])
            work[name] = {k:terms[k] for k in ('physical_residual','reconstructed_remainder','positive','negative')}
    return dict(independent_work=work,frozen=frozen,endpoint_balance=balances,points=profiles,reduced_box=reduced_obstacle(),smooth_work=smooth_work())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path)
    print(json.dumps(audit(parser.parse_args().runs),indent=2))
