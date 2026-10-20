#!/usr/bin/env python3
"""Independent P3 fixed-geometry transfer assessment; never changes histories.

Reconstructs the native compact seed, Q1 physical field, consistent surface
projection and full tensor stress. Uses orthogonal Legendre projection rather
than the native nodal mass inverse. Optional native-owner/checkpoint checks
use the actual simulator exports. No mechanically equilibrated propagation
trajectory is claimed.
"""
import argparse
import json
import re
from pathlib import Path
import numpy as np

G, ELL, M = 1e6, .15625, 128.
NX, NY = 32, 64


def csv(path):
    return np.atleast_1d(np.genfromtxt(path, delimiter=',', names=True, dtype=None, encoding='utf8'))


def gauss(n):
    z, w = np.polynomial.legendre.leggauss(n)
    return (z+1)/2, w/2


def h(p):
    return M*p*(1+p)/(1-p)**2


def seed_table():
    # Independent integration of dzeta/dtheta for a(phi)=phi and p=1.
    # Endpoint limits are evaluated analytically, not copied native samples.
    peak = .6
    theta = np.linspace(0., np.pi/2, 5000)
    p = peak*np.cos(theta)**2
    v = np.zeros_like(p)
    v[1:-1] = 2*ELL*peak*np.sin(theta[1:-1])*np.cos(theta[1:-1])*np.sqrt(h(peak)/(h(peak)*p[1:-1]-peak*h(p[1:-1])))
    hp = M*(1+3*peak)/(1-peak)**3
    v[0] = 2*ELL*np.sqrt(peak*h(peak)/(peak*hp-h(peak)))
    v[-1] = 2*ELL*np.sqrt(peak*h(peak)/(h(peak)-peak*M))
    r = np.r_[0., np.cumsum((v[1:]+v[:-1])*np.diff(theta)/2)]
    return r, p


R, PHI = seed_table()


def continuum(x, y):
    return np.interp(np.hypot(np.maximum(x-.5, 0.), y-.5), R, PHI, right=0.)


def phase(x, y):
    i = np.minimum((x*NX).astype(int), NX-1)
    j = np.minimum((y*NY).astype(int), NY-1)
    u, v = x*NX-i, y*NY-j
    result = np.zeros(np.broadcast_shapes(np.shape(x), np.shape(y)))
    for a in range(2):
        for b in range(2):
            result += (u if a else 1-u)*(v if b else 1-v)*continuum((i+a)/NX, (j+b)/NY)
    return result


def normalization(s, order=16):
    z, w = gauss(order)
    origins = s['x'][:-1, None]+np.diff(s['x'])[:, None]*z
    y = (np.arange(NY)[:, None]+z)/NY
    integrals = np.sum(h(phase(origins[..., None, None], y))*w/NY, axis=(-2, -1))
    matrix = np.zeros((len(s), len(s)))
    rhs = np.zeros(len(s))
    for j in range(len(s)-1):
        shapes = np.array([1-z, z])
        measure = np.diff(s['x'])[j]*w
        matrix[j:j+2, j:j+2] += (shapes*measure)@shapes.T
        rhs[j:j+2] += shapes@(measure*integrals[j])
    return np.linalg.solve(matrix, rhs)


def lagrange(u):
    return np.array([2*(u-.5)*(u-1), 4*u*(1-u), 2*u*(u-.5)]).T


def inner(a):
    return a[..., 0]**2+a[..., 1]**2+(a[..., 0]+a[..., 1])**2+2*a[..., 2]**2


def load_state(path, name):
    surface = csv(path/f'{name}-surface.csv')
    cells = np.concatenate([csv(p) for p in sorted(path.glob(f'{name}-cells-*.csv'))])
    cells.sort(order=['x0', 'y0'])
    assert len(cells) == NX*NY and len(set(cells['cell'])) == NX*NY
    coefficients = np.stack([np.stack([cells[f'c{c}_{i}'] for i in range(9)], axis=1) for c in range(3)], axis=2)
    return surface, cells, coefficients


def evaluate_profile(surface, x, y):
    active = (x >= surface['x'][0]) & (x <= surface['x'][-1])
    j = np.clip(np.searchsorted(surface['x'], x, side='left')-1, 0, len(surface)-2)
    t = (x-surface['x'][j])/(surface['x'][j+1]-surface['x'][j])
    I = (1-t)*surface['I'][j]+t*surface['I'][j+1]
    Z = (1-t)*surface['Z'][j]+t*surface['Z'][j+1]
    chi = np.where(active, h(phase(x, y))/I, 0.)
    return j, t, chi, np.where(active, Z*chi, 0.), active


def storage(s):
    z, w = gauss(20)
    Z = (1-z)*s['Z'][:-1, None]+z*s['Z'][1:, None]
    I = (1-z)*s['I'][:-1, None]+z*s['I'][1:, None]
    return np.sum(np.diff(s['x'])[:, None]*w*Z**2/I)/(2*G)


def assess(path, native_after=None, native_before=None):
    global M
    M=128.
    if native_after:
        # Use the actual configured constitutive coefficient, independently of
        # the nominal seed label. Rounded Gc gives m=128.00000000016001.
        values=re.findall(r'Accepted FE strain: G=1000000 eta=100000000 lambda=100 m=([^ ]+)',(path/'log.txt').read_text())
        assert values and len(set(values))==1
        M=float(values[0])
    old, cells, a = load_state(path, 'old')
    new, other, b = load_state(path, 'new')
    restarted, rcells, rb = load_state(native_after, 'restored') if native_after else load_state(path, 'restart')
    if native_before:
        restored_old, old_cells, restored_a = load_state(native_before, 'restored')
        assert np.array_equal(restored_old, old) and np.array_equal(restored_a, a)
        for initial, restored in [('old', native_before), ('new', native_after)]:
            assert (path/f'{initial}-persistent.bin').read_bytes()==(restored/'restored-persistent.bin').read_bytes()
            assert np.array_equal(csv(path/f'{initial}-phase.csv'), csv(restored/'restored-phase.csv'))
            for suffix in ['stress.csv','native-loads.csv','ledger.csv','constraints.txt']:
                assert (path/f'{initial}-{suffix}').read_bytes()==(restored/f'restored-{suffix}').read_bytes()
        assert (path/'old-constraints.txt').read_bytes()==(path/'new-constraints.txt').read_bytes()
        _, _, retried_b = load_state(native_before, 'new')
        assert np.array_equal(retried_b, b)
        assert (path/'new-persistent.bin').read_bytes()==(native_before/'new-persistent.bin').read_bytes()
    assert np.array_equal(cells[['x0', 'y0']], other[['x0', 'y0']])
    assert np.array_equal(restarted, new) and np.array_equal(rb, b)
    assert np.array_equal(a[..., :2], b[..., :2])
    for field in ['x', 'y', 'Z', 'cumulative_slip', 'rate']:
        assert np.array_equal(old[field], new[field][:len(old)])
    assert all(np.all(new[f][len(old):] == 0.) for f in ['Z', 'cumulative_slip', 'rate'])
    ir = [normalization(s) for s in [old, new]]
    Ierror = max(np.max(np.abs(s['I']-v))/max(1., np.max(np.abs(v))) for s, v in zip([old, new], ir))
    assert Ierror < 1e-11 and np.min(new['I']) > 0
    nodes = csv(path/'basis-0.csv')
    z, w = gauss(16)
    u = np.r_[z/2, (1+z)/2]
    wx = np.r_[w/2, w/2]
    u, v = np.repeat(u, len(z)), np.tile(z, len(u))
    weights = np.repeat(wx, len(z))*np.tile(w, len(wx))
    shape = lagrange(u)[:, np.rint(nodes['u']*2).astype(int)]*lagrange(v)[:, np.rint(nodes['v']*2).astype(int)]
    x = cells['x0'][:, None]+cells['hx'][:, None]*u
    y = cells['y0'][:, None]+cells['hy'][:, None]*v
    area = cells['hx']*cells['hy']
    volume = area[:, None]*weights
    old_profile = evaluate_profile(old, x, y)
    new_profile = evaluate_profile(new, x, y)
    tau0 = np.einsum('qi,cik->cqk', shape, a)
    tau1 = np.einsum('qi,cik->cqk', shape, b)
    analytic = np.stack([400+71*x+33*y+19*x*y+11*x*x+13*y*y+7*x*x*y+5*x*y*y+3*x*x*y*y,
                         -130+17*x-21*y+9*x*y+5*x*x-7*y*y-4*x*x*y+3*x*y*y+2*x*x*y*y,
                         800-29*x+41*y-17*x*y+23*x*x+31*y*y+11*x*x*y-13*x*y*y+7*x*x*y*y],axis=-1)
    reproduction_error = np.max(np.abs(tau0-analytic))/np.max(np.abs(analytic))
    assert reproduction_error < 1e-12
    tau0[..., 2] -= old_profile[3]
    tau1[..., 2] -= new_profile[3]
    difference = tau1-tau0
    scale = np.sum(volume*inner(tau0))
    error = np.sqrt(np.sum(volume*inner(difference))/scale)
    localized_error = np.sqrt(np.sum(volume*h(phase(x,y))*inner(difference))/np.sum(volume*h(phase(x,y))*inner(tau0)))
    naive = np.sqrt(np.sum(volume*2*(new_profile[3]-old_profile[3])**2)/scale)
    # Independent orthogonal projection in all nine tensor-product modes.
    pu = np.array([np.ones_like(u), 2*u-1, 6*u*u-6*u+1])
    pv = np.array([np.ones_like(v), 2*v-1, 6*v*v-6*v+1])
    delta = new_profile[3]-old_profile[3]
    correction = np.zeros((len(cells), 9))
    moments = np.zeros((len(cells), 9))
    for j in range(3):
        for i in range(3):
            modal = np.sum(delta*weights*pu[i]*pv[j], axis=1)*(2*i+1)*(2*j+1)
            U = np.polynomial.legendre.legval(2*nodes['u']-1, [0.]*i+[1.])
            V = np.polynomial.legendre.legval(2*nodes['v']-1, [0.]*j+[1.])
            correction += modal[:, None]*(U*V)
            moments[:, 3*j+i] = np.sum(volume*difference[..., 2]*pu[i]*pv[j], axis=1)
    projection_error = np.max(np.abs((b-a)[..., 2]-correction))/max(1., np.max(np.abs(correction)))
    assert projection_error < 1e-11
    E0, E1 = [np.sum(volume*inner(t))/(4*G) for t in [tau0, tau1]]
    # The same cohesive weak functional as native bulk work: JxW*N*chi*tau_xy.
    def load(profile, stress, subset=None):
        j, t, chi, _, active = profile
        result = np.zeros(len(new))
        value = volume*chi*stress
        if subset is not None:
            active &= subset
        for index, factor in [(j, 1-t), (j+1, t)]:
            np.add.at(result, index[active], (value*factor)[active])
        return result
    L0 = load(old_profile, tau0[..., 2])
    L1 = load(new_profile, tau1[..., 2])
    ideal = load(new_profile, tau0[..., 2])
    overlap = load(evaluate_profile(new, x, y), tau0[..., 2], x < .5)
    activation = load(evaluate_profile(new, x, y), tau0[..., 2], x > .5)
    if native_after:
        native0=csv(path/'old-native-loads.csv');native1=csv(path/'new-native-loads.csv')
        load_error=max(np.max(np.abs(L0[:len(old)]-native0['shear']))/max(1.,np.max(np.abs(L0))),
                       np.max(np.abs(L1-native1['shear']))/max(1.,np.max(np.abs(L1))))
    else:
        native = csv(path/'native-loads.csv')
        load_error = max(np.max(np.abs(value-native[f]))/max(1., np.max(np.abs(value))) for value, f in [(L0, 'old'), (L1, 'new'), (ideal, 'ideal')])
    assert load_error < 1e-11
    assert np.max(np.abs(L1-ideal))/np.max(np.abs(ideal)) < .005
    assert np.max(np.abs(L1-L0-((overlap-L0)+activation+(L1-ideal)))) < 1e-10
    if native_after:
        ledger0=csv(path/'old-ledger.csv')[0];ledger1=csv(path/'new-ledger.csv')[0]
        assert abs(E0-ledger0['bulk'])<1e-11*E0 and abs(E1-ledger1['bulk'])<1e-11*E1
        assert abs(storage(old)-ledger0['cohesive'])<1e-11 and abs(storage(new)-ledger1['cohesive'])<1e-11
        assert ledger0['fracture']==ledger1['fracture']
        assert np.array_equal(csv(path/'old-phase.csv'), csv(path/'new-phase.csv'))
    else:
        native_metrics = csv(path/'native-metrics.csv')[0]
        assert abs(E0-native_metrics['bulk_old']) < 1e-11*E0
        assert abs(E1-native_metrics['bulk_new']) < 1e-11*E1
    assert error < .005 and abs(E1-E0)/E0 < .005
    # Initial continuum-vs-Q1 representation is separate from remap error.
    init_error = np.sqrt(np.sum(volume*(phase(x,y)-continuum(x,y))**2)/np.sum(volume*continuum(x,y)**2))
    result = dict(scope='synthetic native-component geometry transfer; no physical time/phase evolution or finite-tip mechanics',
                  constitutive_m=M, ranks=len(list(path.glob('old-cells-*.csv'))), cells=len(cells), tensor_DoFs=3*9*len(cells),
                  seed_compact_radius=float(R[-1]), initial_relative_phase_L2=float(init_error),
                  initial_tip=.5, appended_tip=.75,
                  continuum_zero_phase_region_beyond=float(.5+R[-1]),
                  physical_zero_phase_region_beyond=float(np.ceil((.5+R[-1])*NX)/NX),
                  normalization_relative_error=float(Ierror), minimum_normalization=float(np.min(new['I'])),
                  maximum_old_normalization_change=float(np.max(np.abs(new['I'][:len(old)]-old['I']))),
                  independent_projection_relative_error=float(projection_error),
                  complete_Q2_reproduction_relative_error=float(reproduction_error),
                  maximum_cell_moment_defect=float(np.max(np.abs(moments))),
                  unchanged_T_maximum_cell_mean_defect=float(np.max(np.abs(np.sum(volume*delta,axis=1)))),
                  unchanged_T_maximum_full_tensor_difference=float(np.sqrt(2)*np.max(np.abs(delta))),
                  maximum_T_coefficient_change=float(np.max(np.abs(b-a))),
                  reconstructed_full_tensor_relative_L2=float(error), unchanged_T_negative_control_relative_L2=float(naive),
                  localized_full_tensor_relative_L2=float(localized_error),
                  reconstructed_full_tensor_maximum_difference=float(np.sqrt(np.max(inner(difference)))),
                  bulk_energy_old=float(E0), bulk_energy_new=float(E1), bulk_energy_transfer_defect=float(E1-E0),
                  cohesive_energy_old=float(storage(old)), cohesive_energy_new=float(storage(new)),
                  prescribed_geometry_cohesive_energy_change=float(storage(new)-storage(old)),
                  new_surface_cohesive_energy=0., fracture_energy_change=0., phase_obstacle_change=0.,
                  weak_load_independent_relative_error=float(load_error),
                  weak_load_transfer_relative_defect=float(np.max(np.abs(L1-ideal))/np.max(np.abs(ideal))),
                  integrated_old_weak_shear=float(np.sum(L0)), integrated_new_weak_shear=float(np.sum(L1)),
                  intended_overlap_measure_change=float(np.sum(overlap-L0)), intended_new_surface_shear=float(np.sum(activation)),
                  weak_shear_transfer_defect=float(np.sum(L1-ideal)),
                  intended_new_surface_friction_load=1000.*.25,
                  old_material_and_histories_preserved=True, birth_histories_zero=True, restart_records_exact=True,
                  physical_steps_advanced=0, geometry_publications=1, strict_reference_match_passed=False,
                  finite_tip_evolution_qualified=False, virgin_phase_activation_qualified=False)
    if native_after:
        result['scope']='native material DG / manager histories / physical obstacle / simulator checkpoint, zero-time prescribed append'
        result['native_restart_before_append_and_after_commit_exact']=True
        for state,loads in [(old,native0),(new,native1)]:
            z,w=gauss(16)
            Z=(1-z)*state['Z'][:-1,None]+z*state['Z'][1:,None]
            I=(1-z)*state['I'][:-1,None]+z*state['I'][1:,None]
            reference=np.sum(np.diff(state['x'])[:,None]*w*Z/I)
            assert abs(np.sum(loads['cohesive'])-reference)<1e-11*max(1.,abs(reference))
            assert abs(np.sum(loads['friction'])-1000*(state['x'][-1]-state['x'][0]))<1e-11
        result['native_cohesive_weak_change']=float(np.sum(native1['cohesive'])-np.sum(native0['cohesive']))
        result['native_residual_weak_change']=float(np.sum(native1['residual'])-np.sum(native0['residual']))
        result['native_fracture_storage']=float(ledger0['fracture'])
        result['native_physical_stress_maximum_disagreement']=0.
        for name,coefs in [('old',a),('new',b)]:
            points=csv(path/f'{name}-stress.csv')
            lookup={row['cell']:i for i,row in enumerate(cells)}
            ci=np.array([lookup[row['cell']] for row in points])
            u=(points['x']-cells['x0'][ci])/cells['hx'][ci];v=(points['y']-cells['y0'][ci])/cells['hy'][ci]
            sh=lagrange(u)[:,np.rint(nodes['u']*2).astype(int)]*lagrange(v)[:,np.rint(nodes['v']*2).astype(int)]
            ref=np.einsum('qi,qik->qk',sh,coefs[ci])
            state=old if name=='old' else new
            ref[:,2]-=evaluate_profile(state,points['x'],points['y'])[3]
            observed=np.stack([points[f] for f in ['tauxx','tauyy','tauxy']],axis=-1)
            mismatch=float(np.max(np.abs(ref-observed)))
            assert mismatch<1e-11*np.max(np.abs(ref))
            result['native_physical_stress_maximum_disagreement']=max(result['native_physical_stress_maximum_disagreement'],mismatch)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare', type=Path, help='Compare persistent fields with another rank layout')
    parser.add_argument('--native-after-restart',type=Path)
    parser.add_argument('--native-before-restart',type=Path)
    args = parser.parse_args()
    result = assess(args.directory,args.native_after_restart,args.native_before_restart)
    if args.compare:
        for name in ['old', 'new', 'gauss12', 'restart']:
            a, cells, T = load_state(args.directory, name)
            b, other, U = load_state(args.compare, name)
            assert np.array_equal(a,b) and np.array_equal(cells,other) and np.array_equal(T,U)
        for name in ['before.archive', 'after.archive']:
            assert (args.directory/name).read_bytes() == (args.compare/name).read_bytes()
        result['rank_comparison'] = dict(all_persistent_fields_exact=True, before_and_after_archives_identical=True,
                                         other_ranks=len(list(args.compare.glob('old-cells-*.csv'))))
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
