"""Matched, unaveraged native-QP comparison; run after A/B/C on 1 and 2 ranks."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analyze_moment_cycle import samples
from analyze_inclined_moment import independent_jump

ROOT = Path(__file__).resolve().parent
T = np.array([.5, np.sqrt(3)/2])
N = np.array([-np.sqrt(3)/2, .5])


def csv(path):
    return np.atleast_1d(np.genfromtxt(path, delimiter=',', names=True, dtype=None, encoding=None))


def particle_data(out, stage, step):
    p = np.concatenate([csv(f) for f in out.glob(f'clean_{stage}_{step}_rank*.csv')])
    return np.sort(p, order='id')


def rms(v, weights):
    return float(np.sqrt(np.sum(weights*v*v)/np.sum(weights)))


def tensor_rms(t, w):
    return rms(np.sqrt(t[:, 0]**2+t[:, 1]**2+2*t[:, 2]**2), w)


def traces(data, values):
    """Exact Q2 FE-history traces, not visualization smoothing of DG values.

    Recover the tensor-product polynomial from its native 3x3 Gauss values;
    evaluate each side independently at three face Gauss points.
    """
    cell = np.floor((data[:, :2]+.5)*64).astype(int)
    order = np.lexsort((data[:, 1], data[:, 0], cell[:, 1], cell[:, 0]))
    ids = cell[order].reshape(64, 64, 9, 2)
    assert np.all(ids == ids[:, :, :1, :])
    r = ((data[order, :2]+.5)*64-cell[order]).reshape(-1, 9, 2)
    basis = np.array([(i,j) for i in range(3) for j in range(3)])
    vandermonde = r[:, :, 0, None]**basis[:, 0]*r[:, :, 1, None]**basis[:, 1]
    coeff = np.linalg.solve(vandermonde, values[order].reshape(-1, 9, 3)).reshape(64,64,9,3)
    gauss = (np.polynomial.legendre.leggauss(3)[0]+1)/2
    result = []; interior_max = []; interior_sum = 0.; interior_weight = 0.
    for axis in range(2):
        sides = []
        for side in (0., 1.):
            xy = np.column_stack((np.full(3, side), gauss)) if axis == 0 else np.column_stack((gauss, np.full(3, side)))
            b = xy[:, 0, None]**basis[:, 0]*xy[:, 1, None]**basis[:, 1]
            sides.append(np.einsum('qk,ijkc->ijqc', b, coeff))
        jump = sides[1][:-1]-sides[0][1:] if axis == 0 else sides[1][:,:-1]-sides[0][:,1:]
        result.append(float(np.max(np.abs(jump))))
        if axis == 0:
            x = np.broadcast_to((np.arange(1,64)/64-.5)[:,None,None],(63,64,3))
            y = np.broadcast_to(((np.arange(64)[:,None]+gauss)/64-.5)[None,:,:],(63,64,3))
        else:
            x = np.broadcast_to(((np.arange(64)[:,None]+gauss)/64-.5)[:,None,:],(64,63,3))
            y = np.broadcast_to((np.arange(1,64)/64-.5)[None,:,None],(64,63,3))
        mask = (abs(T[0]*x+T[1]*y)<=.2)&(abs(N[0]*x+N[1]*y)<=.1)
        w = np.broadcast_to(np.polynomial.legendre.leggauss(3)[1]/128,mask.shape)[mask]
        interior_max.append(float(np.max(abs(jump[mask]))))
        interior_sum += np.sum(w*(jump[mask,0]**2+jump[mask,1]**2+2*jump[mask,2]**2))
        interior_weight += np.sum(w)
    return dict(max_abs_component_face_jump=max(result), x_faces=result[0], y_faces=result[1],
                interior_max_component=max(interior_max), interior_tensor_rms=float(np.sqrt(interior_sum/interior_weight)))


def main():
    report = {'definition': 'Fresh prescribed-slip inclined diagnostic, ordinary particle advection/history; no analytic exact solution.',
              'cases': {}, 'mpi': {}, 'comparisons': {}}
    saved = {}; profiles = {}; hashes = []; coefficient_base = None
    for ranks in (1,2):
        for case in 'ABC':
            label = f'{case}-'+('one' if ranks == 1 else 'two')
            out = ROOT/label
            s = csv(out/'summary.csv')
            assert np.array_equal(s['step'], np.arange(5))
            assert np.all(s['relative'] < 1e-8) and np.all(s['dt'] == .1)
            assert np.max(abs(csv(out/'boundary_flux.csv')['total'])) < 1e-15
            hashes.append((out/'initial_hash.txt').read_text().strip())
            log = (ROOT/f'{label}.log').read_text()
            fresh = re.findall(r'Fault linear solve: iterations=(\d+), fresh=([^, ]+), target=([^, ]+)', log)
            assert len(fresh) >= 5 and all(float(a)<=float(b) for _,a,b in fresh)
            info = dict(fe=(out/'finite_elements.txt').read_text(), fresh_linear_checks=len(fresh), steps=[])
            elapsed = re.search(r'([\d.]+) total', log)
            info['elapsed_seconds'] = float(elapsed[1]) if elapsed else None
            report['cases'][label] = info
            p0 = particle_data(out,'after',0)
            assert np.max([np.max(abs(p0[c])) for c in ('xx','yy','xy')]) == 0
            g = csv(out/'geometry.csv'); nodes = np.column_stack((g['x'],g['y']))@T
            for step in range(5):
                d = samples(out,'fields',step,15); h = samples(out,'histories',step,9)
                coef = samples(out,'coefficients',step,7)
                if coefficient_base is None: coefficient_base = coef.copy()
                np.testing.assert_allclose(coef,coefficient_base,rtol=2e-13,atol=1e-14)
                np.testing.assert_array_equal(d[:,:3],h[:,:3])
                saved[case,ranks,step] = d,h
                pa = particle_data(out,'after',step)
                if step:
                    before = particle_data(out,'before',step)
                    preceding = particle_data(out,'after',step-1)
                    np.testing.assert_array_equal(before,preceding)
                    assert abs(independent_jump(d,h[:,3:6]-d[:,9:12])-float(s['Jtotal'][step])) < 2e-10
                displacement = np.sqrt((pa['x']-p0['x'])**2+(pa['y']-p0['y'])**2)
                assert np.array_equal(pa['id'],p0['id'])
                record = dict(step=step, time=float(s['time'][step]), Jtotal=float(s['Jtotal'][step]),
                              Fcurrent=float(s['Fcurrent'][step]), relative_residual=float(s['relative'][step]),
                              interior_transfer_weak_jump=independent_jump(d,h[:,3:6]-d[:,9:12],interior=True),
                              newton=int(s['newton'][step]), krylov=int(s['krylov'][step]), alpha=float(s['alpha'][step]),
                              current_stress_rms=tensor_rms(d[:,9:12],d[:,2]),
                              transfer_stress_change_rms=tensor_rms(h[:,3:6]-d[:,9:12],d[:,2]),
                              transfer_mean_tensor_change=np.average(h[:,3:6]-d[:,9:12],axis=0,weights=d[:,2]).tolist(),
                              max_particle_displacement=float(max(displacement)),
                              min_particles_per_cell=int(np.min(np.unique(pa['cell'],return_counts=True)[1])),
                              next_history_traces=traces(h,h[:,3:6]))
                if step:
                    record['stress_change_from_initial_rms'] = tensor_rms(d[:,9:12]-saved[case,ranks,0][0][:,9:12],d[:,2])
                info['steps'].append(record)
                raw = np.concatenate([csv(f) for f in out.glob(f'traction_{step}_rank*.csv')])
                mass = np.zeros(len(nodes)); rhs = np.zeros((len(nodes),3))
                for index,basis in ((raw['segment'].astype(int),1-raw['xi']), (raw['segment'].astype(int)+1,raw['xi'])):
                    w = raw['w']*raw['chi']*basis
                    np.add.at(mass,index,w)
                    for k,name in enumerate(('p','d','sigma')): np.add.at(rhs[:,k],index,w*raw[name])
                v = rhs/mass[:,None]
                np.testing.assert_allclose(v[:,0]+v[:,1],v[:,2],atol=1e-10)
                profiles[case,ranks,step] = nodes,mass,v
                mask=abs(nodes)<=.2
                record['interior_native_normal_mean']=float(np.average(v[mask,2],weights=mass[mask]))
                # Neighbouring-chord defect is a roughness diagnostic, not an exact-solution error.
                chord=v[1:-1,2]-.5*(v[:-2,2]+v[2:,2])
                record['interior_native_normal_chord_rms']=rms(chord[mask[1:-1]],mass[1:-1][mask[1:-1]])
                if ranks == 1:
                    np.savetxt(out/f'weak_profile_{step}.csv',np.column_stack((nodes,mass,v)),delimiter=',',
                               header='s,work_mass,p,d,sigma',comments='')
            assert info['steps'][-1]['max_particle_displacement'] > 1e-4
    assert len(set(hashes)) == 1
    report['common_initial_particle_hash'] = hashes[0]
    report['coefficient_agreement'] = 'All native x,y,phi,chi,V,beta,kappa samples agree at rtol=2e-13, atol=1e-14 across steps/cases/ranks.'
    for case in 'ABC':
        comparisons = []
        for step in range(5):
            d,h = saved[case,1,step]; d2,h2 = saved[case,2,step]
            # Existing nonlinear target: compare errors against field scales, not tiny point zeros.
            # Timestep zero retains zero particle stress. Report its evaluated
            # stress separately: differentiating the independently converged
            # velocity amplifies the rank-dependent initial solve error.
            checks = [(d[:,3:5],d2[:,3:5]), (h[:,3:6],h2[:,3:6])]
            if step: checks.append((d[:,9:],d2[:,9:]))
            for a,b in checks:
                assert np.max(abs(a-b)) <= 1e-8*max(np.max(abs(a)),1e-8)+1e-8
            pa=particle_data(ROOT/f'{case}-one','after',step)
            pb=particle_data(ROOT/f'{case}-two','after',step)
            np.testing.assert_array_equal(pa['id'],pb['id'])
            stress_error=max(float(max(abs(pa[c]-pb[c]))) for c in ('xx','yy','xy'))
            assert stress_error < 1e-8*max(max(abs(pa[c])) for c in ('xx','yy','xy'))+1e-8
            comparisons.append(dict(step=step, velocity_max=float(np.max(abs(d[:,3:5]-d2[:,3:5]))),
                                    current_tensor_max=float(np.max(abs(d[:,9:12]-d2[:,9:12]))),
                                    current_tensor_relative_l2=float(np.linalg.norm(d[:,9:12]-d2[:,9:12])/np.linalg.norm(d[:,9:12])),
                                    next_history_tensor_max=float(np.max(abs(h[:,3:6]-h2[:,3:6]))),
                                    particle_tensor_max=stress_error))
            if step==0:
                grad=d[:,5:9]-d2[:,5:9]
                predicted=2*coefficient_base[0,6]*np.column_stack((grad[:,0],grad[:,3],.5*(grad[:,1]+grad[:,2])))
                error=float(np.max(abs(predicted-(d[:,9:12]-d2[:,9:12]))))
                assert error<1e-8
                comparisons[-1]['initial_stress_gradient_difference_identity_error']=error
        report['mpi'][case] = comparisons
    for case in 'BC':
        for step in (0,1): np.testing.assert_array_equal(saved[case,1,step][0],saved['A',1,step][0])
        report['comparisons'][f'{case}-A'] = []
        for step in range(5):
            a = saved['A',1,step][0]; b = saved[case,1,step][0]
            nodes,mass,v = profiles[case,1,step]; av = profiles['A',1,step][2]
            interior = abs(nodes)<=.2
            report['comparisons'][f'{case}-A'].append(dict(step=step,
                velocity_rms=rms(np.linalg.norm(b[:,3:5]-a[:,3:5],axis=1),a[:,2]),
                current_stress_rms=tensor_rms(b[:,9:12]-a[:,9:12],a[:,2]),
                weak_sigma_difference_mean=float(np.average((v-av)[interior,2],weights=mass[interior])),
                weak_sigma_difference_rms=rms((v-av)[interior,2],mass[interior])))
    (ROOT/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    plot(report,saved,profiles)
    print(json.dumps({k:report[k] for k in ('mpi','comparisons')},indent=2))


def plot(report,saved=None,profiles=None):
    if saved is None:
        saved={};profiles={}
        for case in 'ABC':
            out=ROOT/f'{case}-one'
            saved[case,1,4]=(samples(out,'fields',4,15),None)
            p=csv(out/'weak_profile_4.csv')
            profiles[case,1,4]=(p['s'],p['work_mass'],np.column_stack([p[k] for k in ('p','d','sigma')]))
    fig,axes = plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for case in 'ABC':
        steps=report['cases'][f'{case}-one']['steps']
        axes[0,0].plot([s['time'] for s in steps[1:]],[s['Jtotal'] for s in steps[1:]],'o-',label=case)
        axes[0,1].plot([s['time'] for s in steps[1:]],[s['interior_transfer_weak_jump'] for s in steps[1:]],'o-',label=case)
        d=saved[case,1,4][0]; sel=(abs(d[:,0])<.5/64)&(abs(d[:,1])<.3)
        axes[0,2].plot(d[sel,1],d[sel,9],'.',label=case)
        nodes,mass,v=profiles[case,1,4];sel=abs(nodes)<=.2
        for j in range(3): axes[1,j].plot(nodes[sel],v[sel,j],'o-',label=case)
    titles=['Whole-domain transfer weak-load jump (Pa m)','Interior transfer weak-load jump (Pa m)',
            'Unaveraged native-QP tau_xx near x=0 (Pa)','Work-row pressure (Pa)','Work-row deviatoric normal traction (Pa)','Work-row total normal traction (Pa)']
    for ax,title in zip(axes.flat,titles):ax.set_title(title,fontsize=10);ax.grid(alpha=.25);ax.legend()
    axes[0,0].set_xlabel('t (s)');axes[0,1].set_xlabel('t (s)');axes[0,2].set_xlabel('y (m)')
    for ax in axes[1]:ax.set_xlabel('along-fault s from centre (m)')
    fig.savefig(ROOT/'comparison.png',dpi=160)


if __name__ == '__main__':
    if '--plot-only' in sys.argv: plot(json.loads((ROOT/'comparison.json').read_text()))
    else: main()
