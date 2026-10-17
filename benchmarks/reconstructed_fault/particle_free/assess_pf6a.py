#!/usr/bin/env python3
"""Independent PF6a endpoint/source/FE-energy/work and bounded refinement checks."""
import argparse
import json
from pathlib import Path
import numpy as np
from assess_phase_energy import Reference, csv, gauss, q1, lagrange2, scaled_error


def norm2(a):
    return a[..., 0]**2+a[..., 1]**2+(a[..., 0]+a[..., 1])**2+2*a[..., 2]**2


def inner(a, b):
    return a[..., 0]*b[..., 0]+a[..., 1]*b[..., 1]+(a[..., 0]+a[..., 1])*(b[..., 0]+b[..., 1])+2*a[..., 2]*b[..., 2]


class Ref(Reference):
    def bulk_rule(self, order=None):
        if order is None or order in self.cache:
            return super().bulk_rule(order)
        # Generate the actual bulk/surface intersection partition independently.
        count = round(self.cells['dx'][0]*(self.ns-1))
        a, wa = gauss(6); b, wb = gauss(order)
        x = np.concatenate([(i+a)/count for i in range(count)])
        wx = np.tile(wa/count, count)
        r, s = np.repeat(x, len(b)), np.tile(b, len(x))
        w = np.repeat(wx, len(b))*np.tile(wb, len(x))
        N = q1(r, s)
        ix = np.repeat(self.indices[:, None, :], len(r), axis=1).reshape(-1, 4)
        nn = np.tile(N, (len(self.cells), 1))
        grad = np.empty((len(self.cells), len(r), 4, 2))
        grad[:, :, :, 0] = np.array([-(1-s), 1-s, -s, s]).T[None]/self.cells['dx'][:, None, None]
        grad[:, :, :, 1] = np.array([-(1-r), -r, 1-r, r]).T[None]/self.cells['dy'][:, None, None]
        weights = (self.cells['dx']*self.cells['dy'])[:, None]*w[None]
        xx = self.cells['x0'][:, None]+self.cells['dx'][:, None]*r[None]
        segment = np.minimum(self.ns-2, (xx*(self.ns-1)).astype(int)).ravel()
        xi = (xx*(self.ns-1)).ravel()-segment
        sf = np.zeros((len(segment), self.ns)); sf[np.arange(len(segment)), segment] = 1-xi
        sf[np.arange(len(segment)), segment+1] = xi
        T = self.tensor_values(r, s)
        result = ix, nn, grad.reshape(-1, 4, 2), weights.ravel(), sf, T.reshape(-1, 3)
        self.cache[order] = result
        return result

    def tensor_values(self, r, s):
        out = np.zeros((len(self.cells), len(r), 3))
        lr, ls = lagrange2(r), lagrange2(s)
        for c in range(len(self.cells)):
            for a in self.tensor[self.tensor['cell'] == c]:
                out[c] += (lr[:, round(2*a['r'])]*ls[:, round(2*a['s'])])[:, None]*np.array([a['xx'], a['yy'], a['xy']])
        return out


def grids(table, names, cells):
    out = np.zeros((cells, 3, 3, len(names)))
    for a in table:
        out[int(a['cell']), round(2*a['r']), round(2*a['s'])] = [a[n] for n in names]
    return out


def evaluate_grid(g, r, s):
    return np.einsum('qi,qj,cijk->cqk', lagrange2(r), lagrange2(s), g)


def strains(ref, velocity, r, s):
    # Differentiate the exported native Q2 velocity independently, including zz.
    dr = np.array([4*r-3, 4-8*r, 4*r-1]).T
    ds = np.array([4*s-3, 4-8*s, 4*s-1]).T
    ux = np.einsum('qi,qj,cijk->cqk', dr, lagrange2(s), velocity)/ref.cells['dx'][:, None, None]
    uy = np.einsum('qi,qj,cijk->cqk', lagrange2(r), ds, velocity)/ref.cells['dy'][:, None, None]
    trace = ux[..., 0]+uy[..., 1]
    return np.stack([ux[..., 0]-trace/3, uy[..., 1]-trace/3, .5*(ux[..., 1]+uy[..., 0])], axis=-1), trace


def ledger(ref, old, velocity, pressure, rates, dt, time_order):
    # Diagnostic continuous path: linear physical phase, exact Maxwell T/Z,
    # and fixed endpoint rates. This is accounting, not a phase-first update.
    rule = csv(ref.path/'rule.csv'); r, s = rule['r'], rule['s']
    E, div = strains(ref, velocity, r, s)
    boundary = velocity.copy()
    for c, cell in enumerate(ref.cells):
        for i in range(3):
            for j in range(3):
                x = cell['x0']+.5*i*cell['dx']; y = cell['y0']+.5*j*cell['dy']
                if y not in (0., 1.):
                    boundary[c, i, j, 0] = 0.
                    if x not in (0., 1.): boundary[c, i, j, 1] = 0.
    ED, divD = strains(ref, boundary, r, s)
    E, div, ED, divD = E.reshape(-1, 3), div.ravel(), ED.reshape(-1, 3), divD.ravel()
    pressure = evaluate_grid(pressure, r, s).ravel()
    Tn = old.tensor_values(r, s).reshape(-1, 3)
    Zn = old.surface['Z'].copy(); finalZ = ref.surface['Z'].copy()
    pn, pe = old.nodes['phase'], ref.nodes['phase']; pdot = (pe-pn)/dt
    geometry = ref.bulk_rule(); ix, N, gradN, w, sf, _ = geometry
    V = sf@rates; vs = ref.sf@rates; ws = ref.profiles['weight']
    totals = dict(boundary_work=0., free_mechanical_work=0., pressure_work=0.,
                  bulk_viscous=0., cohesive_viscous=0., friction=0.,
                  contact_residual_work=0., phase_work=0., absolute_free_mechanical_work=0.,
                  absolute_contact_work=0., absolute_phase_work=0.)
    a, wa = gauss(time_order)
    initial_energy = old.evaluate(pn)['energy']; final_energy = ref.evaluate(pe)['energy']
    for t, wt in zip(dt*a, dt*wa):
        beta = np.exp(-t/100); kappa = -1e8*np.expm1(-t/100)
        phase = pn+t*pdot; T = beta*Tn+2*kappa*E; Z = beta*Zn+kappa*rates
        ref.surface['Z'] = Z
        ref.cache[None] = (*geometry[:-1], T)
        e = ref.evaluate(phase)
        I = e['I']; ii, z = sf@I, sf@Z; fs, zs = ref.sf@I, ref.sf@Z
        phi = np.sum(phase[ix]*N, axis=1); chi = ref.m*phi*(1+phi)/(1-phi)**2/ii
        tau = T.copy(); tau[:, 2] -= z*chi
        boundary_work = np.sum(w*(inner(tau, ED)-pressure*divD))
        free = np.sum(w*(inner(tau, E-ED)-pressure*(div-divD)))
        pressure_work = np.sum(w*pressure*div)
        db = np.sum(w*norm2(tau))/(2e8)
        dc = np.sum(ws*zs*zs/fs)/1e8
        friction = np.sum(ws*1000*vs)
        contact = np.sum(ws*zs*vs/fs)+friction-np.sum(w*chi*V*tau[:, 2])
        phase_work = e['gradient']@pdot
        values = [boundary_work, free, pressure_work, db, dc, friction, contact, phase_work,
                  abs(free), abs(contact), abs(phase_work)]
        for key, value in zip(totals, values): totals[key] += float(wt*value)
    ref.surface['Z'] = finalZ; ref.cache[None] = geometry
    change = final_energy-initial_energy
    remainder = change-totals['boundary_work']+totals['bulk_viscous']+totals['cohesive_viscous']+totals['friction']
    identity = remainder-totals['free_mechanical_work']-totals['pressure_work']-totals['contact_residual_work']-totals['phase_work']
    return dict(storage_change=float(change), external_remainder=float(remainder),
                identity_remainder=float(identity), **totals)


def state(path, previous=None, dt=0., work=False):
    ref = Ref(path); p = ref.nodes['phase']; native = csv(path/'gradient.csv')
    e = ref.evaluate(p); high = ref.evaluate(p, 24); finer = ref.evaluate(p, 32)
    physical = csv(path/'physical.csv'); ix, N, _, w, sf, T = ref.bulk_rule()
    phi = np.sum(p[ix]*N, axis=1); chi = ref.m*phi*(1+phi)/(1-phi)**2/(sf@e['I'])
    stress = T.copy(); stress[:, 2] -= (sf@ref.surface['Z'])*chi
    observed = np.stack([physical[k] for k in ['xx', 'yy', 'xy']], axis=-1)
    out = dict(energy=float(e['energy']), energy_error=0.,
               gradient_error=scaled_error(e['gradient'], native['gradient']),
               normalization_error=scaled_error(e['I'], ref.surface['I_accepted']),
               physical_stress_error=scaled_error(stress, observed), localization_error=scaled_error(chi, physical['chi']),
               quadrature_energy_error=abs(e['energy']-high['energy'])/max(1., abs(high['energy'])),
               quadrature_load_error=scaled_error(e['loads'], high['loads']),
               independent_quadrature_uncertainty=max(abs(high['energy']-finer['energy'])/max(1., abs(finer['energy'])), scaled_error(high['loads'], finer['loads'])),
               high_rule_KKT=float(np.max(abs(np.where(p==native['lower'], np.minimum(high['gradient'], 0), high['gradient'])))))
    velocity = grids(csv(path/'velocity.csv'), ['ux', 'uy'], len(ref.cells))
    if previous is not None:
        old = Ref(previous)
        surface = csv(path/'accepted-surface.csv'); V = surface['V']; Z = surface['Z']
        rr = np.repeat(np.array([0., .5, 1.]), 3); ss = np.tile(np.array([0., .5, 1.]), 3)
        E, _ = strains(ref, velocity, rr, ss)
        expected = np.exp(-dt/100)*old.tensor_values(rr, ss)+2*(-1e8*np.expm1(-dt/100))*E
        out['native_source_recurrence_error'] = scaled_error(ref.tensor_values(rr, ss), expected)
        out['cohesive_recurrence_error_Pa_m'] = float(max(abs(Z-np.exp(-dt/100)*old.surface['Z']+1e8*np.expm1(-dt/100)*V)))
        loads = csv(path/'native-loads.csv')
        normal = ref.sf.T@(ref.profiles['weight']*10000)
        cohesive = ref.sf.T@(ref.profiles['weight']*(ref.sf@Z)/(ref.sf@e['I']))
        out['native_weak_load_error'] = dict(shear=scaled_error(e['loads'], loads['shear']), normal=scaled_error(normal, loads['normal']),
                                            cohesive=scaled_error(cohesive, loads['cohesive']), friction=scaled_error(.1*normal, loads['friction']))
        out['mechanical'] = {k:float(csv(path/'mechanical.csv')[k]) for k in ['bulk_norm', 'surface_rms']}
        if work:
            pressure = grids(csv(path/'velocity.csv'), ['p'], len(ref.cells))
            out['ledger'] = ledger(ref, old, velocity, pressure, V, dt, 12)
            refined = ledger(ref, old, velocity, pressure, V, dt, 24)
            out['ledger_time_quadrature_uncertainty'] = max(abs(out['ledger'][k]-refined[k]) for k in refined)
    for k in ['gradient_error','normalization_error','physical_stress_error','localization_error','independent_quadrature_uncertainty']:
        assert out[k]<1e-9,(path,k,out[k])
    assert out['quadrature_energy_error']<1e-6 and out['quadrature_load_error']<1e-6,(path,out)
    if 'native_weak_load_error' in out: assert max(out['native_weak_load_error'].values())<1e-9,(path,out)
    if work:
        assert abs(out['ledger']['identity_remainder'])<1e-7,(path,out['ledger'])
    return out


def operators(path):
    ref = Ref(path); p = ref.nodes['phase']; native = csv(path/'native.csv'); e=ref.evaluate(p)
    native_energy=csv(path/'energy.csv'); actions=np.genfromtxt(path/'actions.csv',delimiter=',',names=True,dtype=None,encoding='utf-8')
    directions=[ref.nodes['d1'],ref.nodes['d2']]; result={}
    result['gradient_error']=scaled_error(e['gradient'],native['gradient'])
    result['energy_error']=max(abs(e[k]-native_energy[k])/max(1.,abs(e[k])) for k in ['bulk','cohesive','fracture'])
    result['I_error']=scaled_error(e['I'],native['I'][:ref.ns]); result['actions']=[]
    for j,d in enumerate(directions):
        independent=ref.evaluate(p+1j*1e-30*d)['gradient'].imag/1e-30
        actual=actions[(actions['kind']=='fixed')&(actions['direction']==j)]['value']
        error=scaled_error(actual,independent);assert error<1e-9
        result['actions'].append(float(error))
    H=[actions[(actions['kind']=='reduced')&(actions['direction']==j)]['value'] for j in [0,1]]
    result['reduced_symmetry']=float(abs(directions[1]@H[0]-directions[0]@H[1])/max(1.,abs(directions[1]@H[0]),abs(directions[0]@H[1])))
    assert max(result[k] for k in ['gradient_error','energy_error','I_error'])<1e-9 and result['reduced_symmetry']<1e-10,result
    return result


def run(path, work=False, restart_parent=None):
    record=json.loads((path/'record.json').read_text());assert record['gate_passed']
    root=path/'run'; metrics=np.atleast_1d(csv(root/'metrics.csv')); result={'label':path.name,'metrics':[], 'seconds':record['seconds'],'max_rss_KiB':record['max_rss_KiB']}
    for row in metrics:
        step=int(row['step']); previous=root/f'state-{step-1}' if step else None
        if previous and not previous.exists():
            previous=restart_parent/'run'/f'state-{step-1}' if restart_parent else None
        out=state(root/f'state-{step}',previous,float(row['dt']),work and previous is not None)
        nativeE=float(row['bulk']+row['cohesive']+row['fracture']);out['energy_error']=abs(out['energy']-nativeE)/max(1.,abs(nativeE));assert out['energy_error']<1e-9
        result['metrics'].append({k:float(row[k]) for k in metrics.dtype.names}|out)
    if (root/'operators').exists(): result['operators']=operators(root/'operators')
    gates=np.genfromtxt(root/'gates.csv',delimiter=',',skip_header=1,dtype=None,encoding='utf-8')
    assert all(bool(r[-1]) for r in gates)
    result['native_gates']=len(gates);result['checked']=True
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('runs',nargs='+',type=Path);p.add_argument('--work',action='store_true');p.add_argument('--matrix',action='store_true');p.add_argument('--contact-only',action='store_true');p.add_argument('--restart-parent',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    if a.contact_only:
        result={'runs':{path.name:contact_checks(path) for path in a.runs}, 'passed':True}
        a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return
    result={'runs':[run(path,a.work,a.restart_parent) for path in a.runs], 'implementation_checks_passed':True}
    if a.matrix: result['comparisons']=matrix_comparisons(a.runs)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))



# Importable postprocessing utilities for comparisons on common physical points.
def samples_at(ref, x, y):
    nx=round(1/ref.cells['dx'][0]);ny=round(1/ref.cells['dy'][0])
    lookup=np.empty((ny,nx),dtype=int)
    lookup[np.rint(ref.cells['y0']*ny).astype(int),np.rint(ref.cells['x0']*nx).astype(int)]=np.arange(len(ref.cells))
    c=lookup[np.clip((y*ny).astype(int),0,ny-1),np.clip((x*nx).astype(int),0,nx-1)]
    r=(x-ref.cells['x0'][c])/ref.cells['dx'][c];s=(y-ref.cells['y0'][c])/ref.cells['dy'][c]
    phase=np.sum(ref.nodes['phase'][ref.indices[c]]*q1(r,s),axis=1)
    g=grids(ref.tensor,['xx','yy','xy'],len(ref.cells))
    T=np.einsum('qi,qj,qijk->qk',lagrange2(r),lagrange2(s),g[c])
    k=np.minimum(ref.ns-2,(x*(ref.ns-1)).astype(int));xi=x*(ref.ns-1)-k
    I=(1-xi)*ref.surface['I_accepted'][k]+xi*ref.surface['I_accepted'][k+1]
    Z=(1-xi)*ref.surface['Z'][k]+xi*ref.surface['Z'][k+1]
    chi=ref.m*phase*(1+phase)/(1-phase)**2/I
    tau=T.copy();tau[:,2]-=Z*chi
    return phase,T,tau


def difference(first,second,step_first,step_second,order=12):
    a,b=Ref(first/'run'/f'state-{step_first}'),Ref(second/'run'/f'state-{step_second}')
    nx=round(1/min(a.cells['dx'][0],b.cells['dx'][0]));ny=round(1/min(a.cells['dy'][0],b.cells['dy'][0]))
    q,w=gauss(order);x=np.repeat((np.arange(nx)[:,None]+q)/nx,ny*order).reshape(nx,order,ny,order).ravel()
    y=np.broadcast_to(((np.arange(ny)[:,None]+q)/ny)[None,None],(nx,order,ny,order)).ravel()
    weights=np.broadcast_to((w[:,None,None]*w[None,None,:]/(nx*ny))[None],(nx,order,ny,order)).ravel()
    aa,bb=samples_at(a,x,y),samples_at(b,x,y)
    result={}
    for i,name in enumerate(['phase','T','stress']):
        square=(aa[i]-bb[i])**2 if i==0 else norm2(aa[i]-bb[i])
        amplitude=bb[i]**2 if i==0 else norm2(bb[i])
        result[name+'_L2_absolute']=float(np.sqrt(np.sum(weights*square)))
        result[name+'_L2_relative']=float(np.sqrt(np.sum(weights*square)/np.sum(weights*amplitude))) if np.sum(weights*amplitude)>0 else None
    sa,sb=csv(a.path/'accepted-surface.csv'),csv(b.path/'accepted-surface.csv')
    for key in ['I','Z','V','slip']:
        result[key+'_relative_l2']=float(np.linalg.norm(sa[key]-sb[key])/max(1e-300,np.linalg.norm(sb[key])))
    result['q_relative_l2']=float(np.linalg.norm(sa['Z']/sa['I']-sb['Z']/sb['I'])/max(1e-300,np.linalg.norm(sb['Z']/sb['I'])))
    if (a.path/'native-loads.csv').exists() and (b.path/'native-loads.csv').exists():
        la,lb=csv(a.path/'native-loads.csv'),csv(b.path/'native-loads.csv')
        result['weak_shear_relative_l2']=float(np.linalg.norm(la['shear']-lb['shear'])/np.linalg.norm(lb['shear']))
    result['phase_max_difference']=float(abs(max(a.nodes['phase'])-max(b.nodes['phase'])))
    return result


def initial_representation(path,seed_path):
    ref=Ref(path/'run/state-0');seed=np.loadtxt(seed_path)
    cuts=np.unique(np.r_[seed[:,0],ref.cells['y0'],ref.cells['y0']+ref.cells['dy']]);q,w=gauss(3)
    y=(cuts[:-1,None]+np.diff(cuts)[:,None]*q).ravel();weights=(np.diff(cuts)[:,None]*w).ravel()
    phi=samples_at(ref,np.full_like(y,.25),y)[0];target=np.interp(y,seed[:,0],seed[:,1])
    return dict(relative_L2=float(np.sqrt(np.sum(weights*(phi-target)**2)/np.sum(weights*target**2))),
                L2_absolute=float(np.sqrt(np.sum(weights*(phi-target)**2))),maximum_sample_error=float(max(abs(phi-target))))


def common_interval_rate(first, second):
    c, f = [np.atleast_1d(csv(p/'run/metrics.csv')) for p in [first, second]]
    end=c[-1]['time']; start=end-c[-1]['dt']
    vc=csv(first/'run'/f"state-{int(c[-1]['step'])}"/'accepted-surface.csv')['V']
    vf=np.zeros_like(vc)
    for row in f:
        overlap=max(0.,min(end,row['time'])-max(start,row['time']-row['dt']))
        if overlap: vf+=overlap*csv(second/'run'/f"state-{int(row['step'])}"/'accepted-surface.csv')['V']
    vf/=(end-start)
    return float(np.linalg.norm(vc-vf)/np.linalg.norm(vf))


def matrix_comparisons(paths):
    lookup={}
    for path in paths:
        case=json.loads((path/'record.json').read_text())['PF6a_case']
        lookup[case['normal_cells'],case['divisor'],case['quadrature']]=path
    configurations=[('temporal_dt_dt2',(32,1,6),(32,2,6)),('temporal_dt2_dt4',(32,2,6),(32,4,6)),
                    ('spatial',(32,4,6),(64,4,6)),('quadrature',(32,4,6),(32,4,12))]
    out={};out['integration_uncertainty']={}
    for name,aa,bb in configurations:
        a,b=lookup[aa],lookup[bb]
        i,j=[int(np.atleast_1d(csv(p/'run/metrics.csv'))[-1]['step']) for p in [a,b]]
        out[name]=difference(a,b,i,j)
        fine=difference(a,b,i,j,24)
        out['integration_uncertainty'][name]=max(abs(fine[k]-out[name][k]) for k in fine if fine[k] is not None)
        if name.startswith('temporal'):out[name]['common_interval_V_relative_l2']=common_interval_rate(a,b)
    seed=Path(__file__).with_name('pf6a-initial-phase.txt')
    out['initial_representation']={str(n):initial_representation(lookup[n,4,6],seed) for n in [32,64]}
    return out


def contact_checks(path):
    """Independent Q1 weak complementarity, including admissible sticking rows."""
    result=[]
    for directory in sorted((path/'run').glob('state-*'),key=lambda d:int(d.name[6:])):
        if not (directory/'native-loads.csv').exists(): continue
        surface=csv(directory/'accepted-surface.csv');nodes=csv(directory/'surface.csv')
        V=surface['V'];r=csv(directory/'native-loads.csv')['residual']
        mass=np.zeros((len(V),len(V)))
        for j,length in enumerate(np.diff(nodes['x'])):
            mass[j:j+2,j:j+2]+=length*np.array([[2.,1.],[1.,2.]])/6
        # This fixture uses the existing native 1e-20 m/s minimum rate.
        active=V==1e-20;free=~active
        assert np.min(V)>=1e-20
        mf=mass[np.ix_(free,free)];rf=r[free]
        free_norm=float(np.sqrt(rf@np.linalg.solve(mf,rf)/mf.sum())) if len(rf) else 0.
        outward=float(np.max(np.maximum(r[active]/mass.sum(axis=1)[active],0))) if any(active) else 0.
        assert max(free_norm,outward)<1e-6,(directory,free_norm,outward)
        result.append(dict(step=int(directory.name[6:]),active=int(sum(active)),free_RMS_Pa=free_norm,
                           sticking_outward_density_Pa=outward,minimum_V=float(min(V)),
                           complementarity_power=float((V-1e-20)@r),
                           unprojected_surface_RMS_Pa=float(csv(directory/'mechanical.csv')['surface_rms'])))
    return result


if __name__ == '__main__':
    main()
