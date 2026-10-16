#!/usr/bin/env python3
"""Independent split-cell L2 projection and bounded transport-method assessment.

No production quadrature, mass matrices, projection code, or error integrals are
imported. Sharp/native piecewise-polynomial products use split Gauss3, which is
exact. The smooth tanh reference uses Gauss16, checked against Gauss24.
"""
import argparse
import csv
import json
import re
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from numpy.polynomial.legendre import leggauss, Legendre

B = np.array([1000., -400., 700.])
KAPPA = -1e8*np.expm1(-.001)

def norm2(a):
    return a[..., 0]**2+a[..., 1]**2+(a[..., 0]+a[..., 1])**2+2*a[..., 2]**2

def gauss(n):
    a, w = leggauss(n)
    return (a+1)/2, w/2

def lagrange(r):
    return np.stack([(2*r-1)*(r-1), 4*r*(1-r), r*(2*r-1)], axis=-1)

def derivative(r):
    return np.stack([4*r-3, 4-8*r, 4*r-1], axis=-1)

def velocity(x, y):
    return np.stack([.001*np.exp(-((x-.45)/.16)**2)*(y-.5),
                     .0006*np.exp(-((x-.55)/.19)**2)*((y-.5)**2+.3*(y-.5))], axis=-1)

def native(x, y, nx):
    # Independently differentiate the tensor-product interpolant of native Q2
    # nodal velocities. This is checked against make_dg_history_trial output.
    x, y = np.broadcast_arrays(x, y)
    ox, oy = np.floor(x*nx)/nx, np.floor(y*2*nx)/(2*nx)
    r, s = (x-ox)*nx, (y-oy)*2*nx
    l, m, dl, dm = lagrange(r), lagrange(s), derivative(r)*nx, derivative(s)*2*nx
    grad = np.zeros(x.shape+(2, 2))
    for i in range(3):
        for j in range(3):
            u = velocity(ox+i/(2*nx), oy+j/(4*nx))
            grad[..., :, 0] += u*dl[..., i, None]*m[..., j, None]
            grad[..., :, 1] += u*l[..., i, None]*dm[..., j, None]
    tr = (grad[..., 0, 0]+grad[..., 1, 1])/3
    return KAPPA*np.stack([2*(grad[..., 0, 0]-tr), 2*(grad[..., 1, 1]-tr),
                          grad[..., 0, 1]+grad[..., 1, 0]], axis=-1)

def exact(case, x, y, t, nx):
    z = x-.4*t
    if case == 'native_history':
        return native(z, y, nx)
    if case == 'signed_jump':
        f = np.where((z >= .25) & (z < .5), 1., np.where((z >= .5) & (z < .75), -1., 0.))
    else:
        d = .15625/np.log(9.)
        f = .5*np.tanh((z-.25)/d)-np.tanh((z-.5)/d)+.5*np.tanh((z-.75)/d)
    return f[..., None]*B

class Data:
    def __init__(self, directory):
        self.path = Path(directory)/'run'
        self.cells = np.genfromtxt(self.path/'cells.csv', names=True, delimiter=',')
        self.basis = np.genfromtxt(self.path/'basis.csv', names=True, delimiter=',')
        self.nx = round(1/float(self.cells['dx'][0]))
        self.ny = 2*self.nx
        self.indices = np.stack([self.cells[f'i{i}'].astype(int) for i in range(9)], axis=1)
        self.ri = np.rint(self.basis['r']*2).astype(int)
        self.si = np.rint(self.basis['s']*2).astype(int)
        self.local_mass = None
        self.cg = None
        self.cache = {}
    def shape(self, r, s):
        return lagrange(r)[:, self.ri]*lagrange(s)[:, self.si]
    def load(self, name):
        a = np.loadtxt(self.path/(name+'.csv'), delimiter=',', skiprows=1)
        assert np.array_equal(a[:, 0], np.arange(len(a)))
        return a[self.indices, 1:]
    def mass(self):
        if self.local_mass is None:
            z, w = gauss(3)
            r, s = np.repeat(z, 3), np.tile(z, 3)
            N = self.shape(r, s)
            self.local_mass = N.T@((np.repeat(w, 3)*np.tile(w, 3))[:, None]*N)/(self.nx*self.ny)
        return self.local_mass
    def cg_system(self):
        if self.cg is None:
            c = self.cells
            ix = np.rint((c['x0'][:, None]+c['dx'][:, None]*self.basis['r'])*2*self.nx).astype(int)
            iy = np.rint((c['y0'][:, None]+c['dy'][:, None]*self.basis['s'])*2*self.ny).astype(int)
            ids = iy*(2*self.nx+1)+ix
            n = (2*self.nx+1)*(2*self.ny+1)
            rows, cols = np.broadcast_arrays(ids[:, :, None], ids[:, None, :])
            mat = coo_matrix((np.broadcast_to(self.mass(), rows.shape).ravel(), (rows.ravel(), cols.ravel())), shape=(n, n)).tocsc()
            self.cg = ids, splu(mat)
        return self.cg
    def quadrature(self, case, t, order=None):
        order = order or (16 if case == 'finite_width' else 3)
        key = case, t, order
        if key in self.cache:
            return self.cache[key]
        z, wz = gauss(order)
        # Keep quadrature data in flat arrays grouped by physical cell.
        cell_ids, xx, yy, ww, NN = [], [], [], [], []
        for k, c in enumerate(self.cells):
            a, b = c['x0'], c['x0']+c['dx']
            cuts = [a, b]
            if case == 'signed_jump':
                cuts += [v+.4*t for v in [.25, .5, .75] if a < v+.4*t < b]
            elif case == 'native_history':
                cuts += [v/self.nx+.4*t for v in range(-self.nx, 2*self.nx+1) if a < v/self.nx+.4*t < b]
            cuts = sorted(set(cuts))
            for l, u in zip(cuts[:-1], cuts[1:]):
                if u-l < 1e-15:
                    continue
                x = np.repeat(l+(u-l)*z, order)
                y = np.tile(c['y0']+c['dy']*z, order)
                N = self.shape((x-a)/c['dx'], (y-c['y0'])/c['dy'])
                cell_ids.extend([k]*len(x)); xx.extend(x); yy.extend(y)
                ww.extend((u-l)*c['dy']*np.repeat(wz, order)*np.tile(wz, order)); NN.extend(N)
        result = np.array(cell_ids), np.array(xx), np.array(yy), np.array(ww), np.array(NN)
        self.cache[key] = result
        return result
    def project(self, case, t, space, order=None):
        k, x, y, w, N = self.quadrature(case, t, order)
        f = exact(case, x, y, t, self.nx)
        rhs = np.zeros((len(self.cells), 9, 3))
        np.add.at(rhs, k, w[:, None, None]*N[:, :, None]*f[:, None, :])
        return self.project_rhs(rhs,space)
    def project_rhs(self,rhs,space):
        if space == 'DG':
            return np.linalg.solve(self.mass(), rhs.transpose(1, 0, 2).reshape(9, -1)).reshape(9, len(self.cells), 3).transpose(1, 0, 2)
        ids, solve = self.cg_system()
        global_rhs = np.zeros(((2*self.nx+1)*(2*self.ny+1), 3))
        np.add.at(global_rhs, ids.ravel(), rhs.reshape(-1, 3))
        return solve.solve(global_rhs)[ids]
    def errors(self, values, case, t, space, order=None):
        p = self.project(case, t, space, order)
        k, x, y, w, N = self.quadrature(case, t, order)
        f = exact(case, x, y, t, self.nx)
        u = np.einsum('qi,qic->qc', N, values[k])
        pp = np.einsum('qi,qic->qc', N, p[k])
        q = np.sum(w*norm2(f))
        error, rep, evolution = [np.sum(w*norm2(v)) for v in [u-f, pp-f, u-pp]]
        l1 = np.sum(w*np.sqrt(norm2(u-f)))
        integral = np.sum(w[:, None]*(u-f), axis=0)
        return dict(relative_L2=np.sqrt(error/q), representation_L2=np.sqrt(rep/q),
                    evolution_L2=np.sqrt(evolution/q), orthogonality_defect=abs(error-rep-evolution)/q,
                    relative_L1_quadrature_estimate=l1/np.sum(w*np.sqrt(norm2(f))), tensor_integral_error=integral.tolist(),
                    integral_error_full_norm=float(np.sqrt(norm2(integral))),
                    quadratic_norm=np.sum(w*norm2(u)), exact_quadratic_norm=q)
    def difference(self, a, b):
        return np.sqrt(np.einsum('cik,ij,cjk->', a-b, self.mass(), np.stack([2*(a-b)[..., 0]+(a-b)[..., 1], 2*(a-b)[..., 1]+(a-b)[..., 0], 2*(a-b)[..., 2]], axis=-1)))
    def evaluate(self, values, x, y):
        ix=np.clip(np.floor(x*self.nx).astype(int),0,self.nx-1)
        iy=np.clip(np.floor(y*self.ny).astype(int),0,self.ny-1)
        lookup=np.empty((self.ny,self.nx),dtype=int)
        lookup[np.rint(self.cells['y0']*self.ny).astype(int),np.rint(self.cells['x0']*self.nx).astype(int)]=np.arange(len(self.cells))
        k=lookup[iy,ix]
        N=self.shape((x-self.cells['x0'][k])/self.cells['dx'][k],(y-self.cells['y0'][k])/self.cells['dy'][k])
        return np.einsum('qi,qic->qc',N,values[k])
    def transported_initial_error(self, values, initial, case):
        # Exact characteristics of the actual initialized polynomial, split at
        # its translated cell faces. Analytic inflow supplies x-vt<0.
        k,x,y,w,N=self.quadrature('native_history',.2,16 if case=='finite_width' else 3)
        u=np.einsum('qi,qic->qc',N,values[k])
        advected=self.evaluate(initial,x-.08,y)
        outside=x<.08
        advected[outside]=exact(case,x[outside],y[outside],.2,self.nx)
        physical=exact(case,x,y,.2,self.nx)
        q=np.sum(w*norm2(physical))
        return dict(evolution_from_initialized_field_L2=np.sqrt(np.sum(w*norm2(u-advected))/q),
                    transported_initialization_L2=np.sqrt(np.sum(w*norm2(advected-physical))/q))
    def weak_loads(self, values, points):
        u=self.evaluate(values,points['x'],points['y'])
        return weak_loads(u,points)
    def extrema_fronts(self, values, case, time=.2):
        # Pulse solutions are independent of y. Verify rather than assume.
        transverse=0.
        columns=np.rint(self.cells['x0']*self.nx).astype(int)
        for r in range(3):
            local=values[:,self.ri==r,:]
            canonical=np.zeros((self.nx,3))
            for col in range(self.nx): canonical[col]=local[np.flatnonzero(columns==col)[0],0]
            transverse=max(transverse,float(np.max(np.sqrt(norm2(local-canonical[columns,None,:])))))
        # Keep the measured iterative-solver transverse residue in the ledger;
        # the reported fronts and polynomial extrema are the y=.5 trace.
        ids = np.flatnonzero((self.cells['y0'] <= .5) & (self.cells['y0']+self.cells['dy'] > .5))
        ids = ids[np.argsort(self.cells['x0'][ids])]
        c = self.cells[ids]
        N = self.shape(np.tile([0., .5, 1.], len(ids)), np.repeat((.5-c['y0'])/c['dy'], 3))
        v = np.einsum('qi,qic->qc', N, np.repeat(values[ids], 3, axis=0))[:, 0].reshape(-1, 3)/1000
        a, b, cc = 2*(v[:, 0]-2*v[:, 1]+v[:, 2]), -3*v[:, 0]+4*v[:, 1]-v[:, 2], v[:, 0]
        vertex = np.divide(-b, 2*a, out=np.zeros_like(a), where=abs(a)>1e-30)
        extrema = list(v[:, [0, 2]].ravel())+list((a*vertex**2+b*vertex+cc)[(vertex>0)&(vertex<1)])
        d = .15625/np.log(9.)
        from scipy.optimize import brentq, minimize_scalar
        def f(x):
            return float(exact(case, np.array(x), np.array(.5), time, self.nx)[0]/1000)
        peak = 1. if case == 'signed_jump' else -minimize_scalar(lambda x: -f(x), bounds=(.25+.4*time, .5+.4*time), method='bounded', options={'xatol':1e-15}).fun
        # Exact crossing locations include overlap of all tanh transitions.
        levels = [[.1*peak, .9*peak], [.8*peak, -.8*peak], [-.9*peak, -.1*peak]]
        fronts = [z+.4*time for z in [.25,.5,.75]]
        widths, exact_widths = [], []
        for center, targets in zip(fronts, levels):
            crossings = []
            for target in targets:
                roots = []
                for j in range(len(ids)):
                    coeff = [a[j], b[j], cc[j]-target]
                    for root in np.roots(np.trim_zeros(coeff, 'f')):
                        if abs(root.imag)<1e-9 and -1e-9<=root.real<=1+1e-9:
                            roots.append(c['x0'][j]+c['dx'][j]*root.real)
                # A DG face jump can cross a level without a polynomial root
                # inside either adjacent cell. This includes the exact initial
                # cell-aligned discontinuity, whose transition width is zero.
                for j in range(len(ids)-1):
                    if (v[j,2]-target)*(v[j+1,0]-target)<=0:
                        roots.append(c['x0'][j]+c['dx'][j])
                crossings.append(min(roots, key=lambda z: abs(z-center)) if roots else float('nan'))
            widths.append(abs(crossings[1]-crossings[0]))
            exact_widths.append(0. if case=='signed_jump' else abs(brentq(lambda x:f(x)-targets[1],center-.12,center+.12)-brentq(lambda x:f(x)-targets[0],center-.12,center+.12)))
        # Inspect all y rows at GL3 x nodes for transverse variation.
        # Per-cell native component range is bounded separately for history.
        return dict(transverse_defect_Pa=transverse,minimum=min(extrema), maximum=max(extrema),
                    overshoot=max(0., max(extrema)-1), undershoot=max(0., -1-min(extrema)),
                    analytic_peak=peak, peak_attenuation=1-max(extrema)/peak, negative_peak_attenuation=1+min(extrema)/peak,
                    analytic_range_overshoot=max(0.,max(extrema)-peak),analytic_range_undershoot=max(0.,-min(extrema)-peak),
                    front_widths=widths, analytic_front_widths=exact_widths)

def projection_verification(data):
    def polynomial(x,y):
        return np.stack([1+x+y+x*x*y*y,x*x-.3*x*y+y*y,1-.7*x*y+.4*x*x*y*y],axis=-1)
    k,x,y,w,N=data.quadrature('native_history',0.)
    rhs=np.zeros((len(data.cells),9,3))
    f=polynomial(x,y)
    np.add.at(rhs,k,w[:,None,None]*N[:,:,None]*f[:,None,:])
    nodes_x=data.cells['x0'][:,None]+data.cells['dx'][:,None]*data.basis['r']
    nodes_y=data.cells['y0'][:,None]+data.cells['dy'][:,None]*data.basis['s']
    result={}
    for space in ['DG','CG']:
        p=data.project_rhs(rhs,space)
        result[space+'_complete_Q2_max_error']=float(np.max(abs(p-polynomial(nodes_x,nodes_y))))
        assert result[space+'_complete_Q2_max_error']<2e-12
        const_rhs=np.broadcast_to(data.mass().sum(axis=1)[None,:,None]*np.array([2.,-1.,.7]),rhs.shape).copy()
        c=data.project_rhs(const_rhs,space)
        result[space+'_constant_max_error']=float(np.max(abs(c-np.array([2.,-1.,.7]))))
        assert result[space+'_constant_max_error']<2e-12
        result[space+'_zero_max_error']=float(np.max(abs(data.project_rhs(np.zeros_like(rhs),space))))
        assert result[space+'_zero_max_error']==0.
    result['minimum_local_mass_eigenvalue']=float(min(np.linalg.eigvalsh(data.mass())))
    assert result['minimum_local_mass_eigenvalue']>0
    return result

def weak_loads(u,points):
    out=np.zeros((int(max(points['segment']))+2,2))
    shear=u[:,0]*points['Sxx']+u[:,1]*points['Syy']+2*u[:,2]*points['Sxy']
    normal=-(u[:,0]*points['Nxx']+u[:,1]*points['Nyy']+2*u[:,2]*points['Nxy'])
    for i in [0,1]:
        np.add.at(out,points['segment'].astype(int)+i,(points['w']*points['N'+str(i)])[:,None]*np.stack([shear,normal],axis=-1))
    return out

def predicted_sharp():
    # A second implementation: normalized Legendre moments, no nodal mass solve.
    n, t = 32, .2
    represented, exact_q = 0., .5
    for i in range(n):
        a, b = i/n, (i+1)/n
        moments = np.zeros(3)
        for lo, hi, f in [(.25+.4*t,.5+.4*t,1.),(.5+.4*t,.75+.4*t,-1.)]:
            l, u = max(a, lo), min(b, hi)
            if l < u:
                for k in range(3):
                    primitive = Legendre.basis(k).integ()
                    moments[k] += f*(primitive(2*n*(u-a)-1)-primitive(2*n*(l-a)-1))/(2*n)
        represented += sum((2*k+1)*n*moments[k]**2 for k in range(3))
    return np.sqrt((exact_q-represented)/exact_q)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scalar',nargs='*',type=Path,default=[])
    parser.add_argument('--dg',nargs='*',type=Path,default=[])
    parser.add_argument('--native',nargs='*',type=Path,default=[])
    args=parser.parse_args()
    result={'predicted_DGQ2_sharp_relative_error':predicted_sharp(),'runs':{},'checks':[]}
    sample=(args.scalar+args.dg+args.native)[0]
    result['projection_verification']=projection_verification(Data(sample))
    assert abs(predicted_sharp()-.1498621703)<5e-11
    result['checks'].append('Independent Legendre sharp-pulse prediction matches 0.1498621703')
    for directory,space in [(p,'CG') for p in args.scalar]+[(p,'DG') for p in args.dg]+[(p,'DG') for p in args.native]:
        data=Data(directory)
        row={'space':space,'nx':data.nx,'scalar_dofs':(2*data.nx+1)*(2*data.ny+1) if space=='CG' else 9*data.nx*data.ny,'cases':{}}
        has_native=(data.path/'field-native_history.csv').is_file() and (data.path/'field-native_history.csv').stat().st_size>30
        has_pulse=directory in args.dg or ((data.path/'field-signed_jump.csv').is_file() and (data.path/'field-signed_jump.csv').stat().st_size>30)
        cases=(['native_history'] if directory in args.native or not has_pulse else ['signed_jump','finite_width']+(['native_history'] if space=='CG' and has_native else []))
        for case in cases:
            labels=[case] if space=='CG' else ([case+'-'+p for p in ['unlimited','scaling']] if case!='native_history' else [case+'-'+str(d) for d in [1,2,4]])
            for label in labels:
                values=data.load('field-'+label)
                stats=data.errors(values,case,.2,space)
                initial=data.load('initial-'+case)
                stats['initialization']=data.errors(initial,case,0.,space)
                stats.update(data.transported_initial_error(values,initial,case))
                stats['quadratic_norm_change_from_initial']=stats['quadratic_norm']-stats['initialization']['quadratic_norm']
                assert stats['orthogonality_defect']<2e-12
                if case!='native_history':
                    stats.update(data.extrema_fronts(values,case))
                    stats['initial_extrema']=data.extrema_fronts(initial,case,0.)
                    stats['projection_extrema']=data.extrema_fronts(data.project(case,.2,space),case)
                if case=='signed_jump' and data.nx==32 and space=='DG':
                    assert abs(stats['representation_L2']-predicted_sharp())<2e-13
                if case=='finite_width':
                    high=data.errors(values,case,.2,space,24)
                    stats['quadrature_check_L2']=abs(high['relative_L2']-stats['relative_L2'])
                    assert stats['quadrature_check_L2']<1e-12
                row['cases'][label]=stats
        if space=='CG':
            metadata=np.genfromtxt(data.path/'metadata.csv',names=True,delimiter=',')
            row['native_metadata']={n:float(metadata[n]) for n in metadata.dtype.names}
        record=json.loads((directory/'record.json').read_text())
        row['run_seconds']=record['seconds']
        row['executable_sha256']=record['executable_sha256']
        if (data.path/'costs.csv').is_file():
            row['costs']=[{n:(str(r[n]) if r.dtype[n].kind in 'US' else float(r[n])) for n in r.dtype.names} for r in np.atleast_1d(np.genfromtxt(data.path/'costs.csv',names=True,delimiter=',',dtype=None,encoding='utf-8'))]
        else:
            log=(directory/'run.log').read_text()
            row['composition_cost_seconds']={}
            for category in ['Assemble composition system','Build composition preconditioner','Solve composition system']:
                matches=re.findall(r'\| '+category+r' +\| +[0-9]+ +\| +([0-9.e+]+)s',log)
                if matches: row['composition_cost_seconds'][category]=float(matches[-1])
            if (data.path/'timer.csv').is_file():
                def times(file):
                    with file.open() as stream: return {r['category']:float(r['seconds']) for r in csv.DictReader(stream)}
                end,start=times(data.path/'timer.csv'),times(data.path/'initial-timer.csv')
                row['composition_cost_seconds']={k:end.get(k,0)-start.get(k,0) for k in row['composition_cost_seconds']}
            fields=int(re.search(r'set Number of fields = ([0-9]+)',(directory/'input.prm').read_text()).group(1))
            row['composition_fields']=fields
            row['composition_seconds_per_scalar']=sum(row['composition_cost_seconds'].values())/fields
        result['runs'][directory.name]=row
        print(directory.name, {c:{k:round(v,12) for k,v in s.items() if k in ['relative_L2','representation_L2','evolution_L2','minimum','maximum']} for c,s in row['cases'].items()},flush=True)
    result['weak_loads']={}
    for directory in args.native:
        data=Data(directory)
        points=np.genfromtxt(data.path/'weak-points.csv',names=True,delimiter=',')
        reference=weak_loads(exact('native_history',points['x'],points['y'],.2,data.nx),points)
        assert np.max(abs(points['w']-points['JxW']*points['chi']))<1e-14
        loads={}
        sources=[(directory,'native_history-'+str(d),d) for d in [1,2,4]]
        sources += [(p,'native_history',None) for p in args.scalar if Data(p).nx==data.nx and (p/'run/field-native_history.csv').stat().st_size>30]
        for path,label,divisor in sources:
            other=Data(path)
            actual=other.weak_loads(other.load('field-'+label),points)
            stats=dict(relative_L2=float(np.linalg.norm(actual-reference)/np.linalg.norm(reference)),
                       shear_relative_L2=float(np.linalg.norm(actual[:,0]-reference[:,0])/np.linalg.norm(reference[:,0])),
                       normal_relative_L2=float(np.linalg.norm(actual[:,1]-reference[:,1])/np.linalg.norm(reference[:,1])),
                       integral_error=np.sum(actual-reference,axis=0).tolist(),loads=actual.tolist())
            if divisor:
                native_load=np.loadtxt(data.path/f'native-loads-{divisor}.csv',delimiter=',',skiprows=1)[:,1:]
                stats['independent_native_contraction_error']=float(np.max(abs(actual-native_load)))
                assert stats['independent_native_contraction_error']<1e-10
            loads[path.name+':'+label]=stats
        result['weak_loads'][str(data.nx)]=dict(reference_norms=np.linalg.norm(reference,axis=0).tolist(),reference=reference.tolist(),methods=loads,quadrature_points=len(points),measure=float(sum(points['w'])))
    # Field differences at fixed mesh are measured uncertainty, not an assumed
    # temporal order or a fitted extrapolation. Store per-case normalization.
    result['time_differences']=[]
    for a in list(args.scalar)+list(args.dg):
        for b in list(args.scalar)+list(args.dg):
            ra,rb=result['runs'][a.name],result['runs'][b.name]
            if a==b or ra['nx']!=rb['nx'] or ra['space']!=rb['space']: continue
            family=lambda p: 'supg' if 'supg' in p.name else ('ev' if '-ev-' in p.name else 'DG')
            if family(a)!=family(b): continue
            # Order entries by actual dt metadata or known retained divisor.
            def dt(p,r):
                if 'native_metadata' in r: return .2/round(.2/float(re.search(r'set Function expression = ([0-9.e+-]+)',(p/'input.prm').read_text()).group(1)))
                return .2/(103*int(p.name.rsplit('-',1)[-1])) if r['nx']==32 else .2/52
            da,db=dt(a,ra),dt(b,rb)
            if da<=db: continue
            data=Data(a)
            for label in ra['cases'].keys() & rb['cases'].keys():
                va,vb=data.load('field-'+label),Data(b).load('field-'+label)
                # Cell order and basis are stable for retained native runs.
                assert np.array_equal(data.cells,Data(b).cells)
                denominator=ra['cases'][label]['exact_quadratic_norm']
                result['time_differences'].append(dict(coarse=a.name,fine=b.name,case=label,coarse_dt=da,fine_dt=db,
                    relative_L2=float(data.difference(va,vb)/np.sqrt(denominator))))
    for directory in args.native:
        data=Data(directory)
        row=result['runs'][directory.name]
        q=row['cases']['native_history-1']['exact_quadratic_norm']
        for a,b in [(1,2),(2,4),(1,4)]:
            result['time_differences'].append(dict(coarse=directory.name,fine=directory.name,case='native_history',
                coarse_divisor=a,fine_divisor=b,relative_L2=float(data.difference(data.load('field-native_history-'+str(a)),data.load('field-native_history-'+str(b)))/np.sqrt(q))))
    result['time_differences'].sort(key=lambda r:(r['coarse'],r['fine'],r['case'],r.get('coarse_divisor',0),r.get('fine_divisor',0)))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__': main()
