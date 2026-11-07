#!/usr/bin/env python3
"""Frozen end-cut Cartesian Q2 stress space; no native solver or evolution.

The only new history freedom is an independent polynomial on either side of
an end line in crossed tensor cells. Q1/label/window cuts remain quadrature only.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
from compare_frozen_finite_end import (Frozen, ROOT, OUT as REFERENCE, GRIDS,
    METRIC, basis, clip, triangles, batches, index_basis, mass, point_traction)

OUT = ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/frozen_split_q2'


def load_fixture(path):
    """Reuse the tested field evaluator with complete saved input, no run tree."""
    data = np.load(path)
    f = Frozen.__new__(Frozen)
    f.state = SimpleNamespace(n=int(data['profile_n']), m=float(data['m']),
                              surface=data['surface'].copy())
    f.phi = data['phase_coeff'].copy()
    f.X = data['surface'][:, 1:3].copy()
    f.t = f.X[-1]-f.X[0]
    f.length = np.linalg.norm(f.t)
    f.t /= f.length
    f.normal = np.array([-f.t[1], f.t[0]])
    f.labels = (f.X-f.X[0])@f.t
    f.S = np.array([f.t[0]*f.normal[0], f.t[1]*f.normal[1],
                    .5*(f.t[0]*f.normal[1]+f.t[1]*f.normal[0])])
    f.I = data['surface'][:, 5].copy()
    f.V = data['prescribed_V'].copy()
    f.G, f.dt = float(data['G']), float(data['dt'])
    assert f.state.n == 64 and len(f.X) == 18 and f.G == 10 and f.dt == .1
    return f


def area(poly):
    p = poly-poly[0]
    return .5*abs(np.sum(p[:, 0]*np.roll(p[:, 1], -1)-p[:, 1]*np.roll(p[:, 0], -1)))


class SplitQ2:
    def __init__(self, f, n, split=True):
        self.f, self.n = f, n
        self.lookup = np.full((n*n, 3), -1, dtype=int)
        polygons, parents = [], []
        offsets = [f.X[0]@f.t, f.X[-1]@f.t]
        for i in range(n):
            for j in range(n):
                square = -2+4/n*(np.array([[0, 0], [1, 0], [1, 1], [0, 1]])+[i, j])
                pieces = [square]
                if split:
                    for offset in offsets:
                        next_pieces = []
                        for poly in pieces:
                            d = poly@f.t-offset
                            if d.min() < 0 < d.max():
                                next_pieces += [clip(poly, f.t, offset), clip(poly, -f.t, -offset)]
                            else:
                                next_pieces.append(poly)
                        pieces = next_pieces
                for poly in pieces:
                    # Discard only exactly zero-area geometric degeneracies.
                    # No small-positive-area merging, mode deletion or cutoff.
                    if area(poly) == 0:
                        continue
                    k = len(polygons)
                    region = np.searchsorted(offsets, poly.mean(axis=0)@f.t)
                    if split:
                        assert self.lookup[i*n+j, region] == -1
                        self.lookup[i*n+j, region] = k
                    else:
                        self.lookup[i*n+j, :] = k
                    polygons.append(poly)
                    parents.append(i*n+j)
        self.polygons = polygons
        self.parents = np.array(parents)
        self.lower = np.array([p.min(axis=0) for p in polygons])
        self.width = np.array([p.max(axis=0)-p.min(axis=0) for p in polygons])
        self.areas = np.array([area(p) for p in polygons])
        self.cut = np.bincount(self.parents)[self.parents] > 1
        self.size = len(polygons)
        self.cut_ids = np.flatnonzero(self.cut)
        self.cut_index = np.full(self.size, -1, dtype=int)
        self.cut_index[self.cut_ids] = np.arange(len(self.cut_ids))
        assert np.max(abs(np.bincount(self.parents, self.areas)-(4/n)**2)) < 2e-11
        assert self.areas.min() > 0 and np.all(self.width > 0)

    def samples(self, x):
        cells = np.minimum(np.floor((x+2)*(self.n/4)).astype(int), self.n-1)
        parent = cells[:, 0]*self.n+cells[:, 1]
        region = np.searchsorted([self.f.X[0]@self.f.t, self.f.X[-1]@self.f.t], x@self.f.t)
        ids = self.lookup[parent, region]
        assert ids.min() >= 0, 'Point assigned to absent end-cut piece'
        B = basis(2*(x-self.lower[ids])/self.width[ids]-1)
        return ids, B

    def evaluate(self, x, coeff):
        ids, B = self.samples(x)
        return np.einsum('pk,pklc->plc', B, coeff[ids])

    def cost(self):
        return dict(cells=self.n**2, pieces=self.size, cut_cells=int(self.cut.sum()//2),
                    additional_pieces=self.size-self.n**2,
                    tensor_scalar_dofs=27*self.size, uncut_scalar_dofs=27*self.n**2,
                    history_bytes_per_physical_tensor=8*27*self.size,
                    history_fraction_added=self.size/self.n**2-1,
                    full_mass_blocks=len(self.cut_ids),
                    full_mass_storage_bytes=8*81*len(self.cut_ids),
                    minimum_piece_fraction=float(self.areas.min()/(4/self.n)**2))


def polynomial(x):
    return 1+.2*x[:, 0]-.3*x[:, 1]+.1*x[:, 0]**2*x[:, 1]**2


def exact_piece_mass(space, k):
    """Green's theorem polynomial moments, independent of triangle quadrature."""
    p = 2*(space.polygons[k]-space.lower[k])/space.width[k]-1
    moments = np.zeros((5, 5))
    for a in range(5):
        for b in range(5):
            value = 0.
            for u, v in zip(p, np.roll(p, -1, axis=0)):
                d = v-u
                px = [math.comb(a+1,i)*u[0]**(a+1-i)*d[0]**i for i in range(a+2)]
                py = [math.comb(b,j)*u[1]**(b-j)*d[1]**j for j in range(b+1)]
                c = np.convolve(px, py)
                value += d[1]*sum(c/(np.arange(len(c))+1))/(a+1)
            moments[a, b] = value*np.prod(space.width[k])/4
    L = [np.array([1.]), np.array([0.,1.]), np.array([-.5,0.,1.5])]
    gram = np.zeros((9, 9))
    for i in range(9):
        for j in range(9):
            px, py = np.convolve(L[i//3],L[j//3]), np.convolve(L[i%3],L[j%3])
            gram[i, j] = np.einsum('a,b,ab->',px,py,moments[:len(px),:len(py)])
    return gram


def project(f, space, tri, q):
    started = time.perf_counter()
    rhs = np.zeros((space.size, 9, 2, 3))
    poly_rhs = np.zeros((space.size, 9))
    gram = np.zeros((len(space.cut_ids), 9, 9))
    measured_area = np.zeros(space.size)
    cut_qpoints = 0
    for x, w in batches(tri, q):
        nt, nq, _ = x.shape
        points = x.reshape(-1, 2)
        ids, B = space.samples(points)
        ids, B = ids.reshape(nt,nq), B.reshape(nt,nq,9)
        assert np.all(ids == ids[:, :1]), 'Integration triangle crosses a history piece'
        cells = ids[:, 0]
        src = f.fields(points)[0].reshape(nt,nq,2,3)
        np.add.at(rhs,cells,np.einsum('tqk,tqlc,tq->tklc',B,src,w))
        np.add.at(poly_rhs,cells,np.einsum('tqk,tq,tq->tk',B,polynomial(points).reshape(nt,nq),w))
        np.add.at(measured_area,cells,w.sum(axis=1))
        cut = space.cut[cells]
        np.add.at(gram,space.cut_index[cells[cut]],np.einsum('tqi,tqj,tq->tij',B[cut],B[cut],w[cut]))
        cut_qpoints += int(cut.sum())*nq
    assembly_seconds = time.perf_counter()-started
    tick = time.perf_counter()
    # Whole cells use their original tensor-product orthogonal mass.
    coefficients = rhs/mass(space.n)[None,:,None,None]
    poly_coeff = poly_rhs/mass(space.n)
    if len(gram):
        scale = np.sqrt(np.diagonal(gram,axis1=1,axis2=2))
        equilibrated = gram/scale[:,:,None]/scale[:,None,:]
        chol = np.linalg.cholesky(equilibrated)
        def solve(b):
            y = np.linalg.solve(chol,b/scale[:,:,None])
            z = np.linalg.solve(chol.transpose(0,2,1),y)
            return z/scale[:,:,None]
        coefficients[space.cut_ids] = solve(rhs[space.cut_ids].reshape(-1,9,6)).reshape(-1,9,2,3)
        poly_coeff[space.cut_ids] = solve(poly_rhs[space.cut_ids,:,None])[:,:,0]
        condition = dict(raw_max=float(np.linalg.cond(gram).max()),
                         equilibrated_max=float(np.linalg.cond(equilibrated).max()))
    else:
        condition = dict(raw_max=25.,equilibrated_max=1.)
    solve_seconds = time.perf_counter()-tick
    # Test polynomial reproduction at independent bounding-box sample points,
    # including extrapolation to box corners for clipped polygons.
    u = np.array([[-1,-1],[1,-1],[-1,1],[1,1],[.17,-.29]])
    x = space.lower[:,None]+.5*(u+1)*space.width[:,None]
    actual = np.einsum('qk,pk->pq',basis(u),poly_coeff)
    poly_error = float(abs(actual-polynomial(x.reshape(-1,2)).reshape(space.size,-1)).max())
    area_error = float(abs(measured_area-space.areas).max())
    assert poly_error < 1e-9, (space.n,q,'polynomial',poly_error)
    assert area_error < 2e-11
    if q == 10:
        mass_error = max((abs(gram[i]-exact_piece_mass(space,int(k))).max()
                          for i,k in enumerate(space.cut_ids)),default=0.)
        assert mass_error < 2e-11, (space.n,'Green mass',mass_error)
    else:
        mass_error = None
    return coefficients, dict(assembly_s=assembly_seconds,solve_s=solve_seconds,
        integration_points=len(tri)*q*q,cut_mass_points=cut_qpoints,
        polynomial_Linf=poly_error,piece_area_Linf=area_error,green_mass_Linf=mass_error,
        **condition)


def point_errors(f, space, coeff):
    cuts = [0.,f.length,.3,f.length-.3,*f.labels]
    for d in (0,1):
        cuts += ((np.linspace(-2,2,max(space.n,f.state.n)+1)-f.X[0,d])/f.t[d]).tolist()
    cuts = np.unique(np.clip(cuts,0,f.length))
    g,w = np.polynomial.legendre.leggauss(10)
    s = (.5*(cuts[1:,None]+cuts[:-1,None])+.5*np.diff(cuts)[:,None]*g).ravel()
    weights = (.5*np.diff(cuts)[:,None]*w).ravel()
    x = f.X[0]+s[:,None]*f.t
    a,b = space.evaluate(x,coeff)@(METRIC@f.S),f.fields(x)[0]@(METRIC@f.S)
    result = {}
    for name,mask in [('all',np.ones(len(s),bool)),('central',(s>.3)&(s<f.length-.3)),
                      ('ends',(s<.3)|(s>f.length-.3))]:
        error = np.einsum('pl,p->l',(a-b)**2,weights*mask)
        norm = np.einsum('pl,p->l',b*b,weights*mask)
        result[name] = dict(L2=np.sqrt(error).tolist(),relative_L2=np.sqrt(error/norm).tolist())
    return result


def evaluate(f, tri, spaces, coeff, saved):
    windows = ['all','central','ends']
    errors = {n:{tag:{win:np.zeros((2,5)) for win in windows} for tag in ['split','uncut']} for n in GRIDS}
    difference = {n:np.zeros(2) for n in GRIDS}
    convergence = {n:np.zeros(2) for n in GRIDS}
    cross = {n:np.zeros(2) for n in GRIDS}
    energy = {n:np.zeros(2) for n in GRIDS}
    mass_surface = {win:np.zeros((len(f.X),len(f.X))) for win in windows}
    source_load = np.zeros((len(f.X),2))
    loads = {n:{tag:np.zeros((len(f.X),2)) for tag in ['split','low','uncut']} for n in GRIDS}
    source_norm = np.zeros(2)
    slip_norm = np.zeros(2)
    for x,w in batches(tri,10):
        x,w = x.reshape(-1,2),w.ravel()
        src,chi,j,u = f.fields(x)
        dot = lambda a,b: np.einsum('...i,ij,...j->...',a,METRIC,b)
        source_norm += w@dot(src,src)
        eta = np.stack([np.full(len(x),10.),10*(1+.5*np.tanh(4*x[:,0]))],axis=1)
        base = -2*eta[:,:,None]*np.expm1(-f.G*f.dt/eta)[:,:,None]*np.array([.03,-.03,.01])
        slip_norm += w@dot(src-base,src-base)
        s,r = (x-f.X[0])@f.t,(x-f.X[0])@f.normal
        masks = dict(all=np.ones(len(x),bool),central=(s>.3)&(s<f.length-.3)&(abs(r)<.4),
                     ends=(np.minimum(abs(s),abs(s-f.length))<.3)&(abs(r)<.6))
        surface_masks = dict(all=np.ones(len(x),bool),central=(s>.3)&(s<f.length-.3),
                             ends=(s<.3)|(s>f.length-.3))
        hats = [(j,1-u),(j+1,u)]
        for a,ha in hats:
            np.add.at(source_load,a,(w*chi*ha)[:,None]*(src@(METRIC@f.S)))
            for b,hb in hats:
                for name,mask in surface_masks.items():
                    np.add.at(mass_surface[name],(a,b),w*chi*ha*hb*mask)
        for n,space in spaces.items():
            split = space.evaluate(x,coeff[n,10])
            low = space.evaluate(x,coeff[n,6])
            ids,B = index_basis(x,n)
            uncut = np.einsum('pk,pklc->plc',B,saved[f'n{n}_cut'][ids])
            for tag,tau in [('split',split),('uncut',uncut)]:
                e = tau-src
                values = np.concatenate([e**2,(e[:,:,0]+e[:,:,1])[:,:,None]**2,dot(e,e)[:,:,None]],axis=2)
                for win,mask in masks.items():
                    errors[n][tag][win] += np.einsum('plc,p->lc',values,w*mask)
            difference[n] += w@dot(split-uncut,split-uncut)
            convergence[n] += w@dot(split-low,split-low)
            cross[n] += w@dot(src-split,split-uncut)
            energy[n] += w@dot(src-split,src+split)/(4*f.G)
            for tag,tau in [('split',split),('low',low),('uncut',uncut)]:
                for a,ha in hats:
                    np.add.at(loads[n][tag],a,(w*chi*ha)[:,None]*(tau@(METRIC@f.S)))
    M = mass_surface['all']
    local_work = np.linalg.solve(M,source_load)
    local_power = f.V@source_load
    result = []
    for n in GRIDS:
        e2 = errors[n]['split']['all'][:,-1]
        old_e2 = errors[n]['uncut']['all'][:,-1]
        identity = abs(old_e2-e2-difference[n])
        assert np.all(identity < 1e-10*old_e2+1e-16), (n,'nested identity',identity)
        assert np.all(abs(energy[n]-e2/(4*f.G)) < 1e-14), (n,'energy',energy[n],e2)
        delta_ref = np.sqrt(convergence[n])
        assert np.all(delta_ref < 1e-9*np.sqrt(source_norm)+1e-12),(n,'quadrature',delta_ref)
        work = {tag:np.linalg.solve(M,load) for tag,load in loads[n].items()}
        work_change = abs(work['split']-work['low']).max(axis=0)
        assert np.all(work_change < 1e-9),(n,'work convergence',work_change)
        work_errors = {}
        for tag in ['split','uncut']:
            d = work[tag]-local_work
            work_errors[tag] = {name:dict(L2=np.sqrt(np.einsum('il,ij,jl->l',d,m,d)).tolist(),
                relative_L2=np.sqrt(np.einsum('il,ij,jl->l',d,m,d)/np.einsum('il,ij,jl->l',local_work,m,local_work)).tolist()) for name,m in mass_surface.items()}
        result.append(dict(n=n,split_L2=np.sqrt(e2).tolist(),uncut_L2=np.sqrt(old_e2).tolist(),
            improvement=np.sqrt(old_e2/e2).tolist(),relative_L2=np.sqrt(e2/source_norm).tolist(),
            localized_source_relative_L2=np.sqrt(e2/slip_norm).tolist(),
            windows_squared_errors={tag:{win:v.tolist() for win,v in ww.items()} for tag,ww in errors[n].items()},
            reference_field_L2_change=delta_ref.tolist(),reference_work_Linf_change=work_change.tolist(),
            nested_decomposition_absolute_error=identity.tolist(),projection_cross_term=cross[n].tolist(),
            energy_local_minus_projected=energy[n].tolist(),energy_identity_error=(energy[n]-e2/(4*f.G)).tolist(),
            slip_power_error=(f.V@loads[n]['split']-local_power).tolist(),
            slip_power_relative_error=((f.V@loads[n]['split']-local_power)/local_power).tolist(),
            work_traction=work_errors,point_traction=point_errors(f,spaces[n],coeff[n,10])))
    return result,dict(source_L2=np.sqrt(source_norm).tolist(),localized_source_L2=np.sqrt(slip_norm).tolist(),
        normalization_measure=float(M.sum()),normalization_relative_error=float(M.sum()/f.length-1),
        local_slip_power=local_power.tolist(),local_work_traction=local_work.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,default=REFERENCE)
    parser.add_argument('--output',type=Path,default=OUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    assert not (args.output/'comparison.json').exists(), 'Preserve completed evidence; choose another output.'
    start = time.perf_counter()
    f = load_fixture(args.reference/'frozen-input.npz')
    old = json.loads((args.reference/'comparison.json').read_text())
    saved = np.load(args.reference/'projection-coefficients.npz')
    tick = time.perf_counter()
    tri = triangles(f,128)
    partition_seconds = time.perf_counter()-tick
    assert len(tri) == old['reference_triangle_count']
    spaces,coeff,timings,costs = {},{},{},{}
    for n in GRIDS:
        tick = time.perf_counter()
        spaces[n] = SplitQ2(f,n)
        costs[n] = spaces[n].cost()
        timings[n] = dict(geometry_s=time.perf_counter()-tick)
        print('Space',n,costs[n],flush=True)
        for q in [6,10]:
            coeff[n,q],timings[n][f'split_q{q}'] = project(f,spaces[n],tri,q)
        uncut,timings[n]['uncut_q10'] = project(f,SplitQ2(f,n,False),tri,10)
        bridge = np.sqrt(np.einsum('ckl,k->l',np.einsum('...i,ij,...j->...',uncut-saved[f'n{n}_cut'],METRIC,uncut-saved[f'n{n}_cut']),mass(n)))
        assert np.all(bridge < 1e-12),(n,'retained uncut bridge',bridge)
        timings[n]['retained_uncut_bridge_L2'] = bridge.tolist()
        print('Projection',n,'complete',flush=True)
    tick = time.perf_counter()
    rows,reference = evaluate(f,tri,spaces,coeff,saved)
    evaluation_seconds = time.perf_counter()-tick
    assert np.max(abs(np.array(reference['source_L2'])-old['reference']['source_L2'])) < 1e-12
    for row in rows:
        n = row['n']
        baseline = next(r for r in old['cases'] if r['n']==n and r['q']=='cut')
        assert np.max(abs(np.array(row['uncut_L2'])-baseline['best_L2'])) < 1e-12
    result = dict(revision=os.popen('git rev-parse HEAD').read().strip(),laws=['constant','layered'],
        scope='Frozen locally split Cartesian Q2; native solver and physical inputs unchanged',
        reference_directory=str(args.reference),reference_triangle_count=len(tri),reference=reference,
        costs=costs,timings=timings,cases=rows,partition_s=partition_seconds,
        evaluation_s=evaluation_seconds,total_s=time.perf_counter()-start,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    # Save only the new candidate, enough to reproduce endpoint views; all old
    # controls/physical inputs remain in their retained directory.
    np.savez_compressed(args.output/'split-coefficients.npz',**{f'n{n}':coeff[n,10] for n in GRIDS})
    for r in rows:print('Result',r['n'],r['split_L2'],r['improvement'],flush=True)
    print('Seconds',result['total_s'],flush=True)


if __name__ == '__main__':
    main()
