"""Independent physical-coordinate references for the bounded section and direction controls."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import erf

from compare import fields, read


def strip_check(folder, angle):
    p = fields(folder, 1)
    d = read(folder/'tip-sections-0001.csv')
    s = read(folder/'tip-planes-0001.csv')
    theta = np.deg2rad(angle)
    t = np.array([np.cos(theta), np.sin(theta)])
    n = np.array([-t[1], t[0]])
    coords = np.column_stack((p['x'],p['y']))
    xi, z = coords@t, coords@n
    corner = np.sort(np.array([[0,0],[2,0],[0,1],[2,1]])@t)
    center = np.array([1.,.5])@n
    width = .09

    def chord(x):
        bounds = np.sort(np.array([-x*t/n, (np.array([2.,1.])-x*t)/n]), axis=0)
        lo, hi = max(bounds[0]), min(bounds[1])
        return (hi-lo, (1+x/5)*(hi-lo)+(hi**3-lo**3)/3,
                width*np.sqrt(np.pi)/2*(erf((hi-center)/width)-erf((lo-center)/width)))

    dx = s['reference_dxi']
    measured = []
    for v in (np.ones(len(p)), 1+xi/5+z*z, np.exp(-((z-center)/width)**2)):
        measured.append(np.bincount(p['section'],p['weight']*v)/dx)
    exact = []
    for mid, h in zip(s['xi'],dx):
        lo, hi = mid-h/2, mid+h/2
        exact.append([quad(lambda x: chord(x)[j],lo,hi,points=corner[(corner>lo)&(corner<hi)],
                           epsabs=1e-13)[0]/h for j in range(3)])
    errors = np.abs(np.asarray(measured)-np.asarray(exact).T).max(axis=1)
    rate = np.bincount(p['section'],p['weight']*p['w'])/s['ds']
    out = dict(stations=len(s),points=len(p),strip_length_error=errors[0],polynomial_error=errors[1],
               gaussian_max_abs=errors[2],area_error=abs(p['weight'].sum()-2),
               power_error=abs(s['ds']@(d['force']*d['V'])-p['weight']@(d['force'][p['section']]*p['w'])),
               rate_error=max(abs(rate-d['V'])))
    assert max(out[k] for k in ('area_error','power_error','rate_error','polynomial_error','strip_length_error')) < 2e-11
    return out


def count(nx, ny):
    # Count nonempty local intervals, with identical geometric tolerance, no histories.
    c, s = np.sqrt(3.)/2,.5
    hx, hy = 8/nx, 4/ny
    N = math.ceil((8*c+4*s)/min(hx,hy))
    knots = np.linspace(0,8*c+4*s,N+1)
    patches = 0
    for i in range(nx):
        for j in range(ny):
            x = i*hx*c+j*hy*s
            v = x+np.sort([0,hx*c,hy*s,hx*c+hy*s])
            a, b = np.searchsorted(knots, [v[0],v[-1]])
            pts = np.sort(np.r_[v,knots[a:b]])
            patches += np.count_nonzero(np.diff(pts)>64*np.finfo(float).eps*knots[-1])
    return dict(stations=N,points=int(patches*36))


def frozen(folder,step):
    # Horizontal midpoint fixture with constant G, zero-clock stress and a seed at (1.5,.5).
    p, old = fields(folder,step),fields(folder,step-1)
    assert np.all(p['x']==old['x']) and np.all(p['y']==old['y'])
    angles = np.radians(np.linspace(-45,45,181))
    ds = read(folder/f'tip-planes-{step:04}.csv')
    G, dt, ell = 1e6,.0025,.25
    beta = 1/(1+dt*G/1e8)
    m = (26.6666666667 / ((8/3)*ell))/(1000**2/(2*G))
    phi, area = p['stage_phi'], p['weight']
    F = m*phi*(1+phi)/(m*phi*(1+phi)+(1-phi)**2)
    def predictor(ij):
        return .5*(p['tau'+ij]+old['tau'+ij])+G*(p['A'+ij]+(1-2*beta)*old['A'+ij])
    u, v = .5*(predictor('yy')-predictor('xx')),predictor('xy')
    distances = np.hypot(p['x']-1.5,p['y']-.5)
    rho = np.maximum(0,1-(distances[None,:]/(ell*np.array([[1],[2],[4]])))**2)**2
    rows = []
    for k in range(len(ds)):
        sel = p['section']==k
        if ds['x'][k]<=1.5 or ds['parent_formed'][k] or not np.any(rho[:,sel]): continue
        A, fraction = area[sel],F[sel]
        age = ds['parent_age'][k]
        scores = []
        for angle in angles:
            tau = u[sel]*np.sin(2*angle)+v[sel]*np.cos(2*angle)
            dn = A*np.cos(angle)/ds['reference_dx'][k]
            M,free = dn@fraction/(beta*G*dt),dn@(fraction*tau)/(beta*G*dt)
            def resistance(V):
                y = dt*abs(V)/.01
                theta = age*np.exp(-y)+dt*(-np.expm1(-y)/y if y else 1.)
                return 3*np.arcsinh(V/.002*np.exp((.6+.02*np.log(.001*theta/.01))/.015))+1000*V
            V = brentq(lambda V: V+M*resistance(V)-free,min(0,free),max(0,free),xtol=1e-32) if free else 0.
            scores.append(rho[:,sel]@(A*((1-fraction)*(tau-resistance(V)))**2)*m/(2*G))
        scores = np.array(scores)
        rows.append((ds['x'][k],scores.max(axis=0),np.rad2deg(angles[scores.argmax(axis=0)])))
    out = {}
    for j, radius in enumerate((1,2,4)):
        active=[(x,score[j],turn[j]) for x,score,turn in rows if score[j]>0]
        out[str(radius)] = dict(sample_points=int(np.sum(rho[j]>0)),stations=len(active),
                                max_integrated_onset=max(row[1] for row in active),
                                most_driven_station=max(active,key=lambda row:row[1]),
                                nearest_station=active[0],kernel_area=area@rho[j],
                                max_remote_weight=max(rho[j][distances>=radius*ell],default=0.))
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['strip','count','frozen'])
    p.add_argument('folder',nargs='?',type=Path)
    p.add_argument('--step',type=int,default=1)
    a=p.parse_args()
    result=count(256,128) if a.mode=='count' else strip_check(a.folder,30) if a.mode=='strip' else frozen(a.folder,a.step)
    print(json.dumps(result,indent=2))
