"""Independent pointwise localized score at the rejected cold exact-line station."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from compare import fields, read


def localized_scores(p, section):
    """Pointwise first-midpoint score on the 30-degree strip fixture, for a cold parent."""
    radius, dt, G = .5, .0025, 1e6
    m = (26.6666666667 / ((8/3)*.25)) / (1000**2/(2*G))
    beta = 1/(1+dt*G/1e8)
    theta = np.pi/6
    center = np.array([1+.5*np.cos(theta),.5+.5*np.sin(theta)])
    rho = np.maximum(0,1-((p['x']-center[0])**2+(p['y']-center[1])**2)/radius**2)**2
    phi = p['stage_phi']
    F = m*phi*(1+phi)/(m*phi*(1+phi)+(1-phi)**2)
    u = .25*(p['tauyy']-p['tauxx'])+.5*G*(p['Ayy']-p['Axx'])
    v = .5*p['tauxy']+G*p['Axy']
    def force(V):
        y = dt*abs(V)/.01
        age = 10*np.exp(-y)+dt*(-np.expm1(-y)/y if y else 1.)
        return 3*np.arcsinh(V/.002*np.exp((.6+.02*np.log(.001*age/.01))/.015))+1000*V
    turns = np.linspace(-45,45,257)
    scores = []
    for angle in np.deg2rad(turns):
        tau = u*np.sin(2*(theta+angle))+v*np.cos(2*(theta+angle))
        dn = p['weight']*np.cos(angle)/section['reference_dxi']
        free, M = dn@(F*tau)/(beta*G*dt),dn@F/(beta*G*dt)
        V = brentq(lambda x:x+M*force(x)-free,min(0,free),max(0,free),xtol=1e-39)
        scores.append((rho*p['weight'])@((1-F)*(tau-force(V)))**2*m/(2*G))
    return turns, np.array(scores), int(np.count_nonzero(rho))


def scan(folder):
    p = fields(folder,1)
    k = 781
    p = p[p['section']==k]
    section = read(folder/'tip-planes-0001.csv')[k]
    assert abs(section['xi']-1.76838)<1e-5 and section['beta']==0.
    turns, scores, disk_points = localized_scores(p, section)
    return dict(station=k,points=len(p),disk_points=disk_points,best_deg=turns[np.argmax(scores)],
                left=scores[0],right=scores[-1],incoming=scores[len(scores)//2],
                gain=max(scores)-scores[len(scores)//2])

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    print(json.dumps(scan(parser.parse_args().folder),indent=2))
