"""Direct pointwise intact-driving score for station 20 of the 30-degree cut fixture."""
import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from compare import fields, read


def scan(folder):
    # First half-duration stage from a stress-free parent in the fixed plane.
    step = fields(folder, 1)
    p = step[step['section'] == 20]
    ds = read(folder/'tip-sections-0001.csv')['ds'][20]
    dt, G = .0025, 1e6
    beta = 1/(1+dt*G/1e8)
    m = (26.6666666667 / ((8/3)*.25)) / (1000**2/(2*G))
    phi, area = p['stage_phi'], p['weight']
    c = m*phi*(1+phi)
    fraction = c/(c+(1-phi)**2)
    u = .25*(p['tauyy']-p['tauxx'])+.5*G*(p['Ayy']-p['Axx'])
    v = .5*p['tauxy']+G*p['Axy']
    angles = np.linspace(-45, 45, 129)
    scores = []
    for turn in np.radians(angles):
        tau = u*np.sin(2*(np.pi/6+turn))+v*np.cos(2*(np.pi/6+turn))
        dn = area*np.cos(turn)/ds
        mobility = dn @ fraction / (beta*G*dt)
        free = dn @ (fraction*tau) / (beta*G*dt)

        def force(rate):
            x = dt*abs(rate)/.01
            age = 10*np.exp(-x)+dt*(-np.expm1(-x)/x if x else 1.)
            return 3*np.arcsinh(rate/.002*np.exp((.6+.02*np.log(.001*age/.01))/.015))+1000*rate

        rate = brentq(lambda v: v + mobility*force(v) - free,
                      min(0., free), max(0., free), xtol=1e-30)
        scores.append(area @ ((1-fraction)*(tau-force(rate)))**2 * m/(2*G))
    return angles, np.array(scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fixed_plane_output', type=Path)
    angles, scores = scan(parser.parse_args().fixed_plane_output)
    best = np.argmax(scores)
    print(f'Maximum at {angles[best]:g} deg, score {scores[best]:.8e}, '
          f'edge scores {scores[0]:.8e} and {scores[-1]:.8e}')
    assert angles[best] == 0 and scores[best] > 1e5*max(scores[0], scores[-1])
