"""Fresh fixed-profile width probes; no accepted timestep or history publication."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import time

import numpy as np
from numpy.polynomial.legendre import leggauss
from prepare_300km_fixture import coordinate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE/'first_long_run/mechanical-width-comparison'
BASE = HERE/'first_long_run/mechanical-velocity-decomposition/fine'
FIXTURE = HERE/'fixtures/modified_bp3_long_run_300km'
MESH = HERE/'first_long_run/mechanical-bulk-refinement/fine/target_cells.txt'
SN = np.sqrt(3.)/2
XT = 50000*(1+.5/SN)
NORMAL = np.array([-SN, .5])
G = 32038120320.


def stationary(ell):
    # Independent implementation of the current AT1 profile table (5000
    # trapezoidal theta panels). Verify against the production export later.
    m = 1e5/((8./3)*ell*(1e12/(2*G)))
    theta = np.linspace(0., np.pi/2, 5000)
    phi = .6*np.cos(theta)**2
    h = lambda p: m*p*(1+p)/(1-p)**2
    hp = lambda p: m*(1+3*p)/(1-p)**3
    hh = h(.6)
    integrand = np.zeros(len(phi))
    integrand[1:-1] = (2*ell*.6*np.sin(theta[1:-1])*np.cos(theta[1:-1])
                      *np.sqrt(hh/(hh*phi[1:-1]-.6*h(phi[1:-1]))))
    integrand[0] = 2*ell*np.sqrt(.6*hh/(.6*hp(.6)-hh))
    integrand[-1] = 2*ell*np.sqrt(.6*hh/(hh-.6*hp(0.)))
    r = np.r_[0., np.cumsum((integrand[1:]+integrand[:-1])*.5*(theta[1]-theta[0]))]
    return r, phi, m


class VirtualProfile:
    def __init__(self, ell, h):
        self.r, self.phi, self.m = stationary(ell)
        self.h = h
        self.extent = self.r[-1]+h*np.sum(abs(NORMAL))

    def q1(self, x, y):
        h = self.h
        i, j = np.floor(x/h), np.floor(y/h)
        a, b = x/h-i, y/h-j
        result = np.zeros(np.broadcast_shapes(np.shape(x), np.shape(y)))
        for u in (0, 1):
            for v in (0, 1):
                r = abs((XT-(i+u)*h)*SN-(100000-(j+v)*h)*.5)
                value = np.interp(r, self.r, self.phi, right=0.)
                result += (a if u else 1-a)*(b if v else 1-b)*value
        return result

    def integrate(self, origin, lo, hi, order=8):
        if lo >= hi:
            return 0.
        cuts = [lo, hi]
        for d in (0, 1):
            ends = origin[d]+NORMAL[d]*np.array([lo, hi])
            for j in range(int(np.floor(min(ends)/self.h)), int(np.ceil(max(ends)/self.h))+1):
                r = (j*self.h-origin[d])/NORMAL[d]
                if lo < r < hi:
                    cuts.append(r)
        cuts = np.unique(cuts)
        nodes, weights = leggauss(order)
        def panel(a, b):
            r = (a+b)/2+(b-a)/2*nodes
            p = origin[:, None]+NORMAL[:, None]*r
            phi = self.q1(p[0], p[1])
            return (b-a)/2*np.dot(weights, self.m*phi*(1+phi)/(1-phi)**2)
        def adaptive(a, b, depth=0):
            low = panel(a, b)
            mid = (a+b)/2
            high = panel(a, mid)+panel(mid, b)
            if abs(high-low) <= 1e-11*max(1., abs(high)):
                return high
            assert depth < 20
            return adaptive(a, mid, depth+1)+adaptive(mid, b, depth+1)
        return sum(adaptive(a, b) for a, b in zip(cuts[:-1], cuts[1:]))


def prepare(ell, mesh=MESH, suffix=''):
    mesh = mesh.resolve()
    out = OUT/f'ell{ell}{suffix}'
    out.mkdir(parents=True)  # Never overwrite an earlier result.
    cells = [coordinate(int(s.split('_')[0]), s.split(':')[1]) for s in mesh.read_text().split()]
    endpoint_h = []
    for top in (False, True):
        xfault = XT if top else XT-100000*.5/SN
        near = [(x, y, h) for x, y, h in cells
                if (y+h == 100000 if top else y == 0) and x <= xfault <= x+h]
        sizes = {h for x, y, h in near}
        assert len(sizes) == 1, sizes
        endpoint_h.append(sizes.pop())
    assert endpoint_h[0] == endpoint_h[1]
    profile = VirtualProfile(ell, endpoint_h[0])
    fault = np.loadtxt(FIXTURE/'fault.txt')[:, :2]
    rows = []
    error = 0.
    for j, (a, b) in enumerate(zip(fault[:-1], fault[1:])):
        for q, z in enumerate((leggauss(3)[0]+1)/2):
            p = (1-z)*a+z*b
            e = profile.extent
            intervals = [(-e, min(-p[1]/NORMAL[1], e)),
                         (max((100000-p[1])/NORMAL[1], -e), e)]
            value = sum(profile.integrate(p, lo, hi) for lo, hi in intervals)
            if value:
                check = sum(profile.integrate(p, lo, hi, 16) for lo, hi in intervals)
                error = max(error, abs(value-check))
            rows.append([3*j+q, *p, value])
    assert error < 1e-6
    with (out/'completion.txt').open('x') as stream:
        stream.write(str(len(rows))+'\n')
        np.savetxt(stream, rows, fmt=['%d', '%.17g', '%.17g', '%.17g'])
    np.savetxt(out/'independent_stationary_profile.csv', np.array([profile.r, profile.phi]).T,
               delimiter=',', header='r,phi', comments='')
    text = (BASE/'run.prm').read_text()
    text = re.sub(r'^set Output directory =.*$', f'set Output directory = {out}', text, flags=re.M)
    text = text.replace('set Length scale = 400', f'set Length scale = {ell}')
    text = text.replace(str(FIXTURE/'completion.txt'), str(out/'completion.txt'))
    text = text.replace(str(MESH), str(mesh))
    max_level = max(len(s.split(':')[1]) for s in mesh.read_text().split())
    text = text.replace('set Initial adaptive refinement = 9', f'set Initial adaptive refinement = {max_level-1}')
    (out/'run.prm').write_text(text)
    record = json.loads((BASE/'launch.json').read_text())
    env = record['environment']
    env.pop('ASPECT_MECHANICAL_FROZEN_PROFILE')
    env['ASPECT_MECHANICAL_WIDTH_PROBE'] = '1'
    env['ASPECT_MECHANICAL_PROBE_NEWTON'] = '0'
    record['command'][-1] = str(out/'run.prm')
    files = [out/'run.prm', out/'completion.txt', mesh, FIXTURE/'fault.txt', FIXTURE/'prestress.txt',
             ROOT/'build-pf-cpdi/aspect-release', HERE/'build/libbp3.release.so',
             ROOT/'benchmarks/reconstructed_fault/performance/build-gmg/libfault_mechanical_modes.release.so',
             ROOT/'tests/reconstructed_fault_mechanical_modes.cc', ROOT/'tests/reconstructed_fault_frozen_profile.h',
             Path(__file__)]
    record['hashes'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    record['profile'] = dict(ell=ell, core_phi=.6, m=profile.m, radius=profile.r[-1],
                             endpoint_h=endpoint_h, completion_order_error=error,
                             nonzero_completion_rows=int(sum(row[3] > 0 for row in rows)))
    (out/'launch.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record['profile'], indent=2))


def execute(ell, suffix=''):
    out = OUT/f'ell{ell}{suffix}'
    record = json.loads((out/'launch.json').read_text())
    for path, digest in record['hashes'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    env = {k: v for k, v in os.environ.items() if not k.startswith('ASPECT_')}
    env.update(record['environment'])
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', DEAL_II_NUM_THREADS='1', LD_BIND_NOW='1')
    start = time.monotonic()
    with (out/'run.log').open('x') as log:
        p = subprocess.run(['timeout', '--kill-after=15', str(record['hard_wall_cap_s'])]+record['command'],
                           cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT)
    log = (out/'run.log').read_text()
    result = dict(seconds=time.monotonic()-start, returncode=p.returncode,
                  expected_noncommitting_stop='MECHANICAL MODES VERIFIED' in log,
                  peak_child_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    (out/'execution.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    assert result['expected_noncommitting_stop'], 'Preserve failed evidence; no automatic retry.'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ell', type=int, choices=(200, 400))
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--mesh', type=Path, default=MESH)
    parser.add_argument('--suffix', default='')
    args = parser.parse_args()
    if args.execute:
        execute(args.ell, args.suffix)
    else:
        prepare(args.ell, args.mesh, args.suffix)
