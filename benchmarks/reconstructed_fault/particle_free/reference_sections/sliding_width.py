"""Compare the retained 5–10 ms interval on nested strips in the 30-degree fixture."""
import argparse
import json
from pathlib import Path

import numpy as np

from compare import fields, read
from strip_width import phase_polynomials


def virgin_scores(old, end, planes, log):
    """Exact integrated angular score at this interval's intact prospective sections."""
    theta = np.pi/6
    center = np.array([1 + .5*np.cos(theta), .5 + .5*np.sin(theta)])
    assert 'Direction neighborhood: center=1.43301 0.75, radius=0.5,' in log
    radius, G = .5, 1e6
    m = (26.6666666667 / ((8/3)*.25)) / (1000**2/(2*G))
    rows = []
    for k in np.flatnonzero(planes['parent_formed'] == 0):
        mask = end['section'] == k
        p, q = old[mask], end[mask]
        rho = np.maximum(0, 1-((q['x']-center[0])**2+(q['y']-center[1])**2)/radius**2)**2
        if not np.any(rho):
            assert not planes['formed'][k]
            continue
        # On the complete eligible strip F=0, so f=V=0 at every angle.
        # With A=0 the physical midpoint stress is the trial stress predictor.
        assert np.all(q['stage_phi'] == 0)
        for field in ('Axx', 'Ayy', 'Axy', 'Azz'):
            assert np.all(p[field] == 0) and np.all(q[field] == 0)
        u = .25*(p['tauyy']+q['tauyy']-p['tauxx']-q['tauxx'])
        v = .5*(p['tauxy']+q['tauxy'])
        np.testing.assert_allclose(v, q['stage_tauxy'], atol=1e-12, rtol=1e-14)
        c, s = np.cos(2*theta), np.sin(2*theta)
        u, v = c*u-s*v, s*u+c*v
        w = rho*q['weight']*m/(2*G)
        uu, uv, vv = w @ (u*u), w @ (u*v), w @ (v*v)
        # J = const + (vv-uu)/2 cos(4 beta) + uv sin(4 beta).
        beta = .25*np.arctan2(2*uv, vv-uu)
        peak = w @ (u*np.sin(2*beta)+v*np.cos(2*beta))**2
        native_deg = np.rad2deg(planes['beta'][k])
        assert abs(np.rad2deg(beta)-native_deg) < .001
        rows.append(dict(station=int(k), xi=planes['xi'][k], selected_deg=native_deg,
                         raw_deg=np.rad2deg(beta), margin_deg=45-abs(np.rad2deg(beta)),
                         score=peak, edge_gap=peak-uu, formed=bool(planes['formed'][k])))
    assert f'eligible={len(rows)}\n' in log
    return rows


def measure(parent, current):
    old, end = fields(parent, 1), fields(current, 2)
    for f in ('cell','q','x','y','weight','section'):
        np.testing.assert_array_equal(old[f], end[f])
    prev = read(parent/'tip-planes-0001.csv')
    planes = read(current/'tip-planes-0002.csv')
    rates = read(current/'tip-sections-0002.csv')
    ledger, = read(current/'bulk-accepted.csv')
    old_ledger, = read(parent/'bulk-accepted.csv')
    assert ledger['time'] == .01 and old_ledger['time'] == ledger['dt'] == .005
    assert ledger['publications'] == 2 and old_ledger['publications'] == 1
    for f in ('slip','age','formed'):
        np.testing.assert_array_equal(prev[f], planes['parent_'+f])
    formed = prev['formed'] == 1
    np.testing.assert_array_equal(planes['beta'][formed], prev['beta'][formed])
    ds = planes['ds']
    slip = ds*(planes['slip']-prev['slip'])
    np.testing.assert_allclose(slip, ds*ledger['dt']*rates['V'], rtol=1e-13, atol=1e-23)
    physical_error = ledger['Wext']-ledger['energy_change']-ledger['Db']-ledger['Dc']-ledger['Df']
    assert abs(physical_error-ledger['phase_remainder']) < 2e-14
    power, = read(current/'tip-power.csv')
    log = (current.parent/'run.log').read_text()
    scores = virgin_scores(old, end, planes, log)
    k = np.argmax(abs(rates['V']))
    result = dict(strips=len(ds), signed_increment_m2=sum(slip), cumulative_slip_m2=ds@planes['slip'],
                  absolute_deformation_m2=ledger['dt']*(ds@rates['integral_abs_w']),
                  peak_rate_m_s=rates['V'][k], peak_xi_m=planes['xi'][k], fracture_J_m=ledger['fracture'],
                  added_fracture_J_m=ledger['fracture']-old_ledger['fracture'], external_work_J_m=ledger['Wext'],
                  friction_J_m=power['friction'], radiation_J_m=power['radiation'],
                  physical_error_J_m=physical_error, ledger_error_J_m=ledger['ledger_error'],
                  work_adjoint_J_m=power['power_adjoint_error'], KKT=ledger['phase_KKT'],
                  mechanics=ledger['mechanical_residual'], newly_formed=int(sum(planes['formed']-prev['formed'])),
                  prospective=scores)
    return result, planes, slip, phase_polynomials(old, end)


def compare(paths):
    coarse, a, sa, ca = measure(*paths[:2])
    fine, b, sb, cb = measure(*paths[2:])
    pairs = sb.reshape(len(sa),2).sum(axis=1)
    np.testing.assert_allclose(b['xi'].reshape(-1,2).mean(axis=1), a['xi'], atol=1e-15, rtol=0)
    np.testing.assert_allclose(b['reference_dxi'].reshape(-1,2).sum(1), a['reference_dxi'], rtol=1e-14)
    norms = 1 / (64*np.outer([1,3,5], [1,3,5]).ravel())
    assert ca.keys() == cb.keys()
    coeff_diff = np.stack([cb[k]-ca[k] for k in ca])
    phase_diff = np.sqrt(np.einsum('cij,i,cij->j', coeff_diff, norms, coeff_diff))
    coarse_increment = np.stack([v[:,1]-v[:,0] for v in ca.values()])
    return dict(coarse=coarse, fine=fine, parent_phase_L2_m=phase_diff[0], endpoint_phase_L2_m=phase_diff[1],
                coarse_increment_phase_L2_m=np.sqrt(np.sum(coarse_increment**2*norms)),
                common_strips=[dict(k=k, xi=a['xi'][k], width=a['reference_dxi'][k], coarse=x, fine=y,
                                    difference=y-x) for k,(x,y) in enumerate(zip(sa,pairs))],
                relative_L1_common_slip=sum(abs(pairs-sa))/sum(abs(sa)))


def plot(result, path):
    import matplotlib.pyplot as plt
    rows = result['common_strips']
    xi = np.array([r['xi'] for r in rows])
    width = rows[0]['width']
    edges = np.r_[xi-width/2, xi[-1]+width/2]
    fig, (ax, delta) = plt.subplots(2,1,sharex=True,figsize=(7,4.9),layout='constrained')
    for key,label in [('coarse','18 strips'),('fine','36 strips, summed in pairs')]:
        ax.stairs([r[key]*1e6 for r in rows], edges, label=label, lw=1.7,
                  linestyle='-' if key=='coarse' else '--', baseline=None)
    ax.set_ylabel('Integrated slip increment (mm²)')
    ax.legend()
    delta.bar(xi,[r['difference']*1e6 for r in rows],width=.9*width,color='#3f7185')
    delta.axhline(0,color='.4',lw=.7)
    delta.set(xlabel='Reference coordinate ξ (m)',ylabel='Fine minus coarse (mm²)')
    for panel in (ax, delta):
        panel.axvline(np.cos(np.pi/6)+.5*np.sin(np.pi/6)+.5,ls=':',color='.35')
    delta.text(.73,.08,'Original tip',transform=delta.transAxes,fontsize=9)
    fig.savefig(path,dpi=155)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folders', metavar='OUTPUT', nargs=4, type=Path,
                        help='Coarse parent, coarse endpoint, fine parent, fine endpoint')
    parser.add_argument('--figure', type=Path)
    args = parser.parse_args()
    result = compare(args.folders)
    if args.figure:
        plot(result,args.figure)
    print(json.dumps(result,indent=2))
