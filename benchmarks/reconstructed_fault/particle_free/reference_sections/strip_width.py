"""Compare the first registration on nested longitudinal strips in the 30-degree fixture."""
import argparse
import json
from pathlib import Path

import numpy as np

from compare import fields, read
from edge_score import localized_scores


def phase_polynomials(parent, end):
    """Recover native Q2 on each Cartesian cell in its tensor Legendre basis."""
    from numpy.polynomial.legendre import legvander2d
    coefficients = {}
    max_residual = 0.
    for cell in np.unique(end['cell']):
        mask = end['cell'] == cell
        # In this 16 x 8 mesh h=1/8 along both coordinates.
        x = 2*np.mod(end['x'][mask]*8, 1)-1
        y = 2*np.mod(end['y'][mask]*8, 1)-1
        basis = legvander2d(x, y, [2, 2])
        samples = np.column_stack((parent['phi'][mask], end['phi'][mask]))
        coeff, _, rank, _ = np.linalg.lstsq(basis, samples, rcond=None)
        assert rank == 9
        max_residual = max(max_residual, np.max(np.abs(basis @ coeff - samples)))
        coefficients[cell] = coeff
    assert max_residual < 1e-13 and len(coefficients) == 128
    return coefficients


def measure(folder):
    parent, end = (fields(folder, i) for i in (0, 1))
    for key in ('x', 'y', 'weight', 'section'):
        np.testing.assert_array_equal(parent[key], end[key])
    for quantity in ('tau', 'A'):
        for component in ('xx', 'yy', 'zz', 'xy'):
            assert np.all(parent[quantity + component] == 0)
    planes = read(folder/'tip-planes-0001.csv')
    section = read(folder/'tip-sections-0001.csv')
    ledger, = read(folder/'bulk-accepted.csv')
    initial, = read(folder/'tip-initial.csv')
    assert ledger['publications'] == ledger['step'] == 1 and ledger['dt'] == .005

    # The analytic forward tip and the native contiguous, exact-positive seed rule.
    tip = np.cos(np.pi/6) + .5*np.sin(np.pi/6) + .5
    seed_support = np.zeros(len(planes), dtype=bool)
    np.logical_or.at(seed_support, parent['section'], parent['phi'] > 0)
    anchor = np.searchsorted(planes['xi'], tip-.25)
    assert seed_support[anchor]
    first, last = anchor, anchor+1
    while first and seed_support[first-1]:
        first -= 1
    while last < len(planes) and planes['xi'][last] <= tip and seed_support[last]:
        last += 1
    seed = np.zeros(len(planes), dtype=bool)
    seed[first:last] = True
    assert np.all(planes['formed'][seed] == 1)

    rows = []
    for k in np.flatnonzero(~seed):
        mask = end['section'] == k
        point = end[mask]
        angle, score, disk_points = localized_scores(point, planes[k])
        if not disk_points:
            assert not planes['formed'][k]
            continue
        i = np.argmax(score)
        rows.append(dict(section=int(k), xi=planes['xi'][k], registered=bool(planes['formed'][k]),
                         accepted_deg=np.rad2deg(planes['beta'][k]), score_max_deg=angle[i],
                         score_max=score[i], edge_gap=score[i]-max(score[0], score[-1]),
                         seed_phase_max=float(np.max(parent['phi'][mask])), phase_max=float(np.max(point['phi'])),
                         increment_max=float(np.max(point['phi']-parent['phi'][mask]))))
    ds = section['ds']
    signed_slips = ds*section['slip']
    return dict(width=planes['reference_dxi'][0], strips=len(planes), points=len(end),
                seed_strips=int(np.count_nonzero(seed)), eligible=len(rows),
                new_planes=sum(r['registered'] for r in rows), directions=rows,
                signed_slip=sum(signed_slips), integrated_absolute_local_slip=ledger['dt']*(ds@section['integral_abs_w']),
                fracture0=initial['fracture'], added_fracture=ledger['fracture']-initial['fracture'],
                work=ledger['Wext'], ledger_error=ledger['ledger_error'], kkt=ledger['phase_KKT'],
                mechanics=ledger['mechanical_residual'],
                work_adjoint=read(folder/'tip-power.csv')[0]['power_adjoint_error']), signed_slips, planes, phase_polynomials(parent, end)


def compare(coarse, fine):
    a, sa, pa, ca = measure(coarse)
    b, sb, pb, cb = measure(fine)
    assert len(pb) == 2*len(pa)
    np.testing.assert_allclose(pb['reference_dxi'].reshape(-1,2).sum(axis=1), pa['reference_dxi'], rtol=1e-14)
    np.testing.assert_allclose(pb['xi'].reshape(-1,2).mean(axis=1), pa['xi'], atol=1e-15, rtol=0.)
    assert ca.keys() == cb.keys()
    norms = np.ravel(1/(np.outer([1,3,5], [1,3,5])*64))  # h^2 / ((2i+1)(2j+1))
    diff = np.stack([cb[k]-ca[k] for k in ca])
    seed_norm, end_norm = np.sqrt(np.einsum('cij,i,cij->j', diff, norms, diff))
    assert seed_norm < 1e-13
    delta = np.stack([c[:,1]-c[:,0] for c in ca.values()])
    increment_norm = np.sqrt(np.sum(delta**2*norms))
    return dict(coarse=a, fine=b, seed_L2_difference=seed_norm, end_phase_L2_difference=end_norm,
                coarse_phase_increment_L2=increment_norm,
                max_coarse_strip_integrated_slip_difference=float(np.max(abs(sb.reshape(-1,2).sum(axis=1)-sa))))


def plot(result, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.8), sharex=True, layout='constrained')
    for i, key in enumerate(('coarse', 'fine')):
        case = result[key]
        rows = case['directions']
        xi = [r['xi'] for r in rows]
        color = ('#176798', '#bb5426')[i]
        label = f"{case['strips']} strips"
        axes[0].plot(xi, [r['score_max_deg'] for r in rows], ':', color=color, marker='+',
                     label=label+' · pointwise score scan')
        axes[0].plot(xi, [r['accepted_deg'] for r in rows], '-', color=color, label=label+' · native angle')
        for r in rows:
            axes[0].plot(r['xi'], r['accepted_deg'], marker='o', ms=5, color=color,
                         mfc=color if r['registered'] else 'white')
        axes[1].semilogy(xi, [r['score_max'] for r in rows], 'o-', color=color, label=label)
    axes[0].axhline(-45, color='.5', ls='--', lw=1)
    axes[0].set(ylabel='Relative plane angle (degrees)', ylim=(-48, 5))
    axes[0].legend(fontsize=8, ncols=2, loc='lower right')
    axes[1].set(xlabel='Reference coordinate ξ (m)', ylabel='Peak integrated score (J/m)')
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=.18)
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('coarse_output', type=Path)
    parser.add_argument('fine_output', type=Path)
    parser.add_argument('--figure', type=Path)
    args = parser.parse_args()
    result = compare(args.coarse_output, args.fine_output)
    if args.figure:
        plot(result, args.figure)
    print(json.dumps(result, indent=2))
