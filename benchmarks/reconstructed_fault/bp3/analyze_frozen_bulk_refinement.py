"""Compare the one frozen bulk refinement, with independent profile/chi checks."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from analyze_mechanical_modes import table
from run_mechanical_modes import parameters

HERE = Path(__file__).resolve().parent
BASE = HERE/'first_long_run/mechanical-modes-preflight-fixed'
OUT = HERE/'first_long_run/mechanical-bulk-refinement'


def main():
    fine = OUT/'fine'; capture = OUT/'capture'
    assert 'MECHANICAL MODES VERIFIED' in (fine/'run.log').read_text()
    assert 'FROZEN PROFILE PROLONGATION VERIFIED' in (fine/'run.log').read_text()
    cells = {}
    with (capture/'phase_cells.csv').open() as stream:
        for row in csv.DictReader(stream):
            cells[row['cell']] = np.array([float(row[k]) for k in ('x', 'y', 'h', 'phi0', 'phi1', 'phi2', 'phi3')])
    surface = table(capture/'surface.csv')
    prm = parameters((fine/'parameters.prm').read_text())
    assert prm['Phase field model', 'Geometric function type'] == 'AT1'
    ell = float(prm['Phase field model', 'Length scale'])
    curvature = float(prm['Phase field model', 'Degradation curvature parameter'])
    G = float(prm['Material model', 'Phase field fault', 'Elastic shear moduli'])
    cohesion = float(prm['Material model', 'Phase field fault', 'Cohesions'])
    Gc = float(prm['Material model', 'Phase field fault', 'Critical energy release rates'])
    eta = float(prm['Material model', 'Phase field fault', 'Reference viscosities'])
    dt0 = float(prm['Material model', 'Phase field fault', 'Initial time step'])
    m = Gc/((8./3)*ell*(cohesion**2/(2*G)))
    kappa0 = -eta*np.expm1(-dt0*G/eta)
    step11 = json.loads((HERE/'first_long_run/mechanical-response-analysis/mechanical_response.json').read_text())
    ratio = step11['kappa_ratio']

    def parent(cell):
        while cell not in cells:
            root, digits = cell.split(':'); digits = digits[:-1]
            cell = root.split('_')[0]+f'_{len(digits)}:'+digits
        return cells[cell]

    def check_localization(directory):
        data = []
        for path in sorted(directory.glob('mechanical_mode_qp_rank*.csv')):
            with path.open() as stream:
                data.extend(row for row in csv.DictReader(stream) if row['mode'] == 'broad_3000m')
        assert len({(row['cell'], row['qp']) for row in data}) == len(data)
        values, errors, weights, kap = [], [], [], []
        for row in data:
            x, y, h, *phi = parent(row['cell'])
            a, b = (float(row['x'])-x)/h, (float(row['y'])-y)/h
            value = (1-a)*(1-b)*phi[0]+a*(1-b)*phi[1]+(1-a)*b*phi[2]+a*b*phi[3]
            value = max(value, 0.)
            j, z = int(row['segment']), float(row['xi'])
            Ih = (1-z)*surface['Ih'][j]+z*surface['Ih'][j+1]
            # Independent algebraic form of 1/g-1, using the configured AT1
            # parameters and captured FE coefficients, not an analytic profile.
            expected = m*value*(1+curvature*value)/(1-value)**2/Ih
            actual = float(row['chi'])
            values.append(actual); errors.append(expected-actual)
            weights.append(float(row['JxW']));kap.append(float(row['kappa']))
        w, v, e = map(np.array, (weights, values, errors))
        relative = float(np.sqrt(np.dot(w, e*e)/np.dot(w, v*v)))
        assert relative < 1e-11
        assert np.max(np.abs(np.array(kap)/kappa0-1)) < 1e-14
        return dict(QPs=len(data), chi_relative_weighted_L2=relative,
                    chi_max_absolute_error=float(max(abs(e))), kappa_Pa_s=float(kap[0]))

    checks = {name: check_localization(directory) for name, directory in (('coarse', BASE), ('fine', fine))}
    coarse_modes, fine_modes = table(BASE/'mechanical_modes.csv'), table(fine/'mechanical_modes.csv')
    coarse_nodes, fine_nodes = table(BASE/'mechanical_mode_nodes.csv'), table(fine/'mechanical_mode_nodes.csv')
    results = []
    for c, f in zip(coarse_modes, fine_modes):
        assert c['mode'] == f['mode']
        cn, fn = [nodes[nodes['mode'] == c['mode']] for nodes in (coarse_nodes, fine_nodes)]
        for key in ('node', 'xd', 'deltaV'):
            np.testing.assert_array_equal(cn[key], fn[key])
        entries = {}
        for name, row, data in (('coarse', c, cn), ('fine', f, fn)):
            direct = np.dot(data['deltaV'], data['K_deltaV'])/row['mass_norm']-row['instantaneous_friction']-row['damping']
            relaxation = np.dot(data['deltaV'], data['G_shear_delta_x'])/row['mass_norm']
            assert abs((direct-relaxation)/row['mechanical_shear']-1) < 1e-12
            entries[name] = dict(direct_source=direct*ratio, bulk_relaxation=relaxation*ratio,
                                 net_mechanical_shear=float(row['mechanical_shear']*ratio),
                                 native_initial_mechanical_shear=float(row['mechanical_shear']),
                                 mechanical_fraction_remaining=float(row['mechanical_shear']/direct),
                                 mass_norm=float(row['mass_norm']))
        entries.update(mode=str(c['mode']), fine_over_coarse=float(f['mechanical_shear']/c['mechanical_shear']),
                       change_percent=float(100*(f['mechanical_shear']/c['mechanical_shear']-1)),
                       modal_mass_change_percent=float(100*(f['mass_norm']/c['mass_norm']-1)),
                       common_coarse_mass_shear_change_percent=float(100*(f['mechanical_shear']*f['mass_norm']/(c['mechanical_shear']*c['mass_norm'])-1)))
        results.append(entries)
    cc, fc = [np.atleast_1d(table(path/'frozen_profile_check.csv'))[0] for path in (capture, fine)]
    integrals = {key: float(fc[key]/cc[key]-1) for key in ('phi_integral', 'phi_squared_integral')}
    assert max(abs(v) for v in integrals.values()) < 1e-12
    source_files = [capture/'phase_cells.csv', capture/'surface.csv', fine/'target_cells.txt',
                    BASE/'mechanical_modes.csv', fine/'mechanical_modes.csv',
                    fine/'frozen_profile_check.csv', capture/'frozen_profile_check.csv', fine/'run.prm']
    result = dict(question='One bulk refinement with unchanged physical Q1 localization; no real time steps',
                  coefficient_units='Pa/(m/s)', kappa_initial_Pa_s=kappa0,
                  kappa_step11_Pa_s=kappa0*ratio, target_scale_factor=ratio,
                  scale_note='Same validated uniform-Maxwell scaling applied to both meshes; timestep controller untouched',
                  phase_integral_relative_changes=integrals, phase_max_QP_error=float(fc['phi_max_error']),
                  normalization='Saved nodal I_h retained exactly; coarse and fine chi independently checked against the same captured field',
                  localization_checks=checks, modes=results,
                  fine_native_checks=[{key: row[key].item() for key in fine_modes.dtype.names} for row in fine_modes],
                  execution={name: json.loads((OUT/name/'execution.json').read_text()) for name in ('capture', 'fine')},
                  hashes={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files})
    (OUT/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for col, c in enumerate(coarse_modes):
        for name, data, style in (('97.65625 m', coarse_nodes, '-'), ('48.828125 m', fine_nodes, '--')):
            data = data[(data['mode'] == c['mode']) & (data['xd'] > 14999.9) & (data['xd'] < 18000.1)]
            data = data[np.argsort(data['xd'])]
            axes[0, col].plot(data['xd']/1000, data['deltaV']/1e-12, style, label=name)
            axes[1, col].plot(data['xd']/1000, data['delta_q']/data['mass_row']*ratio/1000, style, label=name)
        axes[0, col].set_title(c['mode']); axes[1, col].set_xlabel('Down dip (km)')
        for ax in axes[:, col]: ax.grid(alpha=.25);ax.legend(fontsize=8)
    axes[0, 0].set_ylabel('Identical velocity direction / 1e-12 m/s')
    axes[1, 0].set_ylabel('Mechanical shear weak average (kPa)')
    fig.suptitle('Frozen bulk refinement; same coarse Q1 phase field, fault I_h and step-11 Maxwell coefficient')
    fig.tight_layout();fig.savefig(OUT/'bulk_refinement_shear.png', dpi=180);plt.close(fig)
    print(json.dumps({key: value for key, value in result.items() if key not in ('hashes', 'fine_native_checks')}, indent=2))


if __name__ == '__main__': main()
