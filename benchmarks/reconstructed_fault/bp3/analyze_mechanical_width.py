"""Mechanical width comparison using actual Q1 inputs and native work weights."""
import argparse
import csv
import json

import numpy as np
from numpy.polynomial.legendre import leggauss
from analyze_mechanical_modes import table
from run_mechanical_width import OUT, SN, XT, NORMAL, VirtualProfile, stationary, G

KAPPA = 4.912611976502280e18
KAPPA0 = -1e26*np.expm1(-4e6*G/1e26)


def records(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def spectrum_reference(ell, nodes, refinement=1):
    # Infinite, tangentially homogeneous continuum localization: a labeled
    # waveform diagnostic, NOT the exact inclined Cartesian FE operator.
    # For incompressible 2D Stokes, the remaining shear symbol is
    # 4*k_s^2*k_n^2/(k_s^2+k_n^2)^2. Use the actual tapered Q1 input spectrum.
    ds = 6.25/refinement
    dn = 3.125/refinement
    ls, ln = 25600., 6400.
    s = np.arange(int(ls/ds))*ds+16500-ls/2
    n = np.arange(int(ln/dn))*dn-ln/2
    order = np.argsort(nodes['xd'])
    r, phi, m = stationary(ell)
    p = np.interp(abs(n), r, phi, right=0.)
    h = m*p*(1+p)/(1-p)**2
    ih = np.sum(h)*dn
    chi = h/ih
    ks = 2*np.pi*np.fft.rfftfreq(len(s), ds)
    kn = 2*np.pi*np.fft.rfftfreq(len(n), dn)
    # Fourier-transform the actual Q1 hats analytically, rather than aliasing
    # the shortest triangular waveform by sampling it on the FFT grid.
    x, a = nodes['xd'][order], nodes['deltaV'][order]
    line_mass = np.sum(np.diff(x)*(a[:-1]**2+a[:-1]*a[1:]+a[1:]**2)/3)
    active = abs(a) > 0
    assert np.max(abs(np.diff(x)[(abs(a[:-1])+abs(a[1:])) > 0]-100.)) < 1e-7
    transform = 100*np.sinc(ks*100/(2*np.pi))**2 * (
        np.exp(-1j*np.outer(ks, x[active]-16500)) @ a[active])
    sw = abs(transform)**2
    nw = abs(np.fft.rfft(chi))**2
    sw[1:] *= 2
    nw[1:-1] *= 2
    nw *= dn/len(n)
    result = np.zeros(len(ks))
    for j in range(1, len(ks)):
        result[j] = np.dot(nw, 4*ks[j]**2*kn**2/(ks[j]**2+kn**2)**2)
    coefficient = KAPPA*np.dot(sw, result)/(ls*line_mass)
    wavelength = 600. if '600' in str(nodes['mode'][0]) else 200.
    k = 2*np.pi/wavelength
    sinusoid = KAPPA*np.dot(nw, 4*k*k*kn*kn/(k*k+kn*kn)**2)
    return dict(tapered_Q1=coefficient, nominal_sinusoid=sinusoid,
                continuum_Ih=ih, direct=KAPPA*np.dot(chi, chi)*dn,
                input_mass=float(line_mass), spectral_mass_relative=float(np.sum(sw)/(ls*line_mass)-1),
                ds=ds, dn=dn)


def analyze(suffix=''):
    summary, coefficients, columns, profile_lines = {}, [], [], []
    reference_cells, reference_geometry = None, None
    all_nodes = {}
    for ell in (400, 200):
        run = OUT/f'ell{ell}{suffix}'
        assert 'MECHANICAL MODES VERIFIED' in (run/'run.log').read_text()
        assert not (run/'accepted_steps.csv').exists(), 'No accepted history is permitted'
        modes, nodes, parts, surface = [np.atleast_1d(table(run/name)) for name in
            ('mechanical_modes.csv', 'mechanical_mode_nodes.csv', 'mechanical_shear_parts.csv', 'surface.csv')]
        raw = np.concatenate([np.atleast_1d(table(path)) for path in sorted(run.glob('mechanical_mode_qp_rank*.csv'))])
        cells = {}
        for path in run.glob('phase_cells_rank*.csv'):
            for row in csv.DictReader(path.open()):
                assert row['cell'] not in cells
                cells[row['cell']] = np.array([float(row[k]) for k in ('x', 'y', 'h', 'phi0', 'phi1', 'phi2', 'phi3')])
        geometry = np.column_stack([surface['x'], surface['y']])
        if reference_cells is None:
            reference_cells, reference_geometry = cells, geometry
        else:
            assert cells.keys() == reference_cells.keys()
            for key in cells:
                np.testing.assert_array_equal(cells[key][:3], reference_cells[key][:3])
            np.testing.assert_array_equal(geometry, reference_geometry)
        stationary_actual = table(run/'stationary_profile.csv')
        r, phi, m = stationary(ell)
        r_error = float(max(abs(r-stationary_actual['r'])))
        assert r_error < 1e-7 and max(abs(phi-stationary_actual['phi'])) < 2e-15
        nonzero = raw[(raw['mode'] == modes['mode'][0]) & (raw['deltaV'] != 0)]
        sizes = {cells[s][2] for s in nonzero['cell']}
        assert len(sizes) == 1
        hcell = sizes.pop()
        profile = VirtualProfile(ell, hcell)
        nodal_error, chi_error = 0., 0.
        for row in nonzero:
            x, y, hcell, *p = cells[row['cell']]
            a, b = (row['x']-x)/hcell, (row['y']-y)/hcell
            value = (1-a)*(1-b)*p[0]+a*(1-b)*p[1]+(1-a)*b*p[2]+a*b*p[3]
            nodal_error = max(nodal_error, abs(value-profile.q1(row['x'], row['y'])))
            j, z = int(row['segment']), row['xi']
            ih = (1-z)*surface['Ih'][j]+z*surface['Ih'][j+1]
            chi = m*value*(1+value)/(1-value)**2/ih
            chi_error = max(chi_error, abs(chi-row['chi']))
        assert nodal_error < 2e-11 and chi_error < 1e-13
        xd = (100000-surface['y'])/SN
        order = np.argsort(xd)
        for s in np.linspace(15000., 18000., 61):
            origin = np.array([XT-.5*s, 100000-SN*s])
            j = profile.integrate(origin, -profile.extent, profile.extent)
            ih = np.interp(s, xd[order], surface['Ih'][order])
            columns.append(dict(ell=ell, xd=s, actual_Q1_column_h=j, interpolated_Ih=ih,
                                normalization_error=j/ih-1))
        for s in (16500., 16525., 16550., 16575.):
            ih = np.interp(s, xd[order], surface['Ih'][order])
            n = np.linspace(-1000, 1000, 801)
            point = np.array([XT-.5*s, 100000-SN*s])[:, None]+NORMAL[:, None]*n
            p = profile.q1(point[0], point[1])
            chi = m*p*(1+p)/(1-p)**2/ih
            for z, v, c in zip(n, p, chi):
                profile_lines.append(dict(ell=ell, xd=s, normal=z, phi=v, chi=c, Ih=ih))
        for row, part in zip(modes, parts):
            mode = str(row['mode'])
            values = raw[raw['mode'] == mode]
            node = nodes[nodes['mode'] == mode]
            all_nodes[ell, mode] = node
            assert len(set(zip(values['cell'], values['qp']))) == len(values)
            assert max(abs(values['kappa']/KAPPA0-1)) < 1e-14
            mass = np.sum(values['weight']*values['deltaV']**2)
            assert abs(mass/row['mass_norm']-1) < 1e-12
            direct = KAPPA*np.sum(values['weight']*values['chi']*values['deltaV']**2)/mass
            relaxation = part['bulk_relaxation']*KAPPA/KAPPA0
            net = row['mechanical_shear']*KAPPA/KAPPA0
            assert abs((direct-relaxation)/net-1) < 1e-11
            # Independently enumerate all bulk QPs in the probe region, not
            # merely the associated/exported subset. Measure missing support
            # with the same native rule, separately from ray integration error.
            c = np.array(list(cells.values()))
            center_s = (XT-c[:, 0]-c[:, 2]/2)*.5+(100000-c[:, 1]-c[:, 2]/2)*SN
            c = c[(center_s+c[:, 2] > 15000) & (center_s-c[:, 2] < 18000)]
            z, w = leggauss(3)
            z, w = (z+1)/2, w/2
            total_mass, total_direct = 0., 0.
            for ix in range(3):
                for iy in range(3):
                    xq, yq = c[:, 0]+c[:, 2]*z[ix], c[:, 1]+c[:, 2]*z[iy]
                    s = (XT-xq)*.5+(100000-yq)*SN
                    p = ((1-z[ix])*(1-z[iy])*c[:, 3]+z[ix]*(1-z[iy])*c[:, 4]
                         +(1-z[ix])*z[iy]*c[:, 5]+z[ix]*z[iy]*c[:, 6])
                    ih = np.interp(s, xd[order], surface['Ih'][order])
                    v = np.interp(s, node['xd'][order], node['deltaV'][order])
                    chi = m*p*(1+p)/(1-p)**2/ih
                    weight = c[:, 2]**2*w[ix]*w[iy]
                    total_mass += np.sum(weight*chi*v*v)
                    total_direct += KAPPA*np.sum(weight*chi*chi*v*v)
            for key, limit in (('fresh_relative', 1e-10), ('work_pair_relative', 1e-8), ('action_relative', 1e-8)):
                assert row[key] < limit
            prediction = spectrum_reference(ell, node)
            check = spectrum_reference(ell, node, 2)
            prediction['grid_check_relative'] = check['tapered_Q1']/prediction['tapered_Q1']-1
            coefficients.append(dict(ell=ell, mode=mode, direct=direct, relaxation=relaxation, net=net,
                                     signed_dn_us=part['signed_d_us_dn']*KAPPA/KAPPA0,
                                     signed_ds_un=part['signed_d_un_ds']*KAPPA/KAPPA0,
                                     mass=mass, iterations=int(row['iterations']),
                                     fresh_relative=row['fresh_relative'], work_pair_relative=row['work_pair_relative'],
                                     action_relative=row['action_relative'], split_relative=part['closure_relative'],
                                     continuum_Q1_prediction=check['tapered_Q1'],
                                     continuum_sinusoid=check['nominal_sinusoid'],
                                     spectrum_grid_check_relative=prediction['grid_check_relative'],
                                     spectrum_mass_relative=check['spectral_mass_relative'],
                                     mass_over_Q1_line_mass=mass/check['input_mass'],
                                     missing_native_mass_fraction=1-mass/total_mass,
                                     missing_native_direct_fraction=1-direct*mass/total_direct))
        norms = [abs(c['normalization_error']) for c in columns if c['ell'] == ell]
        summary[str(ell)] = dict(cells=len(cells), patch_h=hcell, ell_over_h=ell/hcell,
                                production_profile_radius_error=r_error,
                                FE_profile_QP_error=nodal_error, chi_algebra_error=chi_error,
                                max_column_normalization_error=max(norms),
                                execution=json.loads((run/'execution.json').read_text()))
    for mode in modes['mode']:
        for key in ('xd', 'node', 'deltaV'):
            np.testing.assert_array_equal(all_nodes[400, mode][key], all_nodes[200, mode][key])
    records(OUT/f'coefficients{suffix}.csv', coefficients)
    records(OUT/f'normalization_columns{suffix}.csv', columns)
    records(OUT/f'localization_profiles{suffix}.csv', profile_lines)
    result = dict(kappa=KAPPA, kappa_initial=KAPPA0, profiles=summary, coefficients=coefficients)
    (OUT/f'comparison{suffix}.json').write_text(json.dumps(result, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ell in (400, 200):
        p = [v for v in profile_lines if v['ell'] == ell and v['xd'] == 16500.]
        axes[0, 0].plot([v['normal'] for v in p], [v['phi'] for v in p], label=f'ell={ell} m')
        axes[0, 1].plot([v['normal'] for v in p], [v['chi'] for v in p], label=f'ell={ell} m')
        c = [v for v in columns if v['ell'] == ell]
        axes[1, 0].plot([v['xd']/1000 for v in c], [v['normalization_error']*100 for v in c], label=f'ell={ell} m')
    for mode in modes['mode']:
        node = all_nodes[400, mode]
        keep = (node['xd'] >= 14999.9) & (node['xd'] <= 18000.1)
        order = np.argsort(node['xd'][keep])
        axes[1, 1].plot(node['xd'][keep][order]/1000, node['deltaV'][keep][order]/1e-12, label=mode)
    axes[0, 0].set(xlabel='Normal distance (m)', ylabel='Realized Q1 phi at xd=16.5 km')
    axes[0, 1].set(xlabel='Normal distance (m)', ylabel='Realized chi (1/m)')
    axes[1, 0].set(xlabel='Down dip (km)', ylabel='Column integral chi minus one (%)')
    axes[1, 1].set(xlabel='Down dip (km)', ylabel='Actual Q1 input / 1e-12 m/s')
    for ax in axes.flat:
        ax.legend(fontsize=8)
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(OUT/f'profiles_and_inputs{suffix}.png', dpi=160)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suffix', default='')
    analyze(parser.parse_args().suffix)
