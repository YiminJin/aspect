"""Offline gates for the configured-Dc, narrow-band mechanical comparison."""
import csv
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from analyze_mechanical_modes import table
from analyze_mechanical_width import records
from run_mechanical_width import VirtualProfile, stationary, G, SN, XT, NORMAL

OUT = Path(__file__).resolve().parent/'length-scale-study'
KAPPA = -1e26*np.expm1(-4e6*G/1e26)


def kernel(k, dn=.125):
    # Inverse normal Fourier transform of the incompressible shear symbol:
    # |k| (1-|k| |n|) exp(-|k| |n|). Autocorrelation avoids a 2-D FFT grid.
    r, phi, m = stationary(50)
    n = np.arange(-128., 128.+dn/2, dn)
    p = np.interp(abs(n), r, phi, right=0.)
    h = m*p*(1+p)/(1-p)**2
    ih = np.sum(h)*dn
    chi = h/ih
    size = 1 << (2*len(n)-1).bit_length()
    corr = np.fft.irfft(abs(np.fft.rfft(chi, size))**2, size)[:len(n)]*dn
    distance = np.arange(len(n))*dn
    weights = corr*dn
    weights[1:] *= 2
    result = np.zeros(len(k))
    for start in range(0, len(k), 128):
        z = abs(k[start:start+128, None])*distance
        result[start:start+128] = abs(k[start:start+128])*(((1-z)*np.exp(-z)) @ weights)
    rms = np.sqrt(np.sum(h*n*n)*dn/ih)
    return G*result, dict(Ih=ih, rms_width=rms, radius=r[-1])


def spectrum(nodes, dn=.125):
    order = np.argsort(nodes['xd'])
    x, a = nodes['xd'][order], nodes['deltaV'][order]
    mass = np.sum(np.diff(x)*(a[:-1]**2+a[:-1]*a[1:]+a[1:]**2)/3)
    active = abs(a)>0
    assert max(abs(np.diff(x)[abs(a[:-1])+abs(a[1:])>0]-100))<1e-7
    length = 25600.
    k = 2*np.pi*np.arange(4097)/length
    transform = 100*np.sinc(k*100/(2*np.pi))**2*(np.exp(-1j*np.outer(k, x[active]-16500)) @ a[active])
    weights = abs(transform)**2
    weights[1:] *= 2
    stiffness, profile = kernel(k, dn)
    return float(np.dot(weights, stiffness)/(length*mass)), mass, profile


def moments(profile, origin, order=16):
    e = profile.extent
    cuts = [-e, e]
    for d in range(2):
        limits = origin[d]+NORMAL[d]*np.array([-e,e])
        for j in range(int(np.floor(min(limits)/profile.h)), int(np.ceil(max(limits)/profile.h))+1):
            z = (j*profile.h-origin[d])/NORMAL[d]
            if -e<z<e: cuts.append(z)
    cuts = np.unique(cuts)
    points, weights = leggauss(order)
    a, b = cuts[:-1,None], cuts[1:,None]
    n = (a+b)/2+(b-a)/2*points
    xy = origin[:,None,None]+NORMAL[:,None,None]*n
    p = profile.q1(xy[0],xy[1])
    h = profile.m*p*(1+p)/(1-p)**2
    weighted = (b-a)/2*weights*h
    return np.sum(weighted), np.sum(weighted*n*n)


def analyze():
    coefficients, columns, summaries = [], [], {}
    for label, cell_h in [('probe-candidate',12.20703125),('probe-reference',6.103515625)]:
        run = OUT/label
        assert 'MECHANICAL MODES VERIFIED' in (run/'run.log').read_text()
        assert not (run/'accepted_steps.csv').exists()
        modes, nodes, parts, surface = [np.atleast_1d(table(run/name)) for name in
            ['mechanical_modes.csv','mechanical_mode_nodes.csv','mechanical_shear_parts.csv','surface.csv']]
        raw = np.concatenate([np.atleast_1d(table(p)) for p in sorted(run.glob('mechanical_mode_qp_rank*.csv'))])
        cells = {}
        for path in run.glob('phase_cells_rank*.csv'):
            for row in csv.DictReader(path.open()):
                assert row['cell'] not in cells
                cells[row['cell']] = [float(row[k]) for k in ('x','y','h','phi0','phi1','phi2','phi3')]
        profile = VirtualProfile(50,cell_h)
        production = table(run/'stationary_profile.csv')
        r,p,m = stationary(50)
        assert max(abs(r-production['r']))<1e-7 and max(abs(p-production['phi']))<2e-15
        xd = (100000-surface['y'])/SN
        order = np.argsort(xd)
        _, continuum = kernel(np.array([0.]), .0625)
        normalization, width_error, ih_error = [], [], []
        for s in np.linspace(15000,18000,61):
            origin = np.array([XT-.5*s,100000-SN*s])
            j, second = moments(profile,origin)
            check = moments(profile,origin,32)
            assert abs(check[0]/j-1)<1e-9 and abs(check[1]/second-1)<1e-9
            ih = np.interp(s,xd[order],surface['Ih'][order])
            rms = np.sqrt(second/j)
            normalization.append(abs(j/ih-1))
            width_error.append(abs(rms/continuum['rms_width']-1))
            ih_error.append(abs(j/continuum['Ih']-1))
            columns.append(dict(mesh=label,xd=s,J=j,Ih=ih,normalization_error=j/ih-1,
                                core_phi=float(profile.q1(*origin)),rms_width=rms,
                                rms_relative_error=rms/continuum['rms_width']-1))
        phi_error, chi_error = 0.,0.
        for row in raw[raw['mode']==modes['mode'][0]]:
            x,y,h,*p = cells[row['cell']]
            a,b = (row['x']-x)/h,(row['y']-y)/h
            value=(1-a)*(1-b)*p[0]+a*(1-b)*p[1]+(1-a)*b*p[2]+a*b*p[3]
            phi_error=max(phi_error,abs(value-profile.q1(row['x'],row['y'])))
            j,z=int(row['segment']),row['xi']
            ih=(1-z)*surface['Ih'][j]+z*surface['Ih'][j+1]
            chi_error=max(chi_error,abs(m*value*(1+value)/(1-value)**2/ih-row['chi']))
        assert phi_error<2e-11 and chi_error<1e-13
        for row,part in zip(modes,parts):
            values=raw[raw['mode']==row['mode']]
            node=nodes[nodes['mode']==row['mode']]
            mass=np.sum(values['weight']*values['deltaV']**2)
            assert abs(mass/row['mass_norm']-1)<1e-12
            prediction,line_mass,_=spectrum(node)
            check,_,_=spectrum(node,.0625)
            direct=G*np.sum(values['weight']*values['chi']*values['deltaV']**2)/mass
            relaxation=part['bulk_relaxation']*G/KAPPA
            net=row['mechanical_shear']*G/KAPPA
            assert abs((direct-relaxation)/net-1)<1e-11
            assert max(abs(values['kappa']/KAPPA-1))<1e-14
            for name,limit in [('fresh_relative',1e-10),('work_pair_relative',1e-8),('action_relative',1e-8)]:
                assert row[name]<limit
            coefficients.append(dict(mesh=label,mode=str(row['mode']),elastic_Pa_per_m=net,
                direct=direct,bulk_relaxation=relaxation,signed_dn_us=part['signed_d_us_dn']*G/KAPPA,
                signed_ds_un=part['signed_d_un_ds']*G/KAPPA,continuum_actual_Q1=check,
                continuum_sampling_change=check/prediction-1,relative_to_continuum=net/check-1,
                weighted_mass=mass,mass_over_exact_line_mass=mass/line_mass,
                iterations=int(row['iterations']),fresh_relative=row['fresh_relative'],
                work_relative=row['work_pair_relative'],action_relative=row['action_relative']))
        summaries[label]=dict(cells=len(cells),h=cell_h,continuum=continuum,
            max_column_normalization_error=max(normalization),max_rms_width_error=max(width_error),
            max_column_Ih_error=max(ih_error),FE_profile_error=phi_error,chi_error=chi_error,
            FE_nodal_phase_max=float(np.max(np.array(list(cells.values()))[:,3:])))
        # The reference patch does not touch the physical endpoints: both
        # meshes use their 12.207-m endpoint completion. In-box integrals are
        # deliberately truncated; compare the completed column, not inside=1.
        completion=np.concatenate([np.atleast_1d(table(f)) for f in run.glob('ih_bottom_completion_rank*.csv')])
        endpoint_profile=VirtualProfile(50,12.20703125)
        endpoint_rows=[]
        for row in completion[completion['outside']>0]:
            origin=np.array([row['x'],row['y']]);extent=endpoint_profile.extent
            total=endpoint_profile.integrate(origin,-extent,extent)
            inside=endpoint_profile.integrate(origin,max(-extent,-origin[1]/NORMAL[1]),
                min(extent,(100000-origin[1])/NORMAL[1]))
            endpoint_rows.append(dict(id=int(row['id']),xd=(100000-origin[1])/SN,
                inside_fraction=inside/total,completed_relative_error=row['completed']/total-1,
                inside_relative_error=row['inside']/inside-1,
                sum_error=row['inside']+row['outside']-row['completed']))
        summaries[label]['endpoint_completion']=endpoint_rows
    comparisons=[]
    for mode in modes['mode']:
        a,b=[c for c in coefficients if c['mode']==mode]
        change=a['elastic_Pa_per_m']/b['elastic_Pa_per_m']-1
        comparisons.append(dict(mode=str(mode),coarse_over_fine_minus_one=change,pass_5_percent=bool(abs(change)<=.05)))
    wavelength=np.arange(200.,3000.1,10.)
    k=2*np.pi/wavelength
    aliases=k[:,None]+2*np.pi*np.arange(-20,21)[None,:]/100
    stiffness,_=kernel(aliases.ravel(),.0625)
    screen=np.sum(stiffness.reshape(aliases.shape)*np.sinc(aliases*100/(2*np.pi))**4,axis=1)/((2+np.cos(k*100))/3)
    records(OUT/'stiffness_screen.csv',[dict(wavelength=l,Q1_elastic_Pa_per_m=v,critical_50MPa=.005*50e6/.024,
        critical_60MPa=.005*60e6/.024) for l,v in zip(wavelength,screen)])
    records(OUT/'coefficients.csv',coefficients)
    records(OUT/'profile_columns.csv',columns)
    result=dict(profiles=summaries,coefficients=coefficients,refinement=comparisons,
        screen_min_Pa_per_m=float(min(screen)),screen_alternating_Pa_per_m=float(screen[0]),
        profile_gate=all(s['max_column_normalization_error']<=.01 and s['max_rms_width_error']<=.01 for s in summaries.values()),
        mechanical_gate=all(c['pass_5_percent'] for c in comparisons))
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for label,h in [('probe-candidate',12.20703125),('probe-reference',6.103515625)]:
        rows=[r for r in columns if r['mesh']==label]
        axes[0].plot([r['xd']/1000 for r in rows],[100*r['rms_relative_error'] for r in rows],label=f'h={h:.3f} m')
        axes[1].plot([r['xd']/1000 for r in rows],[100*r['normalization_error'] for r in rows])
        c=[r for r in coefficients if r['mesh']==label]
        axes[2].plot([1500,600,200],[r['elastic_Pa_per_m']/1e6 for r in c],'o-',label=f'h={h:.3f} m')
    axes[0].axhline(1,color='k',ls='--',lw=.8)
    axes[0].set(xlabel='Down dip (km)',ylabel='Localization RMS-width error (%)')
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Down dip (km)',ylabel='Completed column normalization error (%)')
    theory=[r for r in coefficients if r['mesh']=='probe-reference']
    axes[2].plot([1500,600,200],[r['continuum_actual_Q1']/1e6 for r in theory],'k--',label='Actual-input continuum')
    axes[2].set(xlabel='Nominal wavelength (m)',ylabel='Elastic shear stiffness (MPa/m)')
    axes[2].legend(fontsize=8)
    for ax in axes: ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(OUT/'resolution_gates.png',dpi=160)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    analyze()
