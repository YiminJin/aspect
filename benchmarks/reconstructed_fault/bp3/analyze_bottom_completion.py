"""Compare the one completion run against preserved uniform-sliding evidence."""
import json
import numpy as np
from scipy.linalg import solve_banded
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import analyze_uniform_sliding as audit
import bottom_completion as virtual

BASE=virtual.BASE
RUN=virtual.OUT
OUT=RUN/'analysis'

def main():
    OUT.mkdir(exist_ok=True)
    audit.RUN=RUN;audit.OUT=OUT
    if not (OUT/'summary.json').exists():audit.analyze_run()
    read,cat,write=audit.read,audit.cat,audit.write
    baseline=read(BASE/'fault_0.csv');candidate=read(RUN/'fault_0.csv')
    for key in ('x','y','Theta','V','C'):np.testing.assert_array_equal(candidate[key],baseline[key])
    base_clock=read(BASE/'accepted_steps.csv');clock=read(RUN/'accepted_steps.csv')
    for key in ('time','dt'):np.testing.assert_array_equal(clock[key],base_clock[key])
    p=cat(RUN.glob('ih_bottom_completion_rank*.csv'))
    p=audit.select(p,np.argsort(p['id']));np.testing.assert_array_equal(p['id'],np.arange(len(p['id'])))
    n=len(baseline['V']);band=np.zeros((3,n));rhs=np.zeros((3,n))
    for j,z,w,inside,outside in zip(p['segment'].astype(int),p['xi'],p['surface_weight'],p['inside'],p['outside']):
        N=np.array([1-z,z]);band[1,j:j+2]+=w*N*N
        band[0,j+1]+=w*N[0]*N[1];band[2,j]+=w*N[0]*N[1]
        rhs[0,j:j+2]+=w*N*inside;rhs[1,j:j+2]+=w*N*outside
        rhs[2,j:j+2]+=w*N*(inside+outside)
    projected=np.array([solve_banded((1,1),band,r) for r in rhs])
    np.testing.assert_allclose(projected[0],baseline['Ih'],rtol=1e-12,atol=1e-8)
    np.testing.assert_allclose(projected[2],candidate['Ih'],rtol=1e-12,atol=1e-8)
    np.testing.assert_allclose(candidate['Ih']-baseline['Ih'],projected[1],rtol=1e-9,atol=1e-8)
    preflight=json.loads((RUN/'preflight.json').read_text())
    complete=p['y']>=.5*preflight['enclosure']
    assert np.all(p['outside'][complete]==0)
    checks=[];reference_full=virtual.integrate(np.array([virtual.xt-(100000-1000)/virtual.sn*.5,1000.]),
                                             -preflight['enclosure'],preflight['enclosure'])
    for i in np.flatnonzero(p['y']<2000):
        origin=np.array([p['x'][i],p['y'][i]])
        full=virtual.integrate(origin,-preflight['enclosure'],preflight['enclosure'])
        checks.append(dict(id=int(p['id'][i]),y=p['y'][i],production=p['completed'][i],virtual_full=full,
                           error=p['completed'][i]-full))
    audit.records(OUT/'profile_completion_check.csv',checks)
    # Keep this failed precision check visible. The physical in-box remote
    # integrator is unchanged and already has a documented cell/reference
    # discrepancy; do not retune it or suppress the measured comparison.
    full_column_check=bool(max(abs(r['error']) for r in checks)<1e-5)
    # A fixed complete FE reference column split at each clipped height must
    # remain constant, unlike small grid-phase variations of 2-D Q1 columns.
    origin=np.array([virtual.xt-(100000-1000)/virtual.sn*.5,1000.]);a=preflight['enclosure']
    reference_error=0.
    for height in [0,50,100,200,300,400,500]:
        cut=np.clip(-2*height,-a,a)
        total=virtual.integrate(origin,-a,cut)+virtual.integrate(origin,cut,a)
        reference_error=max(reference_error,abs(total-reference_full))
    assert reference_error<1e-7
    delta=candidate['Ih']-baseline['Ih']
    projection=dict(inside_projection_matches_baseline=float(np.max(abs(projected[0]-baseline['Ih']))),
        completed_projection_matches_candidate=float(np.max(abs(projected[2]-candidate['Ih']))),
        maximum_complete_column_error=max(abs(r['error']) for r in checks),
        full_column_absolute_check_1e_5_m_passed=full_column_check,
        fixed_reference_full=reference_full,fixed_reference_split_error=reference_error)
    for y in (400,500,1000,2000,50000):
        projection[f'max_nodal_change_above_y_{y}_m']=float(np.max(abs(delta[candidate['y']>=y])))
    write(OUT/'Ih_projection_difference.csv',dict(y=candidate['y'],baseline=baseline['Ih'],completed=candidate['Ih'],delta=delta))
    base_rows=list(__import__('csv').DictReader((BASE/'analysis/uniform_bulk_summary.csv').open()))
    rows=list(__import__('csv').DictReader((OUT/'uniform_bulk_summary.csv').open()))
    comparison=[]
    for b,c in zip(base_rows,rows):
        assert (b['step'],b['region'])==(c['step'],c['region'])
        r=dict(step=int(c['step']),region=c['region'])
        for key in ('elastic_strain_rms','elastic_over_crack_rms','u_reference_error_rms_over_Vp','chi_max','Ih_min','Ih_max','p_min','p_max','sigma_min','sigma_max'):
            r['baseline_'+key]=float(b[key]);r['completed_'+key]=float(c[key])
        comparison.append(r)
    audit.records(OUT/'comparison.csv',comparison)
    # Mechanical integration stays in-box. At each bottom profile the
    # physical fraction uses in-box h and the ACTUAL completed Q1 denominator.
    j=p['segment'].astype(int);z=p['xi']
    hat=(1-z)*candidate['Ih'][j]+z*candidate['Ih'][j+1]
    write(OUT/'physical_profile_fraction.csv',dict(id=p['id'],y=p['y'],inside=p['inside'],outside=p['outside'],Ih_hat=hat,
        physical_integral_chi=p['inside']/hat,full_integral_chi=p['completed']/hat))
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for root,label in [(BASE,'baseline'),(RUN,'completed')]:
        f=read(root/'fault_2.csv');mask=f['y']<2000
        axes[0].plot(f['y'][mask],f['Ih'][mask],label=label)
        f,w=audit.moments(root,2);axes[1].plot(f['y'][mask],(w['sigma_weak_mean'][mask]-50e6)/1e3,label=label+' weak')
        raw=audit.raw_samples(root,2,f);mask=raw['surface_y']<2000
        axes[1].scatter(raw['surface_y'][mask],(raw['sigma_n'][mask]-50e6)/1e3,s=8)
        vals=base_rows if root==BASE else rows
        vals=[r for r in vals if r['step']=='2' and r['region'].startswith('bottom_')]
        axes[2].plot([100,600,1500],[float(r['elastic_strain_rms']) for r in vals],label=label)
    for ax in axes:ax.set_xlabel('Height above bottom (m)');ax.legend(fontsize=8)
    axes[0].set_ylabel('I_h (m)');axes[1].set_ylabel('sigma_n - 50 MPa (kPa)')
    axes[2].set_ylabel('RMS strain mismatch (1/s)');axes[2].set_yscale('log')
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
    (OUT/'completion_verification.json').write_text(json.dumps(projection,indent=2)+'\n')
    print(json.dumps(projection,indent=2))

if __name__=='__main__':main()
