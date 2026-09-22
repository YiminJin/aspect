"""Offline cell/basis-split integration of the SAVED Q1 phase, without mechanics.

Read the saved nodal phase coefficients, verify the resulting bilinear
polynomials at all nine saved Gauss points, and reintegrate those polynomials.
Zero cells contribute nothing. Never regenerate nodal phase from a profile.
"""
import json
import argparse
import time
import hashlib
import xml.etree.ElementTree as ET
import numpy as np
from numpy.polynomial.legendre import leggauss
from analyze_loading_startup import table, raw_qps
from analyze_loading_tractions import chord, metric
from run_normal_control import BASE
from startup_30km import STUDY
from run_mechanical_width import stationary

OUT=STUDY/'coupling-quadrature'


def clip(poly, direction, bound):
    out=[]
    for a,b in zip(poly,np.roll(poly,-1,axis=0)):
        fa=a@direction-bound;fb=b@direction-bound
        if fa<=0:out.append(a)
        if (fa<0<fb) or (fb<0<fa):out.append(a+(b-a)*fa/(fa-fb))
    return np.array(out).reshape(-1,2)


def run(surface_subdivisions=()):
    OUT.mkdir(exist_ok=True)
    start=time.monotonic()
    profile=table(BASE/'profiles/fault_0.csv')
    native=table(BASE/'work_weak_0.csv')
    points=np.column_stack([profile['x_m'],profile['y_m']]);origin=points[0]
    t=points[1]-origin;t/=np.linalg.norm(t);normal=np.array([-t[1],t[0]])
    s=(points-origin)@t;xd=profile['xd_m'];n=len(s)
    tree=ET.parse(BASE/'reconstructed_faults/reconstructed_faults-00000.vtu')
    I=np.fromstring(tree.find('.//PointData/DataArray[@Name="previous_I_h"]').text,sep=' ')
    rtab,ptab,m=stationary(100.)
    saved_profile=table(BASE/'stationary_profile.csv')
    np.testing.assert_allclose(rtab,saved_profile['r'],rtol=1e-12,atol=1e-10)
    np.testing.assert_allclose(ptab,saved_profile['phi'],rtol=0,atol=1e-14)
    rtab,ptab=saved_profile['r'],saved_profile['phi']
    halfwidth=rtab[-1]
    raw=np.concatenate([raw_qps(p) for p in sorted(BASE.glob('work_qp_0_rank*.csv'))])
    # Include neighboring test supports and normal tails, not just centerlines.
    ss=(np.column_stack([raw['x'],raw['y']])-origin)@t
    selection=(ss>=s[np.argmin(abs(xd-30200))])&(ss<=s[np.argmin(abs(xd-26600))])
    selected=np.unique(raw['cell'][selection]);selected_set=set(selected)
    raw=raw[np.isin(raw['cell'],selected)]
    raw.sort(order=['cell','qp'])
    ids,starts=np.unique(raw['cell'],return_index=True)
    groups={identity:raw[begin:end] for identity,begin,end in zip(ids,starts,np.r_[starts[1:],len(raw)])}
    mesh=np.concatenate([raw_qps(p) for p in sorted(BASE.glob('initial_mesh_*.csv'))])
    phases=np.concatenate([raw_qps(p) for p in sorted(BASE.glob('phase_cells_rank*.csv'))])
    phase_cells={r['cell']:r for r in phases}
    assert len(phase_cells)==len(phases),'Duplicate phase-cell MPI ownership'
    cells={r['cell']:r for r in mesh if r['cell'] in selected_set}
    polygons=[];full_polygons=[];squares=[];fit_error=0.;source_error=0.;baseline=np.zeros((2,n))
    for identity in selected:
        q=groups[identity]
        assert len(q)==9,(identity,len(q))
        cell=cells[identity];center=np.array([cell['x'],cell['y']]);h=cell['h']
        local=(np.column_stack([q['x'],q['y']])-center)/h
        X=np.column_stack([np.ones(9),local[:,0],local[:,1],local[:,0]*local[:,1]])
        p=phase_cells[identity];v=np.array([p['phi'+str(i)] for i in range(4)])
        coef=np.array([sum(v)/4,(-v[0]+v[1]-v[2]+v[3])/2,
                       (-v[0]-v[1]+v[2]+v[3])/2,v[0]-v[1]-v[2]+v[3]])
        fit_error=max(fit_error,float(max(abs(X@coef-q['phi']))))
        iq=q['segment'].astype(int);z=q['xi'];active=q['source_active']==1
        predicted=m*q['phi']*(1+q['phi'])/(1-q['phi'])**2/((1-z)*I[iq]+z*I[iq+1])
        if np.any(active):source_error=max(source_error,float(max(abs(predicted[active]-q['chi'][active]))))
        for end,N in ((0,1-z),(1,z)):
            for power in (1,2):
                baseline[power-1]+=np.bincount(iq[active]+end,weights=q['JxW'][active]*q['chi'][active]**power*N[active],minlength=n)
        # Physical cells and fault knots are separate integration boundaries.
        vertices=(center-origin)+h*np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]])
        squares.append((center-origin,h,coef))
        lo,hi=min(vertices@t),max(vertices@t)
        for j in range(max(0,np.searchsorted(s,lo)-1),min(n-1,np.searchsorted(s,hi))):
            part=clip(vertices,t,s[j+1]);part=clip(part,-t,-s[j])
            if len(part)>=3:full_polygons.append((part,j,center-origin,h,coef))
        poly=clip(vertices,normal,halfwidth);poly=clip(poly,-normal,halfwidth)
        if len(poly)<3:continue
        lo,hi=min(poly@t),max(poly@t)
        for j in range(max(0,np.searchsorted(s,lo)-1),min(n-1,np.searchsorted(s,hi))):
            part=clip(poly,t,s[j+1]);part=clip(part,-t,-s[j])
            if len(part)>=3:polygons.append((part,j,center-origin,h,coef))
    assert fit_error<1e-12 and source_error<1e-12,(fit_error,source_error)
    mask=(xd>=27000)&(xd<=29800)
    np.testing.assert_allclose(baseline[0,mask],native['weight'][mask],rtol=1e-12,atol=0)

    # Bound the absent cells: a nonnegative Q1 field is zero throughout a cell
    # iff every nodal value is zero. All positive cells touching these test
    # supports are represented in the saved raw sample set.
    for p in phases:
        center=np.array([p['x']+p['h']/2,p['y']+p['h']/2])-origin
        coord=center@t;radius=p['h']/2*sum(abs(t))
        if coord+radius>=min(s[mask])-100 and coord-radius<=max(s[mask])+100:
            if max(p['phi'+str(i)] for i in range(4))>0:
                assert p['cell'] in selected_set,p['cell']

    print(f'Validated {len(squares)} cells, {len(polygons)} cell/basis pieces; integrating.',flush=True)
    constant_I=float(np.mean(I[mask]))
    def integrate(order, analytic=False, fixed_I=False, full_profile=False):
        a,w=leggauss(order);a=(a+1)/2;w=w/2
        u,v=np.meshgrid(a,a,indexing='ij');weights=(w[:,None]*w[None,:]*(1-u)).ravel()
        u=u.ravel();v=v.ravel();answer=np.zeros((2,n))
        pieces=full_polygons if full_profile else polygons
        if analytic:
            # The intended distance profile has a cusp at the straight ridge;
            # split it explicitly rather than hiding that kink with high order.
            pieces=[]
            for poly,j,center,h,coef in polygons:
                for direction in (normal,-normal):
                    part=clip(poly,direction,0.)
                    if len(part)>=3:pieces.append((part,j,center,h,coef))
        for poly,j,center,h,coef in pieces:
            for k in range(1,len(poly)-1):
                e1=poly[k]-poly[0];e2=poly[k+1]-poly[0]
                det=abs(e1[0]*e2[1]-e1[1]*e2[0])
                if det<1e-20:continue
                p=poly[0]+u[:,None]*e1+((1-u)*v)[:,None]*e2
                local=(p-center)/h
                phase=coef[0]+coef[1]*local[:,0]+coef[2]*local[:,1]+coef[3]*local[:,0]*local[:,1]
                if analytic:phase=np.interp(abs(p@normal),rtab,ptab,right=0.)
                phase=np.maximum(phase,0.)
                z=(p@t-s[j])/(s[j+1]-s[j]);ih=(1-z)*I[j]+z*I[j+1]
                if fixed_I:ih=constant_I
                chi=m*phase*(1+phase)/(1-phase)**2/ih
                for end,N in ((0,1-z),(1,z)):
                    answer[0,j+end]+=det*np.dot(weights,chi*N)
                    answer[1,j+end]+=det*np.dot(weights,chi**2*N)
        return answer

    def iterated(repetitions):
        a,w=leggauss(3);a=np.concatenate([(i+(a+1)/2)/repetitions-.5 for i in range(repetitions)])
        w=np.tile(w/(2*repetitions),repetitions)
        u,v=np.meshgrid(a,a,indexing='ij');u=u.ravel();v=v.ravel();ww=np.outer(w,w).ravel()
        answer=np.zeros((2,n))
        for center,h,coef in squares:
            p=center+h*np.column_stack([u,v]);coord=p@t;j=np.searchsorted(s,coord)-1
            active=(abs(p@normal)<=halfwidth)&(j>=0)&(j<n-1)
            j=j[active];z=(coord[active]-s[j])/(s[j+1]-s[j])
            phi=np.maximum(coef[0]+coef[1]*u+coef[2]*v+coef[3]*u*v,0.)[active]
            chi=m*phi*(1+phi)/(1-phi)**2/((1-z)*I[j]+z*I[j+1])
            for end,N in ((0,1-z),(1,z)):
                for power in (1,2):
                    answer[power-1]+=np.bincount(j+end,weights=h*h*ww[active]*chi**power*N,minlength=n)
        return answer

    data={'production_Q3':baseline}
    for order in (3,6,10,16):
        data['split_'+str(order)]=integrate(order)
        print(f'Finished split order {order}',flush=True)
    for repetitions in (2,4,8):
        data['iterated_'+str(repetitions)]=iterated(repetitions)
    data['uniform_profile_same_I']=integrate(16,True)
    data['uniform_profile_same_I_32']=integrate(32,True)
    data['FE_constant_I']=integrate(10,False,True)
    data['FE_full_constant_I']=integrate(10,False,True,True)
    full_check=integrate(16,False,True,True)
    full_moment_error=float(np.max(abs(data['FE_full_constant_I'][:,mask]/full_check[:,mask]-1)))
    assert full_moment_error<1e-12
    data['uniform_profile_constant_I']=integrate(16,True,True)
    data['uniform_profile_constant_I_32']=integrate(32,True,True)
    reference=data['split_16'];w=native['weight'][mask]
    summary=dict(cells=len(squares),basis_pieces=len(polygons),halfwidth=float(halfwidth),constant_I=constant_I,
                 saved_phase_fit_error=fit_error,saved_source_error=source_error,
                 full_moment_10_vs_16_error=full_moment_error,levels={})
    for name,values in data.items():
        ratio=values[1]/np.where(values[0]>0,values[0],1)
        output=dict(xd=xd,mass=values[0],chi2=values[1],chi_work=ratio)
        for key in ('mass','chi2','chi_work'):output['chord_'+key]=chord(output[key])
        np.savetxt(OUT/(name+'.csv'),np.column_stack(list(output.values())),delimiter=',',comments='',header=','.join(output))
        summary['levels'][name]={key:dict(mean=float(np.average(output[key][mask],weights=w)),
                 relative_peak_to_peak=float(np.ptp(output[key][mask])/np.mean(output[key][mask])),
                 chord=metric(output['chord_'+key][mask],w)) for key in ('mass','chi2','chi_work')}
        summary['levels'][name]['max_relative_moment_difference_from_split_FE']=float(np.max(abs(values[:,mask]/reference[:,mask]-1)))
    # Correlate only the already-defined neighbor-chord departures. A
    # constant-I diagnostic changes no production data and is not a remedy.
    Ih_weak=(np.roll(I,1)+4*I+np.roll(I,-1))/6
    summary['Ih_nodal_range_m']=[float(min(I[mask])),float(max(I[mask]))]
    summary['chord_correlations']={}
    for name in ('mass','chi2','chi_work'):
        z=reference[0] if name=='mass' else reference[1] if name=='chi2' else reference[1]/np.where(reference[0]>0,reference[0],1)
        summary['chord_correlations'][name]=float(np.corrcoef(chord(z)[mask],-chord(Ih_weak)[mask])[0,1])

    # Independently resolve columns of the saved bilinear field. Known cell
    # boundaries split each normal ray; no global point search or FE rebuild.
    spacing=24.4140625
    levels={}
    for p in phases:
        h=p['h']
        levels.setdefault(h,{})[(round(p['x']/h),round(p['y']/h))]=np.array([p['phi'+str(i)] for i in range(4)])
    assert min(levels)==spacing
    def sample(p):
        result=np.zeros(len(p));found=np.zeros(len(p),dtype=bool)
        for h in sorted(levels):
            for k in np.flatnonzero(~found):
                ij=np.floor(p[k]/h).astype(int);values=levels[h].get(tuple(ij))
                if values is not None:
                    a,b=p[k]/h-ij
                    result[k]=(1-a)*(1-b)*values[0]+a*(1-b)*values[1]+(1-a)*b*values[2]+a*b*values[3]
                    found[k]=True
            if np.all(found):break
        assert np.all(found),'Saved phase-cell coverage incomplete'
        return result
    def column(distance,order):
        center=origin+(s[-1]-distance)*t
        extent=halfwidth+2*spacing
        assert max(abs(sample(center+np.array([-extent,extent])[:,None]*normal)))==0
        cuts=[-extent,-halfwidth,halfwidth,extent]
        for d in (0,1):
            endpoints=center[d]+normal[d]*np.array([-extent,extent])
            boundaries=spacing*np.arange(np.floor(min(endpoints)/spacing),np.ceil(max(endpoints)/spacing)+1)
            cuts.extend(((boundaries-center[d])/normal[d]).tolist())
        cuts=np.unique(cuts);cuts=cuts[(cuts>=-extent)&(cuts<=extent)]
        a,w=leggauss(order)
        r=((cuts[1:]+cuts[:-1])[:,None]+(cuts[1:]-cuts[:-1])[:,None]*a)/2
        weights=(cuts[1:]-cuts[:-1])[:,None]*w/2
        p=center+r.ravel()[:,None]*normal
        phi=sample(p);hh=m*phi*(1+phi)/(1-phi)**2
        J=np.dot(weights.ravel(),hh)
        supported=np.dot(weights.ravel()*(abs(r.ravel())<=halfwidth),hh)
        return J,supported,float(sample(center[None,:])[0])
    distances=np.arange(27000.,29800.1,2.)
    columns=np.array([column(d,8) for d in distances])
    high=np.array([column(d,12) for d in distances])
    column_error=float(np.max(abs(columns[:,:2]/high[:,:2]-1)))
    assert column_error<1e-10,column_error
    represented_I=np.interp(s[-1]-distances,s,I)
    np.savetxt(OUT/'columns.csv',np.column_stack([distances,columns,represented_I,columns[:,0]/represented_I-1]),
               delimiter=',',comments='',header='xd,J_full,J_supported,phi_on_ridge,I_hat,J_over_I_minus_one')
    summary['columns']=dict(relative_8_vs_12_error=column_error,
        full_J_range_m=[float(min(columns[:,0])),float(max(columns[:,0]))],
        full_J_relative_peak_to_peak=float(np.ptp(columns[:,0])/np.mean(columns[:,0])),
        ridge_phi_range=[float(min(columns[:,2])),float(max(columns[:,2]))],
        max_abs_column_normalization_error=float(max(abs(columns[:,0]/represented_I-1))))
    # Reproduce the existing three-surface-point projection RHS, then compare
    # with the independently resolved full-profile cell/basis integral.
    # This identifies where a sampled column signal enters the saved Q1 Ih.
    rhs3=np.zeros(n);MI=np.zeros(n);surface_mass=np.zeros(n)
    a,w=leggauss(3);a=(a+1)/2;w=w/2
    ids=np.flatnonzero(mask)
    for j in range(min(ids)-1,max(ids)+1):
        length=s[j+1]-s[j]
        J=np.array([column(s[-1]-(s[j]+length*z),10)[0] for z in a])
        for end,N in ((0,1-a),(1,a)):
            rhs3[j+end]+=length*np.dot(w,N*J)
        MI[j]+=length*(I[j]/3+I[j+1]/6);MI[j+1]+=length*(I[j]/6+I[j+1]/3)
        surface_mass[j]+=length/2;surface_mass[j+1]+=length/2
    exact_rhs=data['FE_full_constant_I'][0]*constant_I
    observed=MI[mask]/surface_mass[mask];coarse=rhs3[mask]/surface_mass[mask]
    exact=exact_rhs[mask]/surface_mass[mask]
    summary['projection']=dict(reproduced_three_point_RHS_max_relative_error=float(max(abs(rhs3[mask]/MI[mask]-1))),
        exact_full_profile_rhs_relative_peak_to_peak=float(np.ptp(exact)/np.mean(exact)),
        saved_projected_rhs_relative_peak_to_peak=float(np.ptp(observed)/np.mean(observed)),
        three_point_vs_exact_max_relative_error=float(max(abs(rhs3[mask]/exact_rhs[mask]-1))))
    np.savetxt(OUT/'projection_rhs.csv',np.column_stack([xd[mask],observed,coarse,exact]),delimiter=',',comments='',
               header='xd,saved_M_I_per_mass,reproduced_three_point_J_per_mass,accurate_full_J_per_mass')
    if surface_subdivisions:
        convergence={}
        for subdivisions in surface_subdivisions:
            qa=np.concatenate([(k+a)/subdivisions for k in range(subdivisions)])
            qw=np.tile(w/subdivisions,subdivisions)
            rhs=np.zeros(n)
            for j in range(min(ids)-1,max(ids)+1):
                length=s[j+1]-s[j]
                J=np.array([column(s[-1]-(s[j]+length*z),10)[0] for z in qa])
                for end,N in ((0,1-qa),(1,qa)):
                    rhs[j+end]+=length*np.dot(qw,N*J)
            convergence[subdivisions]=dict(max_relative_rhs_error=float(max(abs(rhs[mask]/exact_rhs[mask]-1))),
                rhs_relative_peak_to_peak=float(np.ptp(rhs[mask]/surface_mass[mask])/np.mean(exact)))
            np.savetxt(OUT/f'projection_subdivisions_{subdivisions}.csv',
                np.column_stack([xd[mask],rhs[mask]/surface_mass[mask],exact]),delimiter=',',comments='',
                header='xd,composite_rhs_per_mass,accurate_rhs_per_mass')
            print(subdivisions,convergence[subdivisions],flush=True)
        summary['surface_subdivisions']=convergence
    # True normal profile vs two contrasting *saved-field* slices. No nodal
    # phase or normalization data are changed by this diagnostic.
    radii=np.linspace(-halfwidth-30,halfwidth+30,1501)
    selected_columns=np.array([np.argmin(columns[:,0]),np.argmax(columns[:,0])])
    slices=[sample(origin+(s[-1]-distances[k])*t+radii[:,None]*normal) for k in selected_columns]
    np.savetxt(OUT/'transverse_profiles.csv',np.column_stack([radii,np.interp(abs(radii),rtab,ptab,right=0.),*slices]),
               delimiter=',',comments='',header='r,intended_phi,FE_min_J_phi,FE_max_J_phi')
    summary['transverse_slice_xd_m']=distances[selected_columns].tolist()
    summary['seconds']=time.monotonic()-start
    (OUT/'moments.json').write_text(json.dumps(summary,indent=2)+'\n')
    inputs=[BASE/'profiles/fault_0.csv',BASE/'work_weak_0.csv',BASE/'stationary_profile.csv',
            BASE/'reconstructed_faults/reconstructed_faults-00000.vtu',
            *sorted(BASE.glob('phase_cells_rank*.csv')),*sorted(BASE.glob('initial_mesh_*.csv')),
            *sorted(BASE.glob('work_qp_0_rank*.csv'))]
    record=dict(reference=str(BASE),mechanical_solves=0,source=str(__file__),
                sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (OUT/'inputs.json').write_text(json.dumps(record,indent=2)+'\n')
    plots(summary)
    print(json.dumps(summary,indent=2))


def plots(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data={name:table(OUT/(name+'.csv')) for name in
          ('production_Q3','split_16','FE_constant_I','uniform_profile_constant_I_32')}
    fig,axes=plt.subplots(3,2,figsize=(12,9),sharex=True)
    for label in data:
        d=data[label];mask=(d['xd']>=27000)&(d['xd']<=29800)
        for row,key in enumerate(('mass','chi2','chi_work')):
            y=d[key];mean=np.mean(y[mask])
            axes[row,0].plot(d['xd'][mask]/1000,1e6*(y[mask]/mean-1),label=label,lw=1)
            axes[row,1].plot(d['xd'][mask]/1000,1e6*d['chord_'+key][mask]/mean,lw=1)
            axes[row,0].set_ylabel(key+' deviation [ppm]')
            axes[row,1].set_ylabel(key+' chord [ppm]')
    for ax in axes.flat:ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=8);axes[2,0].set_xlabel('Down-dip distance [km]');axes[2,1].set_xlabel('Down-dip distance [km]')
    fig.suptitle('Saved-profile reintegration; constant-I curves are offline attribution only')
    fig.tight_layout();fig.savefig(OUT/'moments.png',dpi=180);plt.close(fig)
    columns=table(OUT/'columns.csv');transverse=table(OUT/'transverse_profiles.csv');rhs=table(OUT/'projection_rhs.csv')
    fig,axes=plt.subplots(3,1,figsize=(11,10))
    axes[0].plot(columns['xd']/1000,columns['J_full'],label='Accurate full column of saved FE phase',lw=.9)
    axes[0].plot(columns['xd']/1000,columns['I_hat'],label='Saved Q1 I_h',lw=1.2)
    axes[0].set_ylabel('I_h or J [m]');axes[0].set_xlabel('Down-dip distance [km]')
    for key,label in [('saved_M_I_per_mass','Saved projection'),('reproduced_three_point_J_per_mass','Three-point RHS'),
                      ('accurate_full_J_per_mass','Accurate cell/basis-split RHS')]:
        axes[1].plot(rhs['xd']/1000,rhs[key],label=label,lw=1)
    axes[1].set_ylabel('Projection RHS / row mass [m]');axes[1].set_xlabel('Down-dip distance [km]')
    for key,label in [('intended_phi','Intended uniform normal profile'),
                      ('FE_min_J_phi',f'Saved FE at {summary["transverse_slice_xd_m"][0]/1000:.3f} km'),
                      ('FE_max_J_phi',f'Saved FE at {summary["transverse_slice_xd_m"][1]/1000:.3f} km')]:
        axes[2].plot(transverse['r'],transverse[key],label=label,lw=1)
    axes[2].set_xlim(-40,40);axes[2].set_ylim(.43,.605)
    axes[2].set_ylabel('Phase field (ridge close-up)');axes[2].set_xlabel('Normal distance [m]')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(OUT/'profile-and-projection.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--surface-subdivisions',type=int,nargs='+',default=[])
    parser.add_argument('--output',type=str)
    args=parser.parse_args()
    if args.output:
        from pathlib import Path
        OUT=Path(args.output)
    run(args.surface_subdivisions)
