"""Saved-state source-shape/mesh audit; current and history stresses stay distinct."""
import argparse
import csv
import json
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader
from vtkmodules.vtkFiltersCore import vtkAppendFilter, vtkProbeFilter
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkStaticCellLocator
from vtkmodules.util.numpy_support import vtk_to_numpy,numpy_to_vtk
from analyze_uniform_sliding import read,cat,write,records

HERE=Path(__file__).resolve().parent
WORK=HERE/'work-replay-50-local4'
BASE=HERE/'top-source-paired-50-local4'
SHIFT=HERE/'deep-mesh-shift48-uniform-local4'
OUT=HERE/'deep-mesh-feature-analysis'
SN=np.sqrt(3)/2;XT=50000*(1+.5/SN)


def coordinates(x,y):
    return (XT-x)*.5+(100000-y)*SN,(XT-x)*SN-(100000-y)*.5


def grid(root,k):
    app=vtkAppendFilter()
    for p in sorted(root.glob(f'bulk_{k}_*.vtu')):
        r=vtkXMLUnstructuredGridReader();r.SetFileName(str(p));r.Update();g=r.GetOutput()
        # Do interpolation in double; retain the original Float32 nodal data
        # uncertainty, but do not add another Float32 rounding at every probe.
        values=numpy_to_vtk(vtk_to_numpy(g.GetPointData().GetArray('component_9')).astype(float),deep=True)
        values.SetName('component_9');g.GetPointData().RemoveArray('component_9');g.GetPointData().AddArray(values)
        app.AddInputData(g)
    app.Update();return app.GetOutput()


def sample(g,xy,field='component_9'):
    pts=vtkPoints();pts.SetData(numpy_to_vtk(np.column_stack([xy,np.zeros(len(xy))]),deep=True))
    inp=vtkPolyData();inp.SetPoints(pts)
    probe=vtkProbeFilter();probe.SetCellLocatorPrototype(vtkStaticCellLocator())
    probe.SetInputData(inp);probe.SetSourceData(g);probe.Update()
    out=probe.GetOutput().GetPointData()
    assert np.all(vtk_to_numpy(out.GetArray('vtkValidPointMask'))==1)
    return vtk_to_numpy(out.GetArray(field)).copy()


def maxima(d,mask,field,offset=0):
    ids=np.flatnonzero(mask);j=ids[np.argmax(abs(d[field][ids]-offset))]
    return dict(value=float(d[field][j]),xd=float(d['xd'][j]),r=float(d['r'][j]),
                x=float(d['x'][j]),y=float(d['y'][j]))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--with-shift',action='store_true');args=ap.parse_args()
    OUT.mkdir(exist_ok=True)
    cases=[('uniform44',BASE),('work',WORK)]+([('uniform48',SHIFT)] if args.with_shift else [])
    summary={};peaks=[];bins=[];columns={};shape_checks=[]
    reference=read(BASE/'uniform_reference_profile.csv');radius=reference['r'][-1]
    phi=reference['phi'][0];m=reference['h'][0]*(1-phi)**2/(phi*(1+phi))
    for label,root in cases:
        mesh=cat(root.glob('initial_mesh_*.csv'),('cell',));mesh['xd'],mesh['r']=coordinates(mesh['x'],mesh['y'])
        f=read(root/'fault_0.csv');order=np.argsort(f['xd']);xd=f['xd'][order];ih=f['Ih'][order]
        write(OUT/f'{label}_fault.csv',dict(xd=xd,Ih=ih,element_center=(np.r_[xd[:-1],xd[-1]]+xd)/2,
                                          following_spacing=np.r_[np.diff(xd),np.nan]))
        for k in ([0,10] if label=='work' else [0,1,2]):
            pattern=f'work_qp_{k}_rank*.csv' if label=='work' else f'uniform_bulk_{k}_rank*.csv'
            d=cat(root.glob(pattern),('cell',))
            for name,lo,hi in [('small42',42000,43200),('old_edge',43500,44500),('new_edge',47500,48500),('control',45000,47000)]:
                mask=(d['xd']>=lo)&(d['xd']<=hi)
                if not np.any(mask):continue
                for field in ('p','tau_xx','tau_xy','sigma_n'):
                    peaks.append(dict(case=label,step=k,region=name,field=field,
                        **maxima(d,mask,field,50e6 if field=='sigma_n' else 0)))
            for lo in range(35000,52000,100):
                mask=(d['xd']>=lo)&(d['xd']<lo+100)
                if not np.any(mask):continue
                entry=dict(case=label,step=k,xd=lo+50)
                for field in ('p','tau_xx','tau_xy','sigma_n'):
                    entry[field+'_min']=float(min(d[field][mask]));entry[field+'_max']=float(max(d[field][mask]))
                bins.append(entry)
        # Read published fields only as old FE histories, never call them the
        # current constitutive tensor. Pressure remains the accepted unknown.
        g=grid(root,10 if label=='work' else 0)
        points=vtk_to_numpy(g.GetPoints().GetData());s,r=coordinates(points[:,0],points[:,1])
        mask=(s>=35000)&(s<=52000)&(abs(r)<1100)
        nodal=dict(x=points[mask,0],y=points[mask,1],xd=s[mask],r=r[mask])
        for field in ('delta_pressure','tau_xx','tau_yy','tau_xy'):
            nodal[field]=vtk_to_numpy(g.GetPointData().GetArray(field))[mask]
        write(OUT/f'{label}_published_nodes.csv',nodal)
        if label=='work':
            fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True,sharey=True)
            for ax,field,title in zip(axes,('delta_pressure','tau_xx','tau_xy'),
                    ('accepted pressure','published OLD FE tau_xx','published OLD FE tau_xy')):
                value=nodal[field];bound=float(max(abs(value)))
                shown=ax.scatter(nodal['xd']/1000,nodal['r'],c=value,s=7,
                    cmap='coolwarm',norm=SymLogNorm(linthresh=1000,vmin=-bound,vmax=bound))
                fig.colorbar(shown,ax=ax,label='Pa (symmetric log colour)')
                ax.set(title=title,ylabel='Normal r (m)',xlim=(35,48),ylim=(-1000,1000))
                for mark in (40,44):ax.axvline(mark,color='k',ls=':',lw=.7)
                ax.plot(42.93474467,-526.1785484,'kx',ms=7)
            axes[-1].set_xlabel('Down dip (km)');fig.suptitle('Accepted step 10, 29.24190894 yr; X = fixed uniform-feature location')
            fig.tight_layout();fig.savefig(OUT/'work_step10_published_fields.png',dpi=160);plt.close(fig)
        # Saved VTU Q1 corner values reproduce FE phi with Float32 export
        # precision. Verify against double-precision production QP samples.
        d=cat(root.glob('work_qp_0_rank*.csv' if label=='work' else 'uniform_bulk_0_rank*.csv'),('cell',))
        test=(d['xd']>35000)&(d['xd']<43000)
        error=float(max(abs(sample(g,np.column_stack([d['x'][test],d['y'][test]]))-d['phi'][test])))
        assert error<4e-8
        stations=np.arange(35000,52000.1,50.);normal=np.arange(-1100,1100.1,2.)
        ss,rr=np.meshgrid(stations,normal,indexing='ij')
        xy=np.column_stack([(XT-.5*ss-SN*rr).ravel(),(100000-SN*ss+.5*rr).ravel()])
        ph=sample(g,xy).reshape(ss.shape);ph=np.maximum(ph,0.)
        Ih=np.interp(stations,xd,ih);local=m*ph*(1+ph)/(1-ph)**2/Ih[:,None]
        source=np.where(abs(rr)<=radius,local,0.)
        # Same-location counterfactual Q1 interpolation on a complete 48.8-m
        # grid, with the SAME actual I_h denominator. This isolates source
        # shape distortion, not a replacement normalization or production fix.
        h=48.828125;x=xy[:,0];y=xy[:,1];i=np.floor(x/h);j=np.floor(y/h)
        a=x/h-i;b=y/h-j;fine_phi=np.zeros(len(x))
        for u in (0,1):
            for v in (0,1):
                _,r=coordinates((i+u)*h,(j+v)*h)
                fine_phi+=(a if u else 1-a)*(b if v else 1-b)*np.interp(abs(r),reference['r'],reference['phi'],right=0.)
        fine_phi=fine_phi.reshape(ss.shape)
        fine=m*fine_phi*(1+fine_phi)/(1-fine_phi)**2/Ih[:,None]
        fine=np.where(abs(rr)<=radius,fine,0.)
        for s0 in (41000,42900,42950,43000,43900,44000,47900,48000):
            j=int(np.argmin(abs(stations-s0)));q=int(np.argmax(abs(source[j]-fine[j])))
            shape_checks.append(dict(case=label,xd=s0,r=normal[q],actual_chi=source[j,q],fine_grid_chi=fine[j,q],
                difference=source[j,q]-fine[j,q],relative_difference=source[j,q]/fine[j,q]-1,
                absolute_shape_integral=np.trapezoid(abs(source[j]-fine[j]),normal)))
        write(OUT/f'{label}_transverse.csv',dict(xd=ss.ravel(),r=rr.ravel(),phi=ph.ravel(),
            Ih=np.repeat(Ih,len(normal)),chi_full=local.ravel(),chi_supported=source.ravel()))
        centers=np.interp(stations,xd,np.r_[np.diff(xd),np.diff(xd)[-1]])
        col=dict(xd=stations,Ih=Ih,chi_center=local[:,len(normal)//2],
                 integral_full=np.trapezoid(local,normal,axis=1),integral_supported=np.trapezoid(source,normal,axis=1),
                 chi_second_moment=np.trapezoid(source*rr**2,normal,axis=1),fault_spacing=centers)
        columns[label]=(col,normal,source)
        write(OUT/f'{label}_columns.csv',col)
        summary[label]=dict(phi_QP_check_max_error=error,profile_radius=float(radius),cell_count=len(mesh['x']))
        fig,axes=plt.subplots(4,1,figsize=(11,11),sharex=True)
        view=(mesh['xd']>35000)&(mesh['xd']<52000)&(abs(mesh['r'])<1000)
        shown=axes[0].scatter(mesh['xd'][view]/1000,mesh['r'][view],c=mesh['h'][view],s=5,cmap='viridis')
        fig.colorbar(shown,ax=axes[0],label='bulk h (m)')
        axes[0].plot(42.93474467,-526.1785484,'rx',ms=7)
        axes[0].set_ylabel('Cell centers: r (m)')
        axes[1].plot(stations/1000,Ih);axes[1].set_ylabel('I_h (m)')
        second=axes[1].twinx();second.step(xd[:-1]/1000,np.diff(xd),where='post',color='tab:orange',alpha=.7)
        second.set_ylabel('fault spacing (m)',color='tab:orange');second.set_ylim(40,110)
        axes[2].plot(stations/1000,col['integral_supported']);axes[2].set_ylabel('Supported integral chi')
        subset=[b for b in bins if b['case']==label and b['step']==(10 if label=='work' else 2)]
        axes[3].plot([b['xd']/1000 for b in subset],[max(abs(b['p_min']),abs(b['p_max']))/1e3 for b in subset])
        axes[3].set_ylabel('Raw current |p| max (kPa)');axes[3].set_xlabel('Down dip (km)')
        for ax in axes:
            for mark in (40,44,48):ax.axvline(mark,color='k',ls=':',lw=.7)
            ax.set_xlim(35,52)
        fig.suptitle(label+'; work raw QPs stop at 43 km, published fields exported separately')
        fig.tight_layout();fig.savefig(OUT/f'{label}_overlay.png',dpi=150);plt.close(fig)
        fig,axes=plt.subplots(2,1,figsize=(9,8))
        for s0 in (41000,42800,42950,43500,43900,44100,46000,47900,48100):
            j=int(np.argmin(abs(stations-s0)));axes[0].plot(normal,source[j],label=f'{s0/1000:g} km')
            control=source[np.argmin(abs(stations-41000))]
            axes[1].plot(normal,source[j]-control,label=f'{s0/1000:g} km')
        axes[0].set(ylabel='Actual supported chi (1/m)');axes[1].set(xlabel='Normal r (m)',ylabel='chi minus 41-km column (1/m)')
        for ax in axes:ax.legend(ncol=3,fontsize=8)
        fig.tight_layout();fig.savefig(OUT/f'{label}_transverse.png',dpi=150);plt.close(fig)
        fig,ax=plt.subplots(figsize=(9,5))
        for s0 in (41000,42900,42950,43000):
            j=int(np.argmin(abs(stations-s0)));ax.plot(normal,source[j]-fine[j],label=f'{s0/1000:g} km')
        ax.set(xlim=(-850,-400),xlabel='Normal r (m)',ylabel='Actual minus fine-grid chi (1/m)',
               title=label+': source wing defect, same I_h denominator')
        ax.legend();fig.tight_layout();fig.savefig(OUT/f'{label}_wing_defect.png',dpi=150);plt.close(fig)
    records(OUT/'current_stress_peaks.csv',peaks);records(OUT/'current_stress_bins.csv',bins)
    records(OUT/'source_shape_checks.csv',shape_checks)
    if args.with_shift:
        clock=read(SHIFT/'accepted_steps.csv');np.testing.assert_array_equal(clock['step'],[0,1,2])
        old=read(BASE/'accepted_steps.csv');np.testing.assert_allclose(clock['time'],old['time'],rtol=1e-13,atol=1e-8)
        log=(SHIFT/'run.log').read_text();assert 'BP3 REPLAY COMPLETE' in log and 'Timestep 3:' not in log
        checks=re.findall(r'Fault linear solve: iterations=\d+, estimated=[^,]+, fresh=([^,]+), target=([^,]+)',log)
        # Uniform prescribed V uses the ordinary full Stokes solve, rather
        # than assuming a free-interface Krylov trace exists.
        for a,b in checks:assert float(a)<=float(b)
        for k in range(3):
            section=log.split(f'*** Timestep {k}:')[1].split('*** Timestep ')[0]
            residual=re.findall(r'Relative nonlinear residuals .*?: ([^,\n]+), ([^\n]+)',section)[-1]
            assert max(map(float,residual))<1e-8
            f=read(SHIFT/f'fault_{k}.csv');b=read(BASE/f'fault_{k}.csv')
            np.testing.assert_array_equal(f['V'],np.full(len(f['V']),1e-9))
            for name in ('x','y','Theta','C','slip'):np.testing.assert_allclose(f[name],b[name],rtol=1e-13,atol=1e-10)
        summary['guard_and_solve']=dict(accepted_steps=3,fresh_linear_checks=len(checks),extra_step=False)
        old_mesh={row['cell']:row for p in BASE.glob('initial_mesh_*.csv') for row in csv.DictReader(p.open())}
        new_mesh={row['cell']:row for p in SHIFT.glob('initial_mesh_*.csv') for row in csv.DictReader(p.open())}
        for side in ('top','bottom'):
            select=lambda d:{k:v for k,v in d.items() if (float(v['y'])>97000 if side=='top' else float(v['y'])<3000)}
            assert select(old_mesh)==select(new_mesh),'Boundary completion mesh changed'
        target=set((SHIFT/'target_cells.txt').read_text().split())
        extra=set(new_mesh)-target
        for cell in extra:
            root,path=cell.split(':');root=root.split('_')[0]
            assert any(f'{root}_{n}:{path[:n]}' in target for n in range(len(path)))
        changed=[new_mesh[k] for k in set(new_mesh)-set(old_mesh)]
        s,r=coordinates(np.array([float(v['x']) for v in changed]),np.array([float(v['y']) for v in changed]))
        summary['mesh_check']=dict(extra_grading_leaves=len(extra),changed_xd_range=[float(min(s)),float(max(s))],
            changed_r_range=[float(min(r)),float(max(r))],boundary_cells_identical=True,
            endpoint_completion_identical=(SHIFT/'completion.txt').read_bytes()==(BASE/'completion.txt').read_bytes())
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
