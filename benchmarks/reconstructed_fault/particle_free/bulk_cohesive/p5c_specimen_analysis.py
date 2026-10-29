#!/usr/bin/env python3
"""P5c physical-coordinate analysis; native cell averages, no history interpolation."""
from pathlib import Path
import argparse,json
import numpy as np
from scipy.ndimage import label
from numpy.polynomial.legendre import leggauss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from p5c_analysis import read,save

def grid(a,key):
    x=np.unique(a['x']);y=np.unique(a['y']);v=np.empty((len(y),len(x)))
    v[np.searchsorted(y,a['y']),np.searchsorted(x,a['x'])]=a[key]
    return x,y,v

def seed_reference(run):
    a=read(run/'seed-prescription.csv')
    return lambda x,y:np.interp(np.hypot(np.maximum(0.,x-1.),y-.5),a['radius_m'],a['phi'],right=0.)

def initialization(run):
    a=read(run/'phase-0000.csv');x,y,p=grid(a,'phi');dx=np.diff(x);dy=np.diff(y);f=seed_reference(run);z,w=leggauss(8);z=(z+1)/2;w=w/2
    err=norm=0.;graderr=gradnorm=0.
    for u,wu in zip(z,w):
        for v,wv in zip(z,w):
            X=x[:-1][None,:]+u*dx[None,:];Y=y[:-1][:,None]+v*dy[:,None]
            q=(1-u)*(1-v)*p[:-1,:-1]+u*(1-v)*p[:-1,1:]+(1-u)*v*p[1:,:-1]+u*v*p[1:,1:]
            true=f(X,Y);weights=wu*wv*dy[:,None]*dx[None,:];err+=np.sum(weights*(q-true)**2);norm+=np.sum(weights*true**2)
    first=read(run/'phase-0001.csv');s=read(run/'bulk-accepted.csv')[0]
    return {'L2_relative_representation':float(np.sqrt(err/norm)),'L2_absolute_m':float(np.sqrt(err)),'min':float(p.min()),'max':float(p.max()),'nodal_continuum_defect':float(np.max(abs(p-f(x[None,:],y[:,None])))),'unloaded_first_step_max_change':float(np.max(abs(first['phi']-a['phi']))),'unloaded_KKT':float(s['phase_KKT']),'seed_support_radius_m':float(read(run/'seed-prescription.csv')['radius_m'][-1])}

def history(run,out,name):
    files=sorted(run.glob('phase-[0-9][0-9][0-9][0-9].csv'));ledger=read(run/'bulk-accepted.csv');times={0:0.,**{int(r['step']):float(r['time']) for r in ledger}}
    seed=read(run/'phase-0000.csv') if (run/'phase-0000.csv').exists() else None
    if seed is None:raise ValueError('phase-0000 required for physical growth reference')
    x,y,initial=grid(seed,'phi');cont=seed_reference(run);oldreact=None;trap=endwork=0.;absolute_trap=0.;previous_trap=0.;rows=[];onset=None;last=0.;onsetpoints=[];last_phi=None
    dx=x[1]-x[0];dy=y[1]-y[0];mass=np.ones_like(initial)*dx*dy;mass[[0,-1],:]*=.5;mass[:,[0,-1]]*=.5
    X,Y=np.meshgrid(x,y);initial_support=cont(X,Y)>0.
    for p in files:
        i=int(p.stem[-4:]);t=times[i];a=read(p);xx,yy,phi=grid(a,'phi');d=phi-initial
        active=d>1e-8
        if onset is None and active.any():
            onset=[last,t];onsetpoints=np.column_stack([X[active],Y[active],d[active]]).tolist()
        r={'step':i,'time_s':t,'min_phi':float(phi.min()),'max_phi':float(phi.max()),'min_increment':float(d.min()),'max_phase_change_per_step':float(np.max(phi-last_phi)) if last_phi is not None else 0.,'peak_increment':float(d.max()),'integral_growth_m2':float(np.sum(mass*d)),'cap_growth_m2':float(np.sum(mass*d*(X>=1.-.046875))),'remote_growth_m2':float(np.sum(mass*d*(~initial_support))),'peak_initially_intact_growth':float(np.max(d[~initial_support])),'growth_L2_m':float(np.sqrt(np.sum(mass*d*d))),'imposed_relative_displacement_m':.018*max(0.,min(t,.14)-.05)}
        for level in [.01,.05,.1,.2,.4,.6]:
            mask=phi>=level;lab,n=label(mask);seedlabels=np.unique(lab[:,0]);seedlabels=seedlabels[seedlabels>0];connected=np.isin(lab,seedlabels);outside=mask&~connected
            r[f'connected_x_phi{level}']=float(X[connected].max()) if connected.any() else None
            r[f'disconnected_area_phi{level}_m2']=float(np.sum(mass*outside))
            j=int(np.argmin(abs(y-.5)));line=phi[j];keep=np.where(line>=level)[0]
            front=np.nan
            if len(keep):
                k=keep[-1];front=x[k] if k==len(x)-1 else x[k]+(level-line[k])*(x[k+1]-x[k])/(line[k+1]-line[k])
            r[f'centerline_x_phi{level}']=float(front)
        reaction=read(run/f'boundary-reactions-{i:04}.csv');order=np.lexsort((reaction['component'],reaction['y'],reaction['x']));reaction=reaction[order]
        if oldreact is not None:
            assert np.array_equal(reaction[['component','x','y']],oldreact[['component','x','y']])
            du=reaction['increment'];trap+=float(np.dot(.5*(reaction['force']+oldreact['force']),du));endwork+=float(np.dot(reaction['force'],du))
        oldreact=reaction;r['top_horizontal_reaction_N_per_m']=float(np.sum(reaction['force'][(reaction['component']==0)&(abs(reaction['y']-1.)<1e-12)]));r['endpoint_reaction_work_J_m']=endwork;r['trapezoidal_reaction_work_J_m']=trap
        if i:
            l=ledger[ledger['step']==i][0]
            for key in ledger.dtype.names:
                if key not in ['step','time']:r[key]=float(l[key])
            earlier=ledger[ledger['step']<=i];r['cumulative_dissipation_J_m']=float(np.sum(earlier['Db']+earlier['Dc']+earlier['Df']))
            r['storage_change_J_m']=float(l['bulk']+l['cohesive']+l['fracture']-ledger[0]['fracture'])
            r['endpoint_physical_remainder_J_m']=endwork-r['storage_change_J_m']-r['cumulative_dissipation_J_m']
            r['trapezoid_physical_remainder_J_m']=trap-r['storage_change_J_m']-r['cumulative_dissipation_J_m']
            absolute_trap+=abs(r['trapezoid_physical_remainder_J_m']-previous_trap);previous_trap=r['trapezoid_physical_remainder_J_m'];r['absolute_step_trapezoid_remainders_J_m']=absolute_trap
            r['cumulative_signed_BE_remainder_J_m']=float(np.sum(earlier['stress_loss']+earlier['traction_loss']+earlier['phase_remainder']+earlier['fracture_loss']))
            r['cumulative_mechanical_residual_work_J_m']=float(np.sum(earlier['mechanical_work']));r['cumulative_phase_residual_work_J_m']=float(np.sum(earlier['phase_work']))
        rows.append(r);last=t;last_phi=phi.copy()
    # Keep native ledger separate; variable row keys need explicit CSV missing values.
    import csv
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (out/f'{name}-history.csv').open('w') as f:
        c=csv.DictWriter(f,keys);c.writeheader();c.writerows(rows)
    return {'onset_bracket_s':onset,'first_growing_nodes':onsetpoints,'initial_fronts':{k:v for k,v in rows[0].items() if '_x_phi' in k},'final':rows[-1],'initialization':initialization(run),'accepted_steps':len(ledger)},rows

def figures(runs,out,common_steps):
    initial={name:grid(read(run/'phase-0000.csv'),'phi')[2] for name,run in runs.items()};names=list(runs);cache={}
    for name,run in runs.items():
        for step in common_steps:
            a=read(run/f'phase-{step:04}.csv');x,y,p=grid(a,'phi');m=read(run/f'means-{step:04}.csv');cx,cy,tau=grid(m,'tauxy');Y=grid(m,'Y')[2];wx=grid(m,'Wxx')[2];wy=grid(m,'Wxy')[2];speed=np.hypot(wx,wy)
            # Tensor-component direction alpha=atan2(Wxx,Wxy); alpha=0 is +xy flow.
            angle=np.ma.masked_where(speed<=1e-12,np.degrees(np.arctan2(wx,wy)))
            cache[name,step]=(x,y,p,cx,cy,tau,Y,angle,p-initial[name])
    # Compact exact plotting arrays; units and averaging are in plot-definitions.json.
    arrays={'accepted_steps':np.array(common_steps)}
    for name in names:
        arrays[name+'_node_x']=cache[name,common_steps[0]][0];arrays[name+'_node_y']=cache[name,common_steps[0]][1]
        arrays[name+'_cell_x']=cache[name,common_steps[0]][3];arrays[name+'_cell_y']=cache[name,common_steps[0]][4]
        for key,index in [('phi',2),('tauxy_Pa',5),('Y_Pa',6),('increment',8)]:arrays[name+'_'+key]=np.stack([cache[name,i][index] for i in common_steps])
        arrays[name+'_flow_alpha_deg']=np.stack([cache[name,i][7].filled(np.nan) for i in common_steps])
    np.savez_compressed(out/'plotted-fields.npz',**arrays)
    import csv
    with (out/'physical-linecuts.csv').open('w') as f:
        columns=['mode','step','time_s','x_m','kernel_y_m','phi','increment','tauxx_Pa','tauyy_Pa','tauxy_Pa','Y_Pa','mean_speed_per_s','Wxx_per_s','Wxy_per_s','Axx','Axy','Kxx','Kxy','L'];writer=csv.writer(f);writer.writerow(columns)
        for name,run in runs.items():
            m0=read(run/'means-0000.csv');times={0:0.,**{int(r['step']):float(r['time']) for r in read(run/'bulk-accepted.csv')}}
            for step in common_steps:
                m=read(run/f'means-{step:04}.csv');assert np.array_equal(m['cell'],m0['cell'])
                for yy in [.5-.046875,.5,.5+.046875]:
                    for xx in np.unique(m['x']):
                        keep=(abs(m['x']-xx)<1e-12)&(abs(m['y']-yy)<=.00390625+1e-12);weight=m['area'][keep];weight=weight/weight.sum()
                        avg=lambda key:float(weight@m[key][keep])
                        writer.writerow([name,step,times[step],xx,yy,avg('phi'),float(weight@(m['phi'][keep]-m0['phi'][keep])),*[avg(k) for k in ['tauxx','tauyy','tauxy','Y','abs_rate','Wxx','Wxy','Axx','Axy','Kxx','Kxy','L']]])
    dmax=max(float(c[-1].max()) for c in cache.values()) or 1e-8;tmax=max(float(np.max(abs(c[5]))) for c in cache.values()) or 1.;ymax=max(float(c[6].max()) for c in cache.values()) or 8.;amax=max((float(np.max(abs(c[7].compressed()))) if c[7].count() else 0.) for c in cache.values()) or 1.
    def draw(step,zoom=False):
        fig,axes=plt.subplots(len(names),5,figsize=(15,5.4),constrained_layout=True,squeeze=False)
        for i,name in enumerate(names):
            x,y,p,cx,cy,tau,Y,angle,d=cache[name,step];items=[(x,y,p,'phase',0,.65,'viridis'),(x,y,d,'phase increment',0,dmax,'magma'),(cx,cy,tau,'physical τxy [Pa]',-tmax,tmax,'RdBu_r'),(cx,cy,Y,'driving Y [Pa]',0,ymax,'inferno'),(cx,cy,angle,'flow tensor α [deg]',-amax,amax,'twilight')]
            for ax,(xx,yy,z,title,vmin,vmax,cmap) in zip(axes[i],items):
                im=ax.pcolormesh(xx,yy,z,shading='nearest',vmin=vmin,vmax=vmax,cmap=cmap,rasterized=True)
                ax.contour(x,y,p,levels=[.01,.05,.1,.2,.4,.6],colors='white',linewidths=.45);ax.contour(x,y,initial[name],levels=[.1,.4],colors='black',linewidths=.4,linestyles='dashed');ax.plot(1,.5,'r+',ms=7);ax.set_aspect('equal');ax.set_xlim(0,1.5);ax.set_ylim(0,1);ax.set_title(f'{name}: {title}',fontsize=9);ax.set_xlabel('x [m]');ax.set_ylabel('y [m]')
                if zoom:ax.set_xlim(.85,1.2);ax.set_ylim(.32,.68)
                fig.colorbar(im,ax=ax,shrink=.7)
        time_s=0. if step==0 else float(read(runs[names[0]]/'bulk-accepted.csv')['time'][read(runs[names[0]]/'bulk-accepted.csv')['step']==step][0])
        fig.suptitle(f'P5c t={time_s:.5f} s, Δt=0.00125 s; white=current contours, black dashed=initial, +=seed endpoint\nFlow tensor angle is not phase-ridge orientation',fontsize=10)
        return fig
    for step in sorted(set([common_steps[-1],*[i for i in common_steps if i==112]])):
        for zoom in [False,True]:
            fig=draw(step,zoom);fig.savefig(out/f'fields-{step:04}{"-tip" if zoom else ""}.png',dpi=150);plt.close(fig)
    # Render PNG frames to Pillow directly: one compact physical-coordinate GIF.
    from PIL import Image
    import io
    images=[]
    for step in common_steps:
        fig=draw(step);buf=io.BytesIO();fig.savefig(buf,format='png',dpi=85);plt.close(fig);buf.seek(0);images.append(Image.open(buf).convert('P',palette=Image.Palette.ADAPTIVE))
    images[0].save(out/'comparison.gif',save_all=True,append_images=images[1:],duration=300,loop=0)
    save(out/'plot-definitions.json',{'fields':'Gauss8 physical cell averages for stress, Y and W; Q1 nodal phase, no constitutive-state interpolation','alpha':'atan2(Wxx,Wxy) degrees; signed tensor component-space angle, 0 = positive xy. Plot only hypot(mean Wxx,mean Wxy)>1e-12/s. Not a crack tangent or finite-rotation qualification.','contours':[.01,.05,.1,.2,.4,.6],'linecut_kernel':'Physical rectangle dx=.015625 m by 2dy=.015625 m, centered on listed x and y=.5 or .5±ell. Average actual physical fields, never recompute constitutive quantities from averaged histories.', 'growth_cutoff':1e-8,'connected':'4-neighbor nodal superlevel components meeting left boundary; centerline front linearly interpolated on Q1 edge','scales':{'increment_max':dmax,'absolute_tau_limit_Pa':tmax,'Y_max_Pa':ymax,'flow_alpha_limit_deg':amax}})

def main(work,out):
    runs={mode:work/'runs'/f'seed-{mode}' for mode in ['fixed','radial']};summary={};hist={}
    for name,run in runs.items():summary[name],hist[name]=history(run,out,name)
    common=sorted(set.intersection(*[set(int(p.stem[-4:]) for p in run.glob('phase-[0-9][0-9][0-9][0-9].csv')) for run in runs.values()]))
    steps=sorted(set([0,common[-1],*[k for k in common if k%8==0 or k>=104 and k%2==0]]));figures(runs,out,steps)
    save(out/'specimen-summary.json',summary)
    fig,axes=plt.subplots(1,3,figsize=(12,3.2),constrained_layout=True)
    for name,rows in hist.items():
        a=rows[1:];t=[r['time_s'] for r in a]
        axes[0].plot(t,[r['integral_growth_m2'] for r in a],label=name);axes[1].plot([r['imposed_relative_displacement_m']*1000 for r in a],[r['top_horizontal_reaction_N_per_m'] for r in a],label=name);axes[2].plot(t,[r['trapezoid_physical_remainder_J_m'] for r in a],label=name)
    for ax,title in zip(axes,['integrated phase growth [m²]','top horizontal reaction [N/m]' ,'trapezoid physical remainder [J/m]']):ax.set_xlabel('time [s]');ax.set_ylabel(title);ax.legend()
    axes[1].set_xlabel('relative displacement [mm]')
    fig.savefig(out/'response-energy.png',dpi=160);plt.close(fig)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('work',type=Path);p.add_argument('out',type=Path);a=p.parse_args();main(a.work,a.out)
