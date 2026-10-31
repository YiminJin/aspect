#!/usr/bin/env python3
"""Read accepted native CSVs; no evolution, field conversion or mechanical solve."""
from pathlib import Path
import argparse,csv,json,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-cohesive-native-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image

ROOT=Path(__file__).resolve().parents[4]
WORK=ROOT/'build-particle-free/cohesive-rsf-native/runs'
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/cohesive_rsf_native'
REF=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_propagation'
OUTPUT_DIRECTORY=None
G,GC,DC,V0,RAD=1e6,1.,.001,5e-6,25836.021365527624


def table(path,rows):
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def load(labels):
    records={}
    superseded=[]
    for label in labels:
        directory=OUTPUT_DIRECTORY or WORK/label/'output'
        for row in csv.DictReader((directory/'cohesive-index.csv').open()):
            # Index paths describe the producer's original directory. Always
            # read the selected output, even if an old copy still exists.
            stem=directory/Path(row['stem']).name;time=float(row['time_s'])
            d={r['quantity']:float(r['value']) for r in csv.DictReader(Path(str(stem)+'-diagnostics.csv').open())}
            surface=np.genfromtxt(str(stem)+'-surface.csv',delimiter=',',names=True,dtype=None,encoding='utf8')
            geometry=np.genfromtxt(str(stem)+'-geometry.csv',delimiter=',',names=True,dtype=None,encoding='utf8')
            key=round(time,12)
            if key in records:
                previous=records[key]
                superseded.append(dict(time_s=key,previous_run=previous['label'],previous_stem=str(previous['stem']),selected_run=label,selected_stem=str(stem),previous_dt_s=previous['d'].get('dt_s'),selected_dt_s=d.get('dt_s')))
            records[key]=dict(label=label,stem=stem,time=time,d=d,s=surface,g=np.atleast_1d(geometry))
    load.superseded=superseded
    return [records[t] for t in sorted(records)]


def slope(piece,x):
    z=(x-piece['x_m'])/(piece['end_m']-piece['x_m'])
    if piece['hermite']:
        return piece['a']+piece['c1']*z+piece['c2']*z*z+piece['c3']*z*z*z
    return piece['a']+(piece['b']-piece['a'])*(3*z*z-2*z*z*z)


def arclength(record):
    g=record['g'];s=record['s'];X,W=np.polynomial.legendre.leggauss(20);arcs=np.zeros(len(s));offset=0.;cell=0
    for piece in g:
        nodes=np.linspace(piece['x_m'],piece['end_m'],int(piece['cells'])+1)
        def integral(a,b):
            x=.5*(a+b)+.5*(b-a)*X
            return .5*(b-a)*np.dot(W,np.sqrt(1+slope(piece,x)**2))
        for a,b in zip(nodes[:-1],nodes[1:]):
            for q in range(3):arcs[3*cell+q]=offset+integral(a,s['x_m'][3*cell+q])
            offset+=integral(a,b);cell+=1
    return arcs


def front(s,arc,threshold):
    mask=s['damage']>=threshold
    components=[];start=None
    for q,active in enumerate(np.r_[mask,False]):
        if active and start is None:start=q
        if not active and start is not None:
            end=q-1
            if np.any((s['x_m'][start:q]>=3)&(s['x_m'][start:q]<=6)):
                components.append((start,end))
            start=None
    if not components:return dict(arc_m=np.nan,x_m=np.nan,y_m=np.nan,tangent_deg=np.nan,lower_arc_m=np.nan,upper_arc_m=np.nan)
    _,end=max(components,key=lambda pair:pair[1]);next_q=min(end+1,len(s)-1)
    return dict(arc_m=.5*(arc[end]+arc[next_q]),x_m=.5*(s['x_m'][end]+s['x_m'][next_q]),
        y_m=.5*(s['y_m'][end]+s['y_m'][next_q]),tangent_deg=.5*(s['tangent_deg'][end]+s['tangent_deg'][next_q]),
        lower_arc_m=arc[end],upper_arc_m=arc[next_q])


def summarize(records):
    histories=[];profiles=[];tracked=[];maturity={};previous_energy=None;initial=records[0]['s'];initial_damage=np.dot(initial['weight_m'],initial['damage'])
    for rec in records:
        s=rec['s'];d=rec['d'];t=rec['time'];arc=arclength(rec);rec['arc']=arc
        if previous_energy is None:
            if 'bulk_J_m' in d:previous_energy=d['bulk_J_m']
            elif 'conversion_energy_J_m' in d:
                state=json.loads((REF/'initial-accepted.json').read_text())['state']
                K=np.loadtxt(Path(state['kernel_path'])/'stiffness.csv',delimiter=',')
                c=np.r_[state['affine_memory'],state['memory']]
                previous_energy=float(.5*c@K@c)+d['conversion_energy_J_m']
                d['bulk_J_m']=previous_energy
        if histories and 'bulk_increment_J_m' in d:
            defect=d['bulk_J_m']-d['bulk_increment_J_m']-previous_energy
            if 'quadrature_repartition_J_m' in d:
                assert abs(defect-d['quadrature_repartition_J_m'])<1e-10
            d['quadrature_repartition_J_m']=defect
            d['physical_remainder_with_quadrature_J_m']=d['physical_remainder_J_m']-defect
        previous_energy=d.get('bulk_J_m',previous_energy)
        row=dict(d,time_s=t,step=int(Path(rec['stem']).name.split('-')[-1]),
            imposed_gamma=.0003+.004*min(t,.455),integrated_damage_m=float(np.dot(s['weight_m'],s['damage'])),
            integrated_slip_m2=float(np.dot(s['weight_m'],abs(s['jump_m']))))
        row['growth_m']=row['integrated_damage_m']-initial_damage
        for threshold,name in [(0.1,'process'),(.5,'half'),(1.,'full')]:
            row.update({name+'_'+k:v for k,v in front(s,arc,threshold).items()})
        rec['summary']=row;histories.append(row)
        for q in range(len(s)):
            r={k:s[k][q].item() for k in s.dtype.names};r.update(time_s=t,arc_m=arc[q])
            r['physical_age_s']=r['theta_normalized']*DC/V0
            V=r['velocity_m_s'];r['radiation_Pa']=r['damage']*RAD*V
            r['friction_Pa']=r['resistance_Pa']-r['q_Pa']-r['radiation_Pa']
            r['native_admitted']=r['formed'] and str(r['id']).startswith('admit:')
            profiles.append(r)
            if r['native_admitted']:tracked.append(r)
        for cell in np.unique(s['cell']):
            indices=np.flatnonzero(s['cell']==cell)
            if len(indices)!=3:continue
            if np.all(s['damage'][indices]==1.) and np.all(np.char.startswith(s['id'][indices].astype(str),'admit:')):
                key=tuple(s['id'][indices])
                if key not in maturity:
                    maturity[key]=dict(first_full_s=t,cell=int(cell),admitted_s=float(s['admitted_s'][indices[0]]),
                        weights=s['weight_m'][indices].copy(),slip=s['absolute_slip_m'][indices].copy(),theta=s['theta_normalized'][indices].copy())
    last=records[-1]['s'];events=[]
    for ids,event in maturity.items():
        indices=np.array([np.flatnonzero(last['id']==i)[0] for i in ids]);w=event['weights'];norm=w.sum()
        events.append({k:v for k,v in event.items() if k not in ['weights','slip','theta']} | dict(
            post_full_slip_Dc=float(w@(last['absolute_slip_m'][indices]-event['slip'])/(norm*DC)),
            normalized_age_change=float(w@(last['theta_normalized'][indices]-event['theta'])/norm),
            final_mean_velocity_m_s=float(w@last['velocity_m_s'][indices]/norm),
            final_mean_normal_Pa=float(w@last['pn_work_Pa'][indices]/norm),
            final_mean_q_Pa=float(w@last['q_Pa'][indices]/norm)))
    return histories,profiles,tracked,events


def bulk(rec):
    arrays=[np.genfromtxt(p,delimiter=',',names=True,dtype=None,encoding='utf8') for p in sorted(rec['stem'].parent.glob(rec['stem'].name+'-bulk-r*.csv'))]
    a=np.concatenate(arrays);return a[np.lexsort((a['x_m'],a['y_m']))]


def draw_geometry(ax,rec,limits=(3,11.3,-.35,.95)):
    s=rec['s'];points=np.c_[s['x_m'],s['y_m']];segments=np.stack((points[:-1],points[1:]),axis=1)
    ax.plot(points[:,0],points[:,1],c='k',lw=3.8)
    lc=LineCollection(segments,cmap='magma',norm=plt.Normalize(0,1),linewidths=3)
    lc.set_array(.5*(s['damage'][:-1]+s['damage'][1:]));ax.add_collection(lc)
    virgin=s['formed']==0;ax.plot(s['x_m'][virgin],s['y_m'][virgin],'--',c='.65',lw=1,label='temporary candidate')
    native=np.char.startswith(s['id'].astype(str),'admit:');ax.scatter(s['x_m'][native],s['y_m'][native],s=12,facecolors='none',edgecolors='#36adad',linewidths=.65,label='native-admitted material',zorder=4)
    active=abs(s['velocity_m_s'])>1e-3
    every=np.flatnonzero(active)[::4]
    v=s['velocity_m_s'][every];theta=np.deg2rad(s['tangent_deg'][every]);ax.quiver(s['x_m'][every],s['y_m'][every],v*np.cos(theta),v*np.sin(theta),angles='xy',scale_units='xy',scale=.15,color='#1676b6',width=.003)
    for level,marker in [('process','o'),('full','*')]:
        r=rec['summary'];ax.plot(r[level+'_x_m'],r[level+'_y_m'],marker,color='#16833d' if level=='full' else '#2673ba',ms=8,label='a=1 front' if level=='full' else 'a=.1 front')
    ax.set(xlim=limits[:2],ylim=limits[2:],xlabel='x (m)',ylabel='y (m)');ax.set_aspect('equal');return lc


def plots(records,hist,events,out,make_gif):
    plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.2})
    t=np.array([r['time_s'] for r in hist]);ref=list(csv.DictReader((REF/'main-history.csv').open()));rt=np.array([float(r['time_s']) for r in ref])
    fig,axs=plt.subplots(2,2,figsize=(10,6.5),layout='constrained')
    for ax,key,rkey,label in [(axs[0,0],'growth_m','total_fracture_J_m','Added damage integral (m)'),(axs[0,1],'full_arc_m','full_front_arc_m','Connected a=1 front arc (m)'),(axs[1,0],'integrated_slip_m2','integrated_slip_m2','Integral |jump| (m²)')]:
        val=np.array([r[key] for r in hist]);rv=np.array([float(r[rkey]) for r in ref])
        if key=='growth_m':rv-=hist[0]['integrated_damage_m']
        ax.plot(t,val,'-',label='native ASPECT');ax.plot(rt,rv,'--',label='retained reference');ax.set(xlabel='Time (s)',ylabel=label);ax.axvline(.455,color='.5',ls=':',label='hold starts')
    for key,label in [('selected_preference_deg','Accepted-stress preference'),('full_tangent_deg','Actual a=1-front tangent')]:
        axs[1,1].plot(t,[r.get(key,np.nan) for r in hist],label=label)
    axs[1,1].set(xlabel='Time (s)',ylabel='Angle (degrees)');axs[0,0].legend(fontsize=8);axs[1,1].legend(fontsize=8)
    fig.savefig(out/'native-comparison.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(3,1,figsize=(9,6.3),layout='constrained',sharex=True)
    colors=['#3465a4','#df8f2d','#a52950'];chosen=[records[0],records[min(len(records)-1,int(np.argmin(abs(t-.455))))],records[-1]]
    for rec,color in zip(chosen,colors):
        s=rec['s'];label=f'{rec["time"]:.5f} s';axs[0].plot(s['x_m'],s['damage'],c=color,label=label);axs[1].plot(s['x_m'],s['velocity_m_s'],c=color);axs[2].plot(s['x_m'],s['pn_work_Pa']/1000,c=color)
        axs[2].plot(s['x_m'],s['pn_minus_Pa']/1000,c=color,alpha=.3,lw=.7);axs[2].plot(s['x_m'],s['pn_plus_Pa']/1000,c=color,alpha=.3,lw=.7)
    axs[0].legend(ncol=3);axs[0].set(ylabel='Surface damage a',ylim=(-.02,1.05));axs[1].set(ylabel='Signed slip rate (m/s)');axs[2].set(ylabel='Normal pressure (kPa)',xlabel='Physical x (m)')
    axs[2].text(.02,.04,'Thick: variational reaction. Thin: raw sided diagnostics (accuracy limitations retained).',transform=axs[2].transAxes,fontsize=8)
    fig.savefig(out/'native-profiles.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(10,6),layout='constrained')
    positive=hist[1:];times=t[1:]
    def cumul(k):return np.cumsum([r.get(k,0.) for r in positive])
    for k,label in [('trapezoid_work_J_m','Trapezoidal work'),('fracture_increment_J_m','Added fracture'),('friction_J_m','Friction'),('radiation_J_m','Radiation'),('maxwell_J_m','Maxwell')]:axs[0,0].plot(times,cumul(k),label=label)
    for k,label in [('bulk_increment_J_m','Bulk'),('bond_increment_J_m','Cohesive')]:axs[0,1].plot(times,cumul(k),label=label)
    for k,label in [('physical_remainder_with_quadrature_J_m','Physical remainder'),('BE_bulk_J_m','BE bulk remainder'),('surface_remainder_J_m','Surface endpoint remainder')]:axs[1,0].plot(times,cumul(k),label=label)
    remainder=np.array([r['physical_remainder_with_quadrature_J_m'] for r in positive]);den=cumul('friction_J_m')+cumul('radiation_J_m');frac=cumul('fracture_increment_J_m')
    axs[1,1].plot(times,100*np.cumsum(abs(remainder))/np.maximum(den,1e-30),label='Absolute / sliding');axs[1,1].plot(times,100*np.cumsum(abs(remainder))/np.maximum(frac,1e-30),label='Absolute / fracture');axs[1,1].axhline(1,c='.5',ls=':')
    for ax in axs.flat:ax.legend(fontsize=7);ax.set(xlabel='Time (s)',ylabel='J/m' if ax!=axs[1,1] else '%');ax.axvline(.455,color='.6',ls=':')
    fig.savefig(out/'native-energy.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(2,1,figsize=(10,4.5),layout='constrained');draw_geometry(axs[0],records[0]);lc=draw_geometry(axs[1],records[-1]);axs[0].set_title(f'Initial converted state {records[0]["time"]:.5f} s');axs[1].set_title(f'Native accepted state {records[-1]["time"]:.5f} s; arrows are slip-rate vectors')
    axs[1].legend(ncol=3,fontsize=7,loc='upper left');fig.colorbar(lc,ax=list(axs),label='Surface damage a',shrink=.7);fig.savefig(out/'native-geometry.png',dpi=170);plt.close(fig)
    if events:
        first=min(events,key=lambda e:e['first_full_s']);cell=first['cell'];fig,axs=plt.subplots(2,2,figsize=(10,6),layout='constrained')
        series=[]
        for rec in records:
            s=rec['s'];ids=np.flatnonzero(s['cell']==cell)
            if len(ids)!=3 or not np.all(np.char.startswith(s['id'][ids].astype(str),'admit:')):continue
            w=s['weight_m'][ids];w=w/w.sum();speed=float(w@s['velocity_m_s'][ids]);a=float(w@s['damage'][ids]);q=float(w@s['q_Pa'][ids]);pn=float(w@s['pn_work_Pa'][ids]);res=float(w@s['resistance_Pa'][ids]);theta=float(w@s['theta_normalized'][ids]);slip=float(w@s['absolute_slip_m'][ids]);rad=float(w@(s['damage'][ids]*RAD*s['velocity_m_s'][ids]));series.append((rec['time'],slip/DC,a,speed,theta,pn/1000,q/1000,(res-q-rad)/1000,rad/1000))
        z=np.array(series);axs[0,0].plot(z[:,0],z[:,1]);axs[0,0].set(ylabel='Accumulated slip / Dc');axs[0,0].text(.03,.94,f'After full decohesion: +{first["post_full_slip_Dc"]:.4f} Dc',transform=axs[0,0].transAxes,va='top',fontsize=8)
        axs[0,1].plot(z[:,0],z[:,3],label='Slip rate (m/s)');right=axs[0,1].twinx();right.plot(z[:,0],z[:,4],c='#bc5c2f',label='State');right.set_ylabel('Normalized RSF age',color='#bc5c2f');axs[0,1].set(ylabel='Slip rate (m/s)')
        for col,label in [(6,'Cohesive'),(7,'Friction'),(8,'Radiation')]:axs[1,0].plot(z[:,0],z[:,col],label=label)
        axs[1,0].set(ylabel='Resistance (kPa)');axs[1,0].legend(fontsize=8);axs[1,1].plot(z[:,0],z[:,5]);axs[1,1].set(ylabel='Current normal pressure (kPa)')
        for ax in axs.flat:ax.axvline(first['first_full_s'],c='.4',ls=':',label='whole cell a=1');ax.axvline(.455,c='.7',ls='--');ax.set_xlabel('Time (s)')
        fig.suptitle(f'Native material admitted at {first["admitted_s"]:.5f} s: fracture then mature RSF sliding');fig.savefig(out/'native-mature-material.png',dpi=160);plt.close(fig)
    if not make_gif:return
    frames=[];selection=np.unique(np.r_[np.linspace(0,len(records)-1,min(45,len(records))).astype(int),len(records)-1])
    for i in selection:
        rec=records[i];data=bulk(rec);x=np.unique(data['x_m']);y=np.unique(data['y_m']);dx=x[1]-x[0];dy=y[1]-y[0]
        fig,axs=plt.subplots(2,1,figsize=(8,6.7),layout='constrained',gridspec_kw={'height_ratios':[3,1]})
        field=data['Txy_Pa'].reshape(len(y),len(x))/1000
        im=axs[0].pcolormesh(np.r_[x-dx/2,x[-1]+dx/2],np.r_[y-dy/2,y[-1]+dy/2],field,vmin=-1,vmax=7,cmap='RdBu_r',shading='flat')
        s=rec['s'];axs[0].plot(s['x_m'],s['y_m'],'k-',lw=1);axs[0].set(xlim=(0,12),ylim=(-6,6),xlabel='x (m)',ylabel='y (m)');axs[0].set_aspect('equal');fig.colorbar(im,ax=axs[0],label='Physical reconstructed shear stress τxy (kPa)',shrink=.8)
        lc=draw_geometry(axs[1],rec);axs[1].legend(loc='upper left',fontsize=6,ncol=3);fig.colorbar(lc,ax=axs[1],label='Surface a',shrink=.8)
        fig.suptitle(f'Native ASPECT: t={rec["time"]:.5f} s; γ={rec["summary"]["imposed_gamma"]:.7f}; '+('loading' if rec['time']<=.455+1e-12 else 'displacement hold'))
        fig.canvas.draw();frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()));plt.close(fig)
    frames[0].save(out/'native-propagation.gif',save_all=True,append_images=frames[1:],duration=180,loop=0,optimize=True)


def compare_half_step(hist, path, out, start):
    fine = [{k:float(v) if v else np.nan for k,v in row.items()}
            for row in csv.DictReader(path.open())]
    # Restart exports include the preclosure checkpoint and its unchanged first
    # pending interval. Compare increments only after the common .460 s anchor.
    fine = [row for row in fine if row['time_s'] >= start-1e-12]
    assert fine and abs(fine[0]['time_s']-start) < 1e-12
    coarse = {round(r['time_s'],12):r for r in hist}
    origin = coarse[round(fine[0]['time_s'],12)]
    rows = []
    for f in fine:
        key = round(f['time_s'],12)
        if key not in coarse:continue
        c = coarse[key]
        cg = c['integrated_damage_m']-origin['integrated_damage_m']
        fg = f['integrated_damage_m']-fine[0]['integrated_damage_m']
        cs = c['integrated_slip_m2']-origin['integrated_slip_m2']
        fs = f['integrated_slip_m2']-fine[0]['integrated_slip_m2']
        rows.append(dict(time_s=key,growth_coarse_m=cg,growth_half_m=fg,
                         growth_absolute_m=fg-cg,growth_relative=(fg-cg)/cg if cg else 0.,
                         slip_coarse_m2=cs,slip_half_m2=fs,slip_absolute_m2=fs-cs,
                         slip_relative=(fs-cs)/cs if cs else 0.,
                         full_front_coarse_x_m=c['full_x_m'],full_front_half_x_m=f['full_x_m'],
                         full_front_difference_m=f['full_x_m']-c['full_x_m']))
    table(out/'timestep-comparison.csv',rows)
    (out/'timestep-results.json').write_text(json.dumps(dict(final=rows[-1],
        max_front_difference_m=max(abs(r['full_front_difference_m']) for r in rows),
        common_times=len(rows)),indent=2)+'\n')
    fig,axs=plt.subplots(1,3,figsize=(11,3.2),layout='constrained')
    for ax,ck,fk,label in [(axs[0],'growth_coarse_m','growth_half_m','Added damage integral (m)'),
                           (axs[1],'slip_coarse_m2','slip_half_m2','Added integral |jump| (m²)'),
                           (axs[2],'full_front_coarse_x_m','full_front_half_x_m','Developed front x (m)')]:
        ax.plot([r['time_s'] for r in rows],[r[ck] for r in rows],label='Δt=.000125 s')
        ax.plot([r['time_s'] for r in rows],[r[fk] for r in rows],'--',label='Δt=.0000625 s')
        ax.set(xlabel='Time (s)',ylabel=label);ax.ticklabel_format(axis='x',useOffset=False)
    axs[0].legend(fontsize=8);fig.savefig(out/'native-timestep.png',dpi=160);plt.close(fig)


def active_analysis(labels, out):
    """Only native accepted outputs; no archived trajectory or fixed specimen."""
    records=load(labels);rows=[]
    for r in records:
        s=r['s'];row=dict(r['d'],time_s=r['time'])
        row['integrated_damage_m']=float(s['weight_m']@s['damage'])
        row['integrated_absolute_jump_m2']=float(s['weight_m']@abs(s['jump_m']))
        row['fully_decohered_measure_m']=float(s['weight_m']@(s['damage']==1.))
        rows.append(row)
    initial=rows[0]['integrated_damage_m']
    for row in rows:row['growth_m']=row['integrated_damage_m']-initial
    table(out/'active-history.csv',rows)
    positive=[r for r in rows if r.get('dt_s',0.)>0.]
    loss=sum(r['friction_J_m']+r['radiation_J_m'] for r in positive)
    fracture=sum(r['fracture_increment_J_m'] for r in positive)
    remainder=sum(abs(r['physical_remainder_with_quadrature_J_m']) for r in positive)
    result=dict(runs=labels,start_s=rows[0]['time_s'],end_s=rows[-1]['time_s'],
        growth_m=rows[-1]['growth_m'],positive_steps=len(positive),
        absolute_physical_remainder_J_m=remainder,
        sliding_dissipation_J_m=loss,added_fracture_J_m=fracture,
        relative_sliding=remainder/loss if loss else None,
        relative_fracture=remainder/fracture if fracture else None,
        maximum_source_scaled=max(r.get('source_scaled',0.) for r in rows),
        maximum_weak_Pa=max(r.get('weak_error_Pa',0.) for r in rows),
        minimum_sampling_clearance_m=min(r['sampling_clearance_m'] for r in rows),
        minimum_lifting_clearance_m=min(r['lifting_clearance_m'] for r in rows),
        maximum_rank_memory_bytes=max(r['peak_rank_memory_bytes'] for r in rows))
    (out/'active-results.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axs=plt.subplots(2,2,figsize=(9,6),layout='constrained')
    t=[r['time_s'] for r in rows]
    axs[0,0].plot(t,[r['growth_m'] for r in rows]);axs[0,0].set(xlabel='Time (s)',ylabel='Added damage integral (m)')
    axs[0,1].plot(t,[r['integrated_absolute_jump_m2'] for r in rows]);axs[0,1].set(xlabel='Time (s)',ylabel='Integral |jump| (m²)')
    for r in [records[0],records[-1]]:
        axs[1,0].plot(r['s']['x_m'],r['s']['damage'],label=f"{r['time']:.6g} s")
        axs[1,1].plot(r['s']['x_m'],r['s']['pn_work_Pa'],label=f"{r['time']:.6g} s")
    axs[1,0].set(xlabel='Physical x (m)',ylabel='Surface damage a');axs[1,0].legend()
    axs[1,1].set(xlabel='Physical x (m)',ylabel='Variational normal reaction (Pa)')
    fig.savefig(out/'active-summary.png',dpi=150);plt.close(fig)
    print(json.dumps(result,indent=2))


def main():
    global WORK, OUTPUT_DIRECTORY
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('runs',nargs='+');ap.add_argument('--gif',action='store_true');ap.add_argument('--no-plots',action='store_true');ap.add_argument('--half-history',type=Path);ap.add_argument('--comparison-start',type=float,default=.460)
    ap.add_argument('--active',action='store_true',help='Analyze native outputs without retained-reference files.')
    ap.add_argument('--work-dir',type=Path,default=Path(os.environ.get('ASPECT_COHESIVE_WORK_DIR',WORK.parent)))
    ap.add_argument('--out',type=Path)
    ap.add_argument('--output-directory',type=Path,help='Analyze one directly launched native output directory.')
    args=ap.parse_args();WORK=args.work_dir.expanduser().resolve()/'runs'
    if args.output_directory:
        assert args.active and len(args.runs)==1, '--output-directory requires one --active label'
        OUTPUT_DIRECTORY=args.output_directory.expanduser().resolve()
    out=args.out or (WORK.parent/'analysis'/args.runs[-1] if args.active else OUT)
    out.mkdir(parents=True,exist_ok=True)
    if args.active:
        active_analysis(args.runs,out);return
    records=load(args.runs);hist,profiles,tracked,events=summarize(records)
    table(out/'native-history.csv',hist);table(out/'native-profiles.csv',profiles)
    if tracked:table(out/'native-material.csv',tracked)
    if events:table(out/'native-maturity.csv',events)
    ref={round(float(r['time_s']),12):r for r in csv.DictReader((REF/'main-history.csv').open())};comparisons=[]
    for row in hist:
        key=round(row['time_s'],12)
        if key not in ref:continue
        r=ref[key];comparisons.append(dict(time_s=key,growth_abs_m=row['integrated_damage_m']-float(r['total_fracture_J_m']),slip_abs_m2=row['integrated_slip_m2']-float(r['integrated_slip_m2']),bulk_abs_J_m=row['bulk_J_m']-float(r['bulk_J_m']),full_front_abs_m=row['full_arc_m']-float(r['full_front_arc_m']),full_front_abs_x_m=row['full_x_m']-float(r['full_front_x_m'])))
    if comparisons:table(out/'native-reference-comparison.csv',comparisons)
    windows=[]
    for lo,hi,name in [(.44,.455,'loading'),(.455,.460,'reference hold'),(.460,1.,'extended hold')]:
        rows=[r for r in hist[1:] if r['time_s']>lo+1e-12 and r['time_s']<=hi+1e-12]
        if not rows:continue
        rem=sum(abs(r['physical_remainder_with_quadrature_J_m']) for r in rows);sliding=sum(r['friction_J_m']+r['radiation_J_m'] for r in rows);fracture=sum(r['fracture_increment_J_m'] for r in rows)
        windows.append(dict(window=name,absolute_physical_remainder_J_m=rem,sliding_J_m=sliding,fracture_J_m=fracture,relative_sliding=rem/sliding if sliding else None,relative_fracture=rem/fracture if fracture else None,maximum_BE_ledger_J_m=max(abs(r['ledger_error_J_m']) for r in rows)))
    table(out/'native-energy-windows.csv',windows)
    result=dict(runs=args.runs,superseded_index_rows=load.superseded,start_s=hist[0]['time_s'],end_s=hist[-1]['time_s'],positive_steps=len(hist)-1,growth_m=hist[-1]['growth_m'],full_front_advance_m=hist[-1]['full_arc_m']-hist[0]['full_arc_m'],new_native_mature_cells=len(events),maturity_events=events,energy_windows=windows,
        max_mechanical_weak_Pa=max(r.get('mechanical_weak_Pa',0.) for r in hist),max_surface_weak_Pa=max(r.get('weak_error_Pa',0.) for r in hist),max_source_scaled=max(r.get('source_scaled',0.) for r in hist),max_quadrature_repartition_J_m=max(abs(r.get('quadrature_repartition_J_m',0.)) for r in hist),absolute_quadrature_repartition_J_m=sum(abs(r.get('quadrature_repartition_J_m',0.)) for r in hist),maximum_rank_memory_bytes=max(r.get('peak_rank_memory_bytes',0.) for r in hist),
        comparison_maxima={k:max(abs(r[k]) for r in comparisons) for k in ['growth_abs_m','slip_abs_m2','bulk_abs_J_m','full_front_abs_m','full_front_abs_x_m']} if comparisons else {})
    (out/'native-results.json').write_text(json.dumps(result,indent=2)+'\n')
    if not args.no_plots:plots(records,hist,events,out,args.gif)
    if args.half_history:compare_half_step(hist,args.half_history,out,args.comparison_start)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
