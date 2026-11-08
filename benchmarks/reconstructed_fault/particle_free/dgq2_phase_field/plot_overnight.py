"""Four compact figures for the frozen restored-build coupled validation."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from assess_layered_campaign import load_case,state_at
from layered_fields import localization
COLORS=dict(A='#0072B2',B='#009E73',D='#CC79A7',C='#D55E00')
LABELS=dict(A='A: +U, h=1/8, dt=.0125',B='B: +U, h=1/8, dt=.00625',D='D: +U, h=1/12, dt=.00625',C='C: −U, h=1/8, dt=.0125')

def finish(fig,axes,out,name):
    for ax in np.asarray(axes).flat:ax.grid(alpha=.18)
    fig.savefig(out/name,dpi=170);plt.close(fig)

def plot(cases,result,out):
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for key,c in cases.items():
        t=np.array([r['time'] for r in c['records']]);e=[r['energy'] for r in c['records']];s=result['cases'][key];color=COLORS[key]
        mass=np.array([v['profile_mass'] for v in e]);r=s['native_stage_balance']['records']
        source=np.r_[0,np.cumsum([v['source'] for v in r])];flux=np.r_[0,np.cumsum([v['boundary flux'] for v in r])]
        defects=np.array([v['balance error'] for v in r])
        ax[0,0].plot(t,mass-mass[0],color=color,label=LABELS[key]);ax[0,0].plot(t,source-flux,':',color=color)
        ax[0,1].plot(t,source,color=color,label=key+' source');ax[0,1].plot(t,-flux,'--',color=color,label=key+' −outflow')
        ax[0,2].plot(t,np.r_[0,np.cumsum(defects)],color=color,label=key+' signed');ax[0,2].plot(t,np.r_[0,np.cumsum(abs(defects))],'--',color=color,label=key+' absolute')
        ax[1,0].plot(t,[v['peak_phase'] for v in e],color=color)
        ax[1,1].plot(t,np.sqrt([v['profile_normal_variance'] for v in e]),color=color)
        ax[1,2].plot(t,100*np.array([v['normalization_relative'] for v in e]),color=color)
    titles=['Integral change (solid) / source − flux (dotted)','SSPRK integrated physical contributions','Native numerical balance defect','Peak phase (Gauss8 samples)','Phase-weighted normal width','Localization measure / curve length − 1']
    yl=['Area (m²)','Area (m²)','Area (m²)','Phase','Length (m)','Percent']
    for a,title,y in zip(ax.flat,titles,yl):a.set(title=title,xlabel='Physical time',ylabel=y)
    ax[0,0].legend(fontsize=8);ax[0,1].legend(ncol=2,fontsize=7);ax[0,2].legend(ncol=2,fontsize=7)
    fig.suptitle('DGQ2 coupled profile: physical redistribution and numerical balance are separate')
    finish(fig,ax,out,'profile-balance-shape.png')

    fields=['driving_shear_traction','normal_traction','slip_rate','cumulative_slip','age','I']
    titles=['Total driving shear','Normal traction, including pressure','Slip rate','Accumulated slip','Age','Normalization I']
    fig,ax=plt.subplots(3,6,figsize=(21,10),layout='constrained')
    for j,(field,title) in enumerate(zip(fields,titles)):
        for key,c in cases.items():ax[0,j].plot(c['labels'],c['records'][-1]['profile'][field],color=COLORS[key],label=key)
        ax[0,j].axvspan(0,.2,color='.8',alpha=.3);ax[0,j].axvspan(.8,1,color='.8',alpha=.3)
        ax[0,j].set(title=title,xlabel='Material label; endpoint regions shaded',ylabel=['Pa','Pa','m/s','m','s','m'][j])
        for row,(ka,kb,label) in enumerate([('A','B','Timestep'),('B','D','Bulk mesh')],1):
            pair=next((p for p in result['pairs'] if p['a']==ka and p['b']==kb),None)
            if pair:
                for region,style in [('central','-'),('ends','--')]:
                    t=[v['time'] for v in pair['selected_times']]; y=[v['values']['surface_common_quadrature'][region][field]['relative_L2'] for v in pair['selected_times']]
                    ax[row,j].plot(t,[100*v if v is not None else np.nan for v in y],style,marker='o',markersize=3,label=region)
            ax[row,j].set(xlabel='Physical time',ylabel=label+' difference (%)')
    ax[0,0].legend(ncol=4);ax[1,0].legend();ax[2,0].legend()
    fig.suptitle('Final signed material-label traces and common-quadrature relative L2 differences')
    finish(fig,ax,out,'coupled-traction-slip-state.png')

    # All field plots use the same moving display window and shared column scales.
    # Subtract only U*t from display coordinates; observers retain physical fields.
    fields_data={};gx=np.linspace(-1.6,1.6,321);gy=np.linspace(-1.05,1.05,211)
    xx,yy=np.meshgrid(gx,gy);points=np.c_[xx.ravel(),yy.ravel()]
    for key,c in cases.items():
        path,cmd,rr=load_case(c['case']);s=state_at(path,cmd,int(rr[-1]['step']));shift=c['arguments']['translation']*float(rr[-1]['time'])
        physical=points.copy();physical[:,0]+=shift;v=s.at(physical);chi=localization(s,physical,v)[2]
        fields_data[key]=(np.stack([v[:,0],v[:,5],chi],axis=1).reshape(len(gy),len(gx),3),s.surface[:,1:3]-[shift,0])
    ranges=[(min(v[0][...,j].min() for v in fields_data.values()),max(v[0][...,j].max() for v in fields_data.values())) for j in range(3)]
    fig,ax=plt.subplots(len(cases),3,figsize=(13,3*len(cases)),layout='constrained',squeeze=False)
    for row,key in enumerate(cases):
        array,curve=fields_data[key]
        for j in range(3):
            lo,hi=ranges[j];im=ax[row,j].imshow(array[...,j],extent=[gx[0],gx[-1],gy[0],gy[-1]],origin='lower',aspect='equal',vmin=lo,vmax=hi,cmap='viridis' if j!=1 else 'coolwarm')
            ax[row,j].plot(curve[:,0],curve[:,1],'k-',lw=.8);ax[row,j].set(xlabel='x − U t (display only)',ylabel=key+': y',title=['Phase','Physical stress τxy','Localization χ'][j]);fig.colorbar(im,ax=ax[row,j],shrink=.75,label=['Phase','Pa','1/m'][j])
    fig.suptitle('Final moving-region fields; identical color scales within each column')
    fig.savefig(out/'moving-region-fields.png',dpi=170);plt.close(fig)

    fig,ax=plt.subplots(2,3,figsize=(15,9),layout='constrained')
    for key,c in cases.items():
        t=np.array([r['time'] for r in c['records']]);e=[r['energy'] for r in c['records'][1:]];s=result['cases'][key];col=COLORS[key]
        for j,scheme in enumerate(['endpoint','trapezoid']):
            v=np.array([r[scheme+'_remainder'] for r in e]);ax[0,j].plot(t,np.r_[0,np.cumsum(v)],color=col,label=key+' signed');ax[0,j].plot(t,np.r_[0,np.cumsum(abs(v))],'--',color=col,label=key+' absolute')
        losses=c['stage_energy']['reaction'];tt=np.cumsum([r['dt'] for r in losses]);vv=np.array([r['loss'] for r in losses]);ax[0,2].plot(tt,np.cumsum(vv),color=col,label=key+' signed');ax[0,2].plot(tt,np.cumsum(abs(vv)),'--',color=col)
        ax[1,1].step(t[1:],s['native_stage_balance']['internal_dt'],where='mid',color=col,label=key+' internal Δt')
        ax[1,1].plot(t[1:],[r['dt'] for r in c['records'][1:]],':',color=col)
        ax[1,2].step(t[1:],[r['limited cells'] for r in s['native_stage_balance']['records']],where='mid',color=col,label=key)
    keys=list(cases);xx=np.arange(len(keys));runtime=[result['cases'][k]['wall_s']/60 for k in keys];ax[1,0].bar(xx,runtime,color=[COLORS[k] for k in keys]);ax[1,0].set(xticks=xx,xticklabels=keys)
    for i,k in enumerate(keys):ax[1,0].text(i,runtime[i],str(result['cases'][k]['cost']['stokes_iterations'])+' iterations',ha='center',va='bottom',fontsize=8)
    for a,title,y in zip(ax.flat,['Endpoint cumulative remainder','Trapezoidal cumulative remainder','Reaction-projection loss','Runtime and total linear effort','Physical (dotted) / internal phase timestep','Limiter cell-stage activations per step'],['Energy / out-of-plane length']*3+['Minutes','Time','Count']):a.set(title=title,ylabel=y)
    for a in [ax[0,0],ax[0,1],ax[0,2],ax[1,1],ax[1,2]]:a.set_xlabel('Physical time');a.legend(ncol=2,fontsize=7)
    fig.suptitle('Distinct energy inventories and computational cost; signed cancellation is not accuracy')
    finish(fig,ax,out,'energy-runtime-subcycles.png')
