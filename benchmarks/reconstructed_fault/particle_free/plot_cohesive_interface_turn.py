#!/usr/bin/env python3
"""Physical-coordinate views of actual native states; no history interpolation."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from PIL import Image

from cohesive_interface_turn import WORK, OUT, Kernel, save, table, G, C0
from analyze_cohesive_interface_turn import summary, rows, state, at, physical_surface

plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.18,'figure.dpi':150})


class View:
    def __init__(self,label):
        self.label=label;self.info=summary(label)
        self.k=Kernel(WORK/self.info['kernel'].removesuffix('-q1'),True,12.,.0021)
        self.q=self.k.s;self.points=np.c_[self.q['x_m'],self.q['y_m']]
        self.arc=np.cumsum(self.k.w)-.5*self.k.w
        mesh=np.genfromtxt(self.k.path/'mesh-curves.csv',delimiter=',',names=True)
        self.polygons=[];self.lines=[]
        for cell in range(self.k.meta['cells']):
            m=mesh[mesh['cell']==cell];faces=[np.c_[m['x_m'][m['face']==f],m['y_m'][m['face']==f]]*12 for f in range(4)]
            self.lines.extend(faces)
            self.polygons.append(np.r_[faces[2],faces[1][1:],faces[3][-2::-1],faces[0][-2:0:-1]])
        shape=(self.k.meta['cells'],len(self.k.full_nodes)+1,5)
        self.field=np.memmap(self.k.path/'field-cell-response.bin',dtype=np.float64,mode='r',shape=shape)

    def values(self,s):
        c=np.r_[s['wall_memory'],s['memory']]
        F=np.einsum('ijk,j->ik',self.field,self.k.field_transform@c)
        V=self.k.Q@np.array(s.get('nodal_rate',np.zeros(len(self.k.nodes))))
        return F,V,10000+self.k.Nm@c,10000+self.k.Np@c

    def map(self,ax,s,stress=True,mesh=False,zoom=True,arrows=True):
        F,V,pm,pp=self.values(s)
        if stress:
            field=PolyCollection(self.polygons,array=F[:,3]/1000,cmap='RdBu_r',clim=(-3,3),edgecolors='none',rasterized=True)
            ax.add_collection(field)
        else:field=None
        if mesh:ax.add_collection(LineCollection(self.lines,colors='.7',linewidths=.3,alpha=.55))
        # Surface pieces use actual Gauss locations and stored damage; the
        # prospective tied tail is a candidate geometry, not damaged material.
        midpoint=.5*(self.points[:-1]+self.points[1:]);segments=np.stack([self.points[:-1],self.points[1:]],axis=1)
        damage=np.array(s['histories']['damage']);colors=.5*(damage[:-1]+damage[1:])
        ax.plot(self.points[:,0],self.points[:,1],color='.35',lw=4,zorder=3)
        surf=LineCollection(segments,array=colors,cmap='viridis',clim=(0,1),linewidths=3,zorder=4);ax.add_collection(surf)
        ax.plot([3,6],[0,0],'k--',lw=.9,zorder=5,label='retained old segment')
        ax.plot(6,0,'ko',ms=4,zorder=6);ax.plot(9,self.points[-1,1],'kx',ms=5,zorder=6)
        for threshold,marker in [(.05,'o'),(.5,'s'),(.999,'D')]:
            active=np.flatnonzero((self.points[:,0]>6)&(damage>=threshold))
            if len(active):
                last=active[-1];ax.plot(*self.points[last],marker=marker,ms=4,mfc='none',mec='white',mew=.9,zorder=8)
        if arrows:
            ids=np.arange(5,len(self.points)-3,12);jump=np.array(s['histories']['jump'])
            # Only arrow length is amplified; numbers/profiles retain metres.
            ax.quiver(self.points[ids,0],self.points[ids,1]+.15,self.q['tx'][ids]*jump[ids],self.q['ty'][ids]*jump[ids],
                      color='k',scale=.006,scale_units='xy',angles='xy',width=.004,zorder=7)
        if zoom:ax.set(xlim=(5.4,9.3),ylim=(-.65,3.25))
        else:ax.set(xlim=(0,12),ylim=(-6,6))
        ax.set_aspect('equal');ax.set(xlabel='x [m]',ylabel='y [m]')
        return field,surf

    def export(self,s,frame):
        F,V,pm,pp=self.values(s);h=s['histories'];data=[]
        for i,p in enumerate(self.points):
            data.append(dict(frame=frame,time_s=s['clock_s'],imposed_gamma=s['gamma'],x_m=p[0],y_m=p[1],
                weight_m=self.k.w[i],tx=self.q['tx'][i],ty=self.q['ty'][i],damage=h['damage'][i],
                signed_jump_m=h['jump'][i],jump_x_m=h['jump'][i]*self.q['tx'][i],jump_y_m=h['jump'][i]*self.q['ty'][i],
                velocity_m_s=V[i],normal_minus_Pa=pm[i],normal_plus_Pa=pp[i],theta=h['theta'][i],
                old_material=int(self.k.oq[i]),candidate_angle_deg=self.k.meta['angle_deg']))
        return data


def energy(ax,label):
    r=rows(label);x=r['time_s'];ax.plot(x,np.cumsum(r['trapezoid_work_J_m']),label='external trapezoidal work')
    info=summary(label);q,s=physical_surface(label,0);a=np.array(s['histories']['damage'])
    e=np.array(s['histories']['jump'])-np.array(s['histories']['reference'])
    bond0=q['weight_m']@(.5*(1-a)/(C0*np.where(a==0,1,a))*e**2)
    bulk0=G*(.5+2*info['axial_ratio']**2)*info['scale_m']**2*s['wall_memory']**2
    ax.plot(x,r['bulk_J_m']+r['bond_J_m']-bulk0-bond0,'--',label='bulk + cohesive storage change')
    for key,name in [('sliding_loss_J_m','friction + radiation'),('maxwell_loss_J_m','Maxwell loss'),('fracture_increment_J_m','additional fracture')]:
        ax.plot(x,np.cumsum(r[key]),label=name)
    ax.set(xlabel='time [s]',ylabel='incremental energy [J/m thickness]');ax.legend(fontsize=8)


def main():
    verification=json.loads((OUT/'verification.json').read_text());angle=verification['primary_selection']['angle_deg']
    chosen='half-p'+str(angle);v=View(chosen);s=state(chosen);r=rows(chosen)
    frames=[]
    fig,axs=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for ax,t in zip(axs[0],[0.,.4,.45]):
        if t==.4:
            snaps=json.loads((WORK/'runs'/chosen/'snapshots.json').read_text());ss=snaps[min(snaps,key=lambda x:abs(float(x)-t))]
        else:ss=state(chosen,t)
        field,surf=v.map(ax,ss,mesh=t==.45);ax.set_title(f't = {ss["clock_s"]:.5f} s, γ = {ss["gamma"]:.5f}')
        frames+=v.export(ss,t)
    fig.colorbar(field,ax=axs[0],label='physical cell-mean τxy [kPa]',shrink=.6)
    fig.colorbar(surf,ax=axs[0],label='surface damage a',shrink=.6,pad=.02)
    F,V,pm,pp=v.values(s);h=s['histories'];x=v.q['x_m']
    axs[1,0].plot(x,np.array(h['jump'])*1000,label='signed slip');axs[1,0].plot(x,np.array(state(chosen,0)['histories']['jump'])*1000,'--',label='old incoming slip')
    axs[1,0].set(xlabel='x [m]',ylabel='slip [mm]');axs[1,0].legend()
    axs[1,1].plot(x,pm/1000,label='lower trace');axs[1,1].plot(x,pp/1000,'--',label='upper trace');axs[1,1].axhline(10,color='.6',lw=1,label='background p₀')
    axs[1,1].set(xlabel='x [m]',ylabel='current normal pressure [kPa]');axs[1,1].legend(fontsize=8)
    energy(axs[1,2],chosen)
    fig.suptitle('Computed +30° terminal candidate: old straight fault retained; new tail born tied\nSlip arrows amplified; white ○/□/◇ mark a≥0.05/0.5/0.999; no opening or bulk phase change')
    fig.savefig(OUT/'selected-evolution.png');plt.close(fig)

    fig,axs=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    candidates=np.genfromtxt(OUT/'candidate-scores.csv',delimiter=',',names=True,dtype=None,encoding=None)
    for width,color in [(1.5,'C0'),(3.,'C1')]:
        c=candidates[candidates['width_m']==width];c=np.sort(c,order='angle_deg')
        axs[0,0].plot(c['angle_deg'],c['mean_new_damage'],'o-',color=color,label=f'turn width {width:g} m')
    for a in [20,30]:
        lo=at('half-p'+str(a))['append_mean_damage'];hi=at('fine-p'+str(a))['append_mean_damage']
        axs[0,0].plot([a,a],[lo,hi],'k-',lw=3)
    axs[0,0].set(xlabel='candidate terminal angle [°]',ylabel='mean NEW damage at 0.45 s');axs[0,0].legend(fontsize=9)
    for label,color in [('angular-p30','C0'),('select-p20','C2'),('select-p0','C3'),('wide-p60','C1')]:
        rr=rows(label);mask=rr['time_s']<=.45+1e-12;axs[0,1].plot(rr['time_s'][mask],rr['append_mean_damage'][mask],color=color,label=label.replace('angular-','w1.5 ').replace('select-','w1.5 ').replace('wide-','w3 '))
    axs[0,1].set(xlabel='time [s]',ylabel='mean new damage');axs[0,1].legend(fontsize=9)
    for ax,label,title in [(axs[1,0],chosen,'Primary family: +30°, width 1.5 m'),(axs[1,1],'wide-fine-p60','Alternative geometry: +60°, width 3 m')]:
        view=v if label==chosen else View(label);field,surf=view.map(ax,state(label),stress=False,mesh=True)
        ax.set_title(title);ax.set(xlim=(5.5,9.3),ylim=(-.4,3.4));fig.colorbar(surf,ax=ax,label='surface damage a',shrink=.6)
        if view is not v:view.k.close()
    fig.suptitle('Selected direction is conditional on the declared candidate geometry and preview rule\nWidth sensitivity remains a physical path assumption; these are separate candidate specimens')
    fig.savefig(OUT/'direction-comparison.png');plt.close(fig)

    mature=View('maturehalf128');ms=json.loads((WORK/'runs/maturehalf128/final.json').read_text())
    fig,axs=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    f,a=mature.map(axs[0,0],ms,stress=True,mesh=False,zoom=False);axs[0,0].set_title('12 × 12 m finite specimen; mature Γ')
    fig.colorbar(f,ax=axs[0,0],label='physical cell-mean τxy [kPa]',shrink=.7)
    for label in ['mature128','maturewide128']:
        view=mature if label=='mature128' else View(label)
        ss=state(label,.55);_,vel,pm,pp=view.values(ss)
        axs[0,1].plot(view.q['x_m'],vel*1000,label=f'{summary(label)["physical_width_m"]:g} m turn')
        axs[1,0].plot(view.q['x_m'],pm/1000,label=f'{summary(label)["physical_width_m"]:g} m lower')
        axs[1,0].plot(view.q['x_m'],pp/1000,'--',label=f'{summary(label)["physical_width_m"]:g} m upper')
        frames+=view.export(ss,label)
        if view is not mature:view.k.close()
    axs[0,1].axvspan(6,7.5,color='.5',alpha=.1,label='primary turn');axs[0,1].set(xlabel='x [m]',ylabel='slip rate at load end [mm/s]');axs[0,1].legend(fontsize=9)
    axs[1,0].set(xlabel='x [m]',ylabel='current normal pressure [kPa]');axs[1,0].legend(fontsize=8)
    energy(axs[1,1],'maturehalf128')
    fig.suptitle('Sustained nonzero sliding through a smooth, closed turn\nGeometry width affects response; no sharp-corner or finite-sliding contact claim')
    fig.savefig(OUT/'turning-mechanics.png');plt.close(fig);mature.k.close()

    fig,axs=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for label in ['fine-p30','half-p30']:
        rr=rows(label);axs[0].plot(rr['time_s'],np.cumsum(abs(rr['physical_remainder_J_m'])),label=label)
        axs[1].plot(rr['time_s'],rr['append_damage_integral_m'],label=label)
        axs[2].semilogy(rr['time_s'],np.maximum(abs(rr['ledger_error_J_m']),1e-17),label=label)
    axs[0].set(xlabel='time [s]',ylabel='sum |physical work remainder| [J/m]');axs[1].set(xlabel='time [s]',ylabel='new ∫a ds [m]')
    axs[2].set(xlabel='time [s]',ylabel='exact backward-Euler ledger error [J/m]')
    for ax in axs:ax.legend(fontsize=9)
    fig.suptitle('Physical work error halves with Δt; algebraic closure is a separate check')
    fig.savefig(OUT/'work-and-time.png');plt.close(fig)

    snapshots=json.loads((WORK/'runs'/chosen/'snapshots.json').read_text());allstates=[state(chosen,0)]+list(snapshots.values())
    selected=[]
    for time in np.r_[np.linspace(0,.3,5),np.linspace(.32,.45,20)]:
        ss=min(allstates,key=lambda ss:abs(ss['clock_s']-time))
        if not selected or ss['clock_s']!=selected[-1]['clock_s']:selected.append(ss)
    images=[]
    for i,ss in enumerate(selected):
        fig,axs=plt.subplots(1,3,figsize=(12,4.6),layout='constrained')
        field,surf=v.map(axs[0],ss,stress=True,mesh=False,zoom=True)
        fig.colorbar(field,ax=axs[0],label='cell-mean τxy [kPa]',shrink=.7);fig.colorbar(surf,ax=axs[0],label='a',shrink=.7,pad=.01)
        F,vel,pm,pp=v.values(ss);x=v.q['x_m'];axs[1].plot(x,ss['histories']['damage'],label='damage a')
        axs[1].plot(x,np.array(ss['histories']['jump'])*1000,label='slip [mm]');axs[1].set(xlim=(3,9),ylim=(0,1.5),xlabel='x [m]');axs[1].legend(fontsize=8)
        axs[2].plot(x,pm/1000,label='lower normal pressure');axs[2].plot(x,pp/1000,'--',label='upper normal pressure')
        axs[2].set(xlim=(3,9),ylim=(7,12.5),xlabel='x [m]',ylabel='current pressure [kPa]');axs[2].legend(fontsize=8)
        fig.suptitle(f'Computed +30° continuation, width 1.5 m | t={ss["clock_s"]:.6f} s, γ={ss["gamma"]:.6f}\nOld straight histories retained; grey future tail is tied; slip arrows amplified')
        fig.canvas.draw();image=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3]);images.append(image.quantize(colors=160));plt.close(fig)
        frames+=v.export(ss,i)
    images[0].save(OUT/'selected-deflection.gif',save_all=True,append_images=images[1:],duration=180,loop=0,optimize=False)
    table(OUT/'plotted-surface-profiles.csv',frames)
    save(OUT/'plot-definitions.json',dict(surface_fields='Actual Gauss3 owned histories and current sided pressure; no constitutive history interpolation.',
        maps='Native Gauss8 physical cell-mean stress; colors are cell averages, not pointwise error estimates.',
        bulk_phase='Not used in this exact-jump experiment. Surface damage a owns the only fracture inventory.',
        arrows='Actual signed tangential jump vector t delta, graphically amplified; not a crack-direction estimator.',
        candidate_tip='Old endpoint (6,0); prospective appended end x=9 m, explicitly tied where a=0.',
        frames=len(images),units=dict(stress='Pa',slip='m',velocity='m/s',energy='J/m thickness',state='dimensionless theta')))
    v.k.close()
    print(json.dumps(dict(pngs=4,gif_frames=len(images),surface_rows=len(frames))))


if __name__=='__main__':main()
