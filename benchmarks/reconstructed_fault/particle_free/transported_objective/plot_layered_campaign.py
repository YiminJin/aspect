#!/usr/bin/env python3
"""Compact campaign histories, translation-shifted fields, and fixed-cell animation."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-layered-mpl')
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from assess_layered_campaign import OUT,load_case,state_at
from layered_fields import dot3

def main():
    p=argparse.ArgumentParser();p.add_argument('cases',nargs='+');p.add_argument('--animate');p.add_argument('--prefix',default='layered');a=p.parse_args()
    cases=[json.loads((OUT/(n+'.json')).read_text()) for n in a.cases]
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for index,c in enumerate(cases):
        color=f'C{index%10}'
        t=[r['time'] for r in c['records']];label=c['case'].removeprefix('layered-');labels=np.array(c['labels']);central=(labels>=.2)&(labels<=.8);ends=~central
        for mask,style,suffix in [(central,'-','central'),(ends,'--','ends')]:
            for ax,key in zip(axes.flat[:4],['slip_rate','cumulative_slip','normal_traction','work_shear_traction']):
                ax.plot(t,[np.mean(np.array(r['profile'][key])[mask]) for r in c['records']],style,color=color,label=label+' '+suffix)
        remainder=np.array([r['energy'].get('endpoint_remainder',0) for r in c['records']])
        energy_line,=axes.flat[4].plot(t,np.cumsum(remainder),color=color,label=label+' signed')
        axes.flat[4].plot(t,np.cumsum(abs(remainder)),':',color=energy_line.get_color(),label=label+' absolute')
        axes.flat[5].step(t,[r['dt'] if r['step'] else c['records'][1]['dt'] for r in c['records']],where='pre',color=color,label=label)
    for ax,title in zip(axes.flat,['Slip rate (m/s)','Cumulative slip (m)','Normal traction incl. pressure (Pa)','Work-adjoint shear traction (Pa)','Endpoint remainder: signed / absolute (J/m)','Accepted dt (s)']):
        ax.set(title=title,xlabel='time (s)');ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=7,ncol=2);axes.flat[5].legend(fontsize=7)
    fig.savefig(OUT/(a.prefix+'-histories.png'),dpi=160);fig.savefig(OUT/(a.prefix+'-histories.pdf'));plt.close(fig)
    selected=cases[:3]
    fig,axes=plt.subplots(3,len(selected),figsize=(4*len(selected),8),squeeze=False,constrained_layout=True)
    for row,(key,title) in enumerate([('slip_rate','V (m/s)'),('cumulative_slip','Cumulative slip (m)'),('age','log10(age / s)')]):
        values=[np.array([r['profile'][key] for r in c['records']]) for c in selected]
        if key=='age':values=[np.log10(v) for v in values]
        lo=min(v.min() for v in values);hi=max(v.max() for v in values)
        for col,(c,value) in enumerate(zip(selected,values)):
            times=[r['time'] for r in c['records']]
            im=axes[row,col].pcolormesh(c['labels'],times,value,shading='nearest',vmin=lo,vmax=hi,cmap='viridis')
            axes[row,col].set(xlabel='material label',ylabel='time (s)',xlim=(0,1),ylim=(0,times[-1]))
            for edge in (.2,.8):axes[row,col].axvline(edge,color='white',lw=.6,ls='--')
            if row==0:axes[row,col].set_title(c['case'].removeprefix('layered-'))
        fig.colorbar(im,ax=axes[row,:].tolist(),label=title)
    fig.savefig(OUT/(a.prefix+'-label-histories.png'),dpi=160);plt.close(fig)
    selected=cases[:3];fig,axes=plt.subplots(2,len(selected),figsize=(4*len(selected),6),squeeze=False,constrained_layout=True)
    field_images=[[],[]]
    for j,c in enumerate(selected):
        path,cmd,rows=load_case(c['case']);s=state_at(path,cmd,int(rows[-1]['step']));t=c['records'][-1]['time'];shift=cmd['arguments']['translation']*t
        xx=np.linspace(-2.,2.5,301);yy=np.linspace(-1.6,1.6,201);x=np.stack(np.meshgrid(xx,yy),axis=-1);sample=x.reshape(-1,2).copy();sample[:,0]+=shift;v=s.at(sample);tau=v[:,3:6]
        for i,z in enumerate([np.sqrt(dot3(tau,tau)),v[:,0]]):
            im=axes[i,j].pcolormesh(xx,yy,z.reshape(len(yy),len(xx)),shading='auto',cmap='viridis');fig.colorbar(im,ax=axes[i,j],label='stress norm (Pa)' if i==0 else 'phase')
            field_images[i].append(im)
            axes[i,j].plot(s.surface[:,1]-shift,s.surface[:,2],'w.-',lw=1,ms=2);axes[i,j].set(aspect='equal',xlabel='x - U t (m)',ylabel='y (m)')
        axes[0,j].set_title(c['case'].removeprefix('layered-'))
    for row in field_images:
        upper=max(float(im.get_array().max()) for im in row)
        for im in row:im.set_clim(0,upper)
    fig.savefig(OUT/(a.prefix+'-shifted-fields.png'),dpi=160);plt.close(fig)
    if a.animate:
        path,cmd,rows=load_case(a.animate);frames=rows[::2]
        if frames[-1]!=rows[-1]:frames.append(rows[-1])
        xx=np.linspace(-2.,4.5,391);yy=np.linspace(-1.6,1.6,193);x=np.stack(np.meshgrid(xx,yy),axis=-1).reshape(-1,2)
        fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True);image=ax.imshow(np.zeros((len(yy),len(xx))),extent=[xx[0],xx[-1],yy[0],yy[-1]],origin='lower',vmin=0,vmax=.35,cmap='magma',aspect='equal')
        line,=ax.plot([],[],'c.-',lw=1,ms=3);fig.colorbar(image,ax=ax,label='Q1 phase')
        h=4/(2**cmd['arguments']['level']*cmd['arguments'].get('depth_repetitions',1))
        for v in np.arange(-cmd['arguments']['halfwidth'],xx[-1]+h/2,h):
            if v>=xx[0]:ax.axvline(v,color='white',lw=.25,alpha=.25)
        for v in np.arange(-2,yy[-1]+h/2,h):
            if v>=yy[0]:ax.axhline(v,color='white',lw=.25,alpha=.25)
        ax.set(xlabel='physical x (m)',ylabel='physical y (m)')
        def update(i):
            r=frames[i];s=state_at(path,cmd,int(r['step']));image.set_data(s.at(x)[:,0].reshape(len(yy),len(xx)));line.set_data(s.surface[:,1],s.surface[:,2]);ax.set_title(f"{a.animate}: t={float(r['time']):.4f} s; fixed h={s.h:g} m cells")
            return image,line
        animation=FuncAnimation(fig,update,frames=len(frames),interval=150,blit=False)
        animation.save(OUT/'layered-cell-crossing.gif',writer=PillowWriter(fps=7));plt.close(fig)
if __name__=='__main__':main()
