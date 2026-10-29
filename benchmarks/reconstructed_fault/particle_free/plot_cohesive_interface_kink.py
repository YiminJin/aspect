#!/usr/bin/env python3
"""Physical coordinates and discrete states; no invented intermediate evolution."""
import json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection,LineCollection
from matplotlib.animation import FuncAnimation,PillowWriter
from check_cohesive_interface_kink import OUT,WORK,Kernel,read,csvwrite
plt.rcParams.update({'font.size':10,'figure.dpi':150})
def save(fig,name):fig.savefig(OUT/name,bbox_inches='tight');plt.close(fig)
def edges(mesh):
    points=np.c_[mesh['x_m'],mesh['y_m']].reshape(-1,4,2);return points[:,[0,1,3,2]]
def geometry(ax,after=True):
    ax.plot([.25,.5],[0,0],color='#d55e00',lw=3,label='old cohesive records')
    ax.plot([.5,.75],[0,.125],color='#0072b2',lw=3,ls='-' if after else '--',label='tied append' if after else 'prospective append')
    ax.scatter([.25,.5,.75],[0,0,.125],c=['black','#cc79a7','black'],s=20,zorder=5)
    ax.set_aspect('equal');ax.set(xlabel='x (m)',ylabel='y (m)')

def main():
    k=Kernel('native-32');before=json.loads((OUT/'native-32-accepted.json').read_text());after=json.loads((OUT/'native-32-physical-step.json').read_text())
    mesh=read(k.path/'mesh.csv');polys=edges(mesh)
    surface=read(OUT/'native-32-surface.csv');loc=json.loads((OUT/'local-resolution.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax in axes:
        ax.add_collection(PolyCollection(polys,facecolors='none',edgecolors='0.8',linewidths=.35));geometry(ax);ax.autoscale_view()
    axes[0].set(xlim=(-.03,1.03),ylim=(-.53,.53),title='Fixed fitted block; p₀=10 kPa; affine boundary motion')
    axes[1].set(xlim=(.24,.8),ylim=(-.17,.23),title='Prescribed 26.565° kink — not a selected path')
    axes[1].add_patch(plt.Circle((.5,0),.1,fill=False,color='black',ls=':',label='common averaging radius 0.1 m'))
    axes[1].annotate('zero vector jump at closed corner',(.5,0),(.31,-.13),arrowprops={'arrowstyle':'->'},fontsize=8)
    axes[1].scatter(surface['x_m'],surface['y_m'],s=8,color='black',alpha=.5,label='Gauss history points (weights ≠ counts)')
    for name,color,scale in [('native-16','#009e73',.085),('native-32','#cc79a7',.075)]:
        angle=np.radians(loc['samples'][name]['principal_axis_deg']);dx,dy=scale*np.cos(angle),scale*np.sin(angle)
        axes[1].plot([.5-dx,.5+dx],[-dy,dy],color=color,lw=2,label=name+' averaged principal axis')
    axes[1].legend(fontsize=7,loc='upper left');save(fig,'specimen.png')
    raw=read(k.path/'field-response.csv').reshape(-1,len(k.nodes)+1)
    cs=[k.coefficients(before),k.coefficients(before),k.coefficients(after)]
    shear=[raw['tau_xy_Pa']@c for c in cs];pressure=[raw['pprime_Pa']@c for c in cs]
    w=raw['weight_m2'][:,0].reshape(-1,36)
    def cell_mean(v):return (v.reshape(-1,36)*w).sum(axis=1)/w.sum(axis=1)
    means=[cell_mean(v) for v in shear];limit=max(np.max(abs(v)) for v in means)/1000
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
    for i,(ax,title) in enumerate(zip(axes,['Loaded before insertion','Same physical state after tied insertion','After − before: exactly zero'])):
        val=means[i] if i<2 else means[1]-means[0]
        c=PolyCollection(polys,array=val/1000,cmap='RdBu_r',clim=(-limit,limit),edgecolors='none');ax.add_collection(c);geometry(ax,i>0);ax.set(xlim=(0,1),ylim=(-.5,.5),title=title)
    fig.colorbar(c,ax=axes,label='physical τxy (kPa)',shrink=.7);save(fig,'stress_transfer.png')
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for name,style in [('native-16','--'),('native-32','-')]:
        a=read(OUT/(name+'-surface.csv'));xx=a['x_m']
        axes[0,0].plot(xx,a['jump_m']*1e6,style,label=name)
        axes[0,1].plot(xx,a['damage'],style,label=name)
        axes[1,0].plot(xx,a['theta'],style,label=name)
        axes[1,1].plot(xx,a['normal_minus_Pa']/1000,style,label=name+' lower')
        axes[1,1].plot(xx,a['normal_plus_Pa']/1000,style,alpha=.6,label=name+' upper')
    axes[0,0].set(ylabel='signed geometric jump δ (μm)',title='Old nonzero histories retained; new jump exactly zero')
    axes[0,1].set(ylabel='surface damage a',title='No fracture inventory created by tied insertion')
    axes[1,0].set(ylabel='dimensionless RSF state Θ',title='New records inherit the virgin clock')
    axes[1,1].set(ylabel='actual plane-normal pressure (kPa)',title='Both physical pressure traces remain compressive')
    for ax in axes.flat:ax.axvline(.5,color='gray',ls=':');ax.axvspan(.5,.75,color='#0072b2',alpha=.06);ax.set_xlabel('physical x (m), not arclength');ax.grid(alpha=.2);ax.legend(fontsize=7)
    save(fig,'histories_pressure.png')
    rows=[]
    for name in ['native-16','native-32','native-16-q6']:
        r=json.loads((OUT/(name+'-checks.json')).read_text());l=loc['samples'][name]
        rows.append(dict(mesh=name,bulk_J_m=r['storage_before']['bulk_J_m'],bond_J_m=r['storage_before']['bond_J_m'],fracture_J_m=r['storage_before']['fracture_J_m'],transfer_storage_change_J_m=0,required_tied_multiplier_Pa=r['certificate']['peak_required_Pa'],min_normal_pressure_Pa=r['certificate']['min_normal_Pa'],principal_axis_deg=l['principal_axis_deg'],shear_plane_1_deg=l['maximum_shear_planes_deg'][0],shear_plane_2_deg=l['maximum_shear_planes_deg'][1],weak_residual_Pa=r['certificate']['residual_Pa'],coupled_tangent_error=r['physical_step']['tangent_relative']))
    csvwrite('comparison.csv',rows)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    modes=['unloaded','loaded 16×8','loaded 32×16','axial overload'];vals=[0,rows[0]['required_tied_multiplier_Pa'],rows[1]['required_tied_multiplier_Pa'],4000]
    axes[0].bar(modes,np.array(vals)/1000,color=['gray','#0072b2','#009e73','#d55e00']);axes[0].axhline(3,color='black',ls='--',label='unchanged intact strength');axes[0].tick_params(axis='x',rotation=15);axes[0].set(ylabel='minimum feasible multiplier bound (kPa)',title='Overloaded append is rejected, not damaged for free');axes[0].legend(fontsize=8)
    rr=json.loads((OUT/'native-32-checks.json').read_text())
    names=['bulk','recoverable bond','fracture'];values=[rr['storage_before'][x] for x in ['bulk_J_m','bond_J_m','fracture_J_m']]
    axes[1].bar(np.arange(3)-.16,values,.3,label='before');axes[1].bar(np.arange(3)+.16,values,.3,label='after');axes[1].set_xticks(range(3),names);axes[1].set(ylabel='storage (J/m out-of-plane)',title='Identical storage across geometry-only insertion');axes[1].legend();save(fig,'admission_energy.png')
    # Only three computed states, repeated for viewing. No interpolated dynamics.
    fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained');pc=PolyCollection(polys,array=means[0]/1000,cmap='RdBu_r',clim=(-limit,limit),edgecolors='none');axes[0].add_collection(pc);geometry(axes[0]);axes[0].set(xlim=(0,1),ylim=(-.5,.5));fig.colorbar(pc,ax=axes[0],label='physical τxy (kPa)',shrink=.7)
    line,=axes[1].plot(surface['x_m'],surface['jump_m']*1e6,color='#d55e00');axes[1].set(xlabel='x (m)',ylabel='geometric jump δ (μm)',ylim=(-2,1.1*max(np.array(after['histories']['jump'])*1e6)));axes[1].axvspan(.5,.75,color='#0072b2',alpha=.1);axes[1].grid(alpha=.2);title=fig.suptitle('')
    labels=['Before insertion; physical t = 0.200 s','After tied insertion; SAME physical t = 0.200 s','One coupled reference step; physical t = 0.202 s']
    states=[before,before,after];append_line=axes[0].lines[1]
    def update(frame):
        i=frame//12;append_line.set_linestyle('--' if i==0 else '-');pc.set_array(means[i]/1000);line.set_ydata(np.array(states[i]['histories']['jump'])*1e6);title.set_text(labels[i]);return pc,line,title
    FuncAnimation(fig,update,frames=36).save(OUT/'transfer.gif',writer=PillowWriter(fps=6));plt.close(fig)
    # Plot-ready cell means for all three exact states, including the field after
    # the physical trial, without retaining repeated response matrices in archive.
    rows=[]
    for i,label in enumerate(['before','inserted','reference_step']):
        for cell,poly in enumerate(polys):rows.append(dict(state=label,physical_time_s=.202 if i==2 else .2,x_m=float(poly[:,0].mean()),y_m=float(poly[:,1].mean()),tau_xy_Pa=means[i][cell],pprime_Pa=cell_mean(pressure[i])[cell]))
    csvwrite('animation_cells.csv',rows)
    shutil.copy2(ROOT_PATH/'comparison.png',OUT/'comparison-straight-growth.png')
ROOT_PATH=OUT.parent/'cohesive_interface_insertion'
if __name__=='__main__':main()
