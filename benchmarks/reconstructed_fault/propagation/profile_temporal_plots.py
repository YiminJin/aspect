#!/usr/bin/env python3
"""Compact lag, scaling, work and spatial diagnosis plots."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from profile_temporal_diagnosis import E, directory, fields, load


def save(fig,name):
    fig.savefig(E/(name+'.png'),dpi=170)
    fig.savefig(E/(name+'.pdf'))
    plt.close(fig)


def main():
    ds=[64,128,256,512,1024,2048,4096]
    fig,ax=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for d in ds:
        rows=load(E/f'diagnosis-{d}/history.json');t=[v['time'] for v in rows];label=f'1/{d} s'
        for a,key in zip(ax.flat,['endpoint_phase_KKT','delta_chi_rate_max','redistributed_interior_force_max','relative_velocity_max']):
            a.plot(t,[v[key] for v in rows],label=label)
        ax[1,1].plot(t,np.cumsum([v['actual_work_difference'] for v in rows]),label=label)
        ax[1,2].plot(t,np.cumsum([v['reaction_work_residual'] for v in rows]),label=label)
    titles=['Endpoint phase KKT (coupling diagnostic)','max |delta chi|/dt [1/(m s)]',
            'Redistributed interior force [Pa]','max |u-lifting| [m/s]',
            'Cumulative bulk minus affine work [J/m]','Cumulative reaction-work residual [J/m]']
    for a,title in zip(ax.flat,titles):
        a.set(xlabel='time [s]',ylabel=title,xlim=(.515625,.625));a.legend(fontsize=7)
    save(fig,'lag-and-scaling')
    fig,ax=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for d in [1024,2048,4096]:
        data=np.load(E/f'diagnosis-{d}/diagnostic-fields.npz');label=f'1/{d} s'
        ax[0].plot(data['yq'],data['final_delta_chi_rate'],label=label)
        ax[1].plot(data['velocity_y'][1:-1],data['final_redistributed_force'][1:-1],label=label)
        y=data['yq'].reshape(256,6).mean(axis=1)
        work=data['accumulated_profile_work'].reshape(256,6).sum(axis=1)
        ax[2].plot(y,work,label=label)
    for a,ylabel in zip(ax,['Final delta chi/dt [1/(m s)]','Final redistributed interior force [Pa]',
                           'Cumulative signed profile work per cell [J/m]']):
        a.set(xlabel='y [m]',ylabel=ylabel,xlim=(.25,.75));a.legend(fontsize=8)
    save(fig,'spatial-contributions')
    fig,ax=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for d in ds:
        phi,R,Z=fields(directory(d));hist=load(directory(d)/'history.json');label=f'1/{d} s'
        ax[0].plot(np.linspace(0,1,257),phi[-1],label=label)
        ax[1].plot([v['slip'] for v in hist],[v['q'] for v in hist],label=label)
        s=load(E/f'diagnosis-{d}/summary.json')
        ax[2].scatter(1/d,abs(s['totals']['actual_work_difference']['signed']),color='C0')
    for d in [64,256,1024]:
        s=load(E/f'smooth-diagnosis-{d}/summary.json')
        ax[2].scatter(1/d,abs(s['totals']['actual_work_difference']['signed']),color='C1')
    ax[0].set(xlabel='y [m]',ylabel='Accepted final phase');ax[0].legend(fontsize=7)
    ax[1].set(xlabel='Cumulative slip [m]',ylabel='Cohesive traction [Pa]');ax[1].legend(fontsize=7)
    ax[2].scatter([],[],color='C0',label='Freely evolving');ax[2].scatter([],[],color='C1',label='Smooth prescribed profile')
    ax[2].set(xlabel='dt [s]',ylabel='|Cumulative bulk-affine work| [J/m]',xscale='log',yscale='log');ax[2].legend(fontsize=8)
    save(fig,'physical-and-control')


if __name__=='__main__':main()
