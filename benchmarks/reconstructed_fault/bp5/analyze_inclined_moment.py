"""Matched native-work A/B comparison on the unrotated Cartesian square."""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_moment_cycle import samples


def table(path):
    return np.genfromtxt(path, delimiter=',', names=True, dtype=None, encoding=None)


def rms(v, w):
    return float(np.sqrt(np.sum(w*v*v)/np.sum(w)))


def independent_jump(data, tensor, interior=False):
    """Q2 weak assembly, 64x64 square; all four physical velocity walls fixed.

    Use full tensor contraction and shared-node summation, then remove exactly
    the Dirichlet rows. No periodic identification or fitting of stress.
    """
    r=(data[:, :2]+.5)*64
    cell=np.floor(r).astype(int)
    r-=cell
    bx=[2*(r[:, 0]-.5)*(r[:, 0]-1),4*r[:, 0]*(1-r[:, 0]),2*r[:, 0]*(r[:, 0]-.5)]
    by=[2*(r[:, 1]-.5)*(r[:, 1]-1),4*r[:, 1]*(1-r[:, 1]),2*r[:, 1]*(r[:, 1]-.5)]
    dx=[(4*r[:, 0]-3)*64,(4-8*r[:, 0])*64,(4*r[:, 0]-1)*64]
    dy=[(4*r[:, 1]-3)*64,(4-8*r[:, 1])*64,(4*r[:, 1]-1)*64]
    load=np.zeros((129,129,2))
    xx,yy,xy=tensor.T
    for i in range(3):
        for j in range(3):
            ix=2*cell[:, 0]+i;iy=2*cell[:, 1]+j
            for c,value in enumerate((xx*dx[i]*by[j]+xy*bx[i]*dy[j],
                                      xy*dx[i]*by[j]+yy*bx[i]*dy[j])):
                np.add.at(load[:, :, c], (ix,iy), data[:, 2]*value)
    if interior:
        x,y=np.meshgrid(np.arange(1,128)/128-.5,np.arange(1,128)/128-.5,indexing='ij')
        mask=(abs(.5*x+np.sqrt(3)/2*y)<=.2)&(abs(-np.sqrt(3)/2*x+.5*y)<=.1)
        return float(np.linalg.norm(load[1:-1,1:-1][mask]))
    return float(np.linalg.norm(load[1:-1,1:-1]))


def main(root):
    result=dict(definition='Native row averages: sum(JxW*chi*N_i*traction)/sum(JxW*chi*N_i)',
                pressure_convention='Physical FE pressure; volume mean zero. No adiabatic/background pressure added.',
                interior='|s|<=0.2 m measured from the box centre', branches={}, comparisons={})
    profiles={};raw={};bulk={};subcell={};hashes=[];previous_weights=None;geometry=None
    for mode in ('production','native_history_reference'):
        case=root/mode;out=case/f'output-{mode}'
        execution=json.loads((case/'execution.json').read_text())
        assert execution['returncode']==0
        hashes.append((out/'initial_hash.txt').read_text().strip())
        summary=table(out/'summary.csv');flux=table(out/'boundary_flux.csv')
        assert np.array_equal(summary['step'],np.arange(5))
        assert np.all(summary['relative']<1e-8) and np.all(summary['dt']==.1)
        assert np.max(abs(flux['total']))<1e-15
        fresh=[]
        for line in (case/'run.log').read_text().splitlines():
            m=re.search(r'Fault linear solve: iterations=(\d+), fresh=([^, ]+), target=([^, ]+)',line)
            if m:
                assert float(m[2])<=float(m[3]);fresh.append([int(m[1]),float(m[2]),float(m[3])])
        assert len(fresh)>=5
        g=table(out/'geometry.csv');xy=np.column_stack([g['x'],g['y']])
        if geometry is not None:np.testing.assert_array_equal(xy,geometry)
        geometry=xy
        s=xy@np.array([.5,np.sqrt(3)/2]);r=xy@np.array([-np.sqrt(3)/2,.5])
        assert np.max(abs(r))<1e-14
        n=len(g);hs=np.diff(s);interior=abs(s)<=.2
        result['geometry']=dict(cells=[64,64],h=1/64,ell=.15625,ell_over_h=10,
             vertices=n,spacing_min=float(min(hs)),spacing_max=float(max(hs)),
             distance_from_intended_line=float(max(abs(r))),
             interior_vertices=int(sum(interior)),interior_s=[float(min(s[interior])),float(max(s[interior]))],
             interior_distance_from_tips=float(min(s[interior]-s[0])))
        info=dict(execution=execution,fresh_linear_checks=fresh,steps={})
        for step in range(5):
            d=table(out/f'traction_{step}_rank0.csv');raw[mode,step]=d
            w=d['w']*d['chi'];seg=d['segment'].astype(int);xi=d['xi']
            assert np.all((xi>=0)&(xi<=1))
            key=np.column_stack([d[k] for k in ('x','y','w','chi','segment','xi','nx','ny')])
            if previous_weights is not None:np.testing.assert_array_equal(key,previous_weights)
            previous_weights=key
            mass=np.zeros(n);rhs=np.zeros((n,3))
            for index,basis in ((seg,1-xi),(seg+1,xi)):
                np.add.at(mass,index,w*basis)
                for j,name in enumerate(('p','d','sigma')):np.add.at(rhs[:,j],index,w*basis*d[name])
            assert np.all(mass>0)
            np.testing.assert_allclose(np.sum(mass),np.sum(w),rtol=2e-14)
            profile=rhs/mass[:,None];profiles[mode,step]=profile
            np.testing.assert_allclose(profile[:,0]+profile[:,1],profile[:,2],atol=1e-10,rtol=1e-13)
            np.savetxt(out/f'weak_profile_{step}.csv',np.column_stack([s,mass,profile]),delimiter=',',
                       header='s,mass,p,d,sigma',comments='')
            data=samples(out,'fields',step,15);bulk[mode,step]=data
            following=samples(out,'histories',step,9)
            if step==0:assert np.max(abs(following[:,3:6]))==0
            else:
                jump=independent_jump(data,following[:,3:6]-data[:,9:12])
                assert abs(jump-summary['Jtotal'][step])<2e-11
                # The next mechanical solve must consume precisely this history,
                # rather than a newly published FE field or a twice-updated tensor.
                if step<4:
                    upcoming=samples(out,'fields',step+1,15)
                    np.testing.assert_allclose(upcoming[:,12:15],following[:,3:6],rtol=0,atol=1e-10)
            # Neighbouring-chord residual removes a local linear trend, not a
            # physical field. Keep unsmoothed profiles in the CSV/figure.
            f=(s[1:-1]-s[:-2])/(s[2:]-s[:-2])
            chord=profile[1:-1]-(1-f[:,None])*profile[:-2]-f[:,None]*profile[2:]
            mask=interior[1:-1]
            stats={}
            for j,name in enumerate(('p','d','sigma')):
                values=profile[interior,j];weights=mass[interior]
                mean=float(np.average(values,weights=weights))
                stats[name]=dict(mean=mean,minimum=float(min(values)),maximum=float(max(values)),
                     mean_removed_rms=rms(values-mean,weights),chord_rms=rms(chord[mask,j],mass[1:-1][mask]))
            info['steps'][step]=dict(relative=float(summary['relative'][step]),Jtotal=float(summary['Jtotal'][step]),
                 assembly_floor=float(summary['assembly_floor'][step]),tractions=stats)
            # A diagnostic removal of each cell's affine spatial trend separates
            # subcell structure from the smooth finite-box stress variation.
            # This does not alter either saved field or its weak traction.
            residuals=[];cell_weights=[]
            for cell in np.unique(d['cell']):
                indices=np.flatnonzero(d['cell']==cell)
                if len(indices)!=9:continue
                xx=d['x'][indices];yy=d['y'][indices]
                sc=.5*np.mean(xx)+np.sqrt(3)/2*np.mean(yy)
                rc=-np.sqrt(3)/2*np.mean(xx)+.5*np.mean(yy)
                if abs(sc)>.18 or abs(rc)>.035:continue
                design=np.column_stack([np.ones(9),(xx-np.mean(xx))*64,(yy-np.mean(yy))*64])
                values=np.column_stack([d[name][indices] for name in ('p','d','sigma')])
                weights=w[indices];weighted=design*np.sqrt(weights[:,None])
                fit=np.linalg.lstsq(weighted,values*np.sqrt(weights[:,None]),rcond=None)[0]
                residuals.extend(values-design@fit);cell_weights.extend(weights)
            residuals=np.array(residuals);cell_weights=np.array(cell_weights)
            subcell[mode,step]=(residuals,cell_weights)
            info['steps'][step]['cell_affine_residual_rms']={name:rms(residuals[:,j],cell_weights)
                for j,name in enumerate(('p','d','sigma'))}
            if step:
                info['steps'][step]['interior_transfer_load_jump']=independent_jump(data,following[:,3:6]-data[:,9:12],True)
        result['branches'][mode]=info
    assert len(set(hashes))==1
    result['initial_particle_hash']=hashes[0]
    for step in range(5):
        a=bulk['production',step];b=bulk['native_history_reference',step]
        np.testing.assert_array_equal(a[:,:3],b[:,:3])
        du=a[:,3:5]-b[:,3:5];dv=np.linalg.norm(du,axis=1)
        diff=profiles['production',step]-profiles['native_history_reference',step]
        if step<=1:
            np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(diff,np.zeros_like(diff))
        f=(s[1:-1]-s[:-2])/(s[2:]-s[:-2])
        chord=diff[1:-1]-(1-f[:,None])*diff[:-2]-f[:,None]*diff[2:]
        # Matched raw QPs restricted by *projected fault coordinate*, using the
        # same native work weights, show what row averaging may suppress.
        qa=raw['production',step];qb=raw['native_history_reference',step]
        sq=(1-qa['xi'])*s[qa['segment'].astype(int)]+qa['xi']*s[qa['segment'].astype(int)+1]
        qp_mask=abs(sq)<=.2;qw=(qa['w']*qa['chi'])[qp_mask]
        comparison=dict(velocity_max=float(max(dv)),velocity_rms=rms(dv,a[:,2]),tractions={})
        ar,aw=subcell['production',step];br,bw=subcell['native_history_reference',step]
        np.testing.assert_array_equal(aw,bw)
        comparison['cell_affine_residual_difference_rms']={name:rms((ar-br)[:,j],aw)
            for j,name in enumerate(('p','d','sigma'))}
        si=.5*a[:,0]+np.sqrt(3)/2*a[:,1];ri=-np.sqrt(3)/2*a[:,0]+.5*a[:,1]
        interior_qp=(abs(si)<=.2)&(abs(ri)<=.1)
        comparison['interior_velocity_rms']=rms(dv[interior_qp],a[interior_qp,2])
        comparison['interior_velocity_max']=float(max(dv[interior_qp]))
        for j,name in enumerate(('p','d','sigma')):
            comparison['tractions'][name]=dict(difference_rms=rms(diff[interior,j],mass[interior]),
                difference_max=float(max(abs(diff[interior,j]))),
                difference_chord_rms=rms(chord[interior[1:-1],j],mass[1:-1][interior[1:-1]]),
                raw_qp_difference_rms=rms((qa[name]-qb[name])[qp_mask],qw))
        result['comparisons'][step]=comparison
    (root/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axs=plt.subplots(3,2,figsize=(11,9),sharex=True)
    for j,name in enumerate(('p','d','sigma')):
        for col,step in enumerate((1,4)):
            for mode,label in (('production','A: ordinary transfer'),('native_history_reference','B: native retention')):
                # Volume normalization fixes mean(p)=0. Surface pressure=1000
                # is the separate adiabatic friction value, NOT this gauge.
                y=profiles[mode,step][:,j]
                axs[j,col].plot(s[interior],y[interior],'.-',label=label)
            axs[j,col].set_title(f'{name}, t={step*.1:g} s');axs[j,col].set_ylabel('Pa, volume-mean pressure gauge')
            axs[j,col].grid(alpha=.3)
    axs[0,0].legend();axs[-1,0].set_xlabel('s from box centre (m)');axs[-1,1].set_xlabel('s from box centre (m)')
    fig.tight_layout();fig.savefig(root/'interior_tractions.png',dpi=160);plt.close(fig)
    # Actual matched bulk QPs, without interpolating onto a synthetic line.
    a=raw['production',4];b=raw['native_history_reference',4]
    sq=.5*a['x']+np.sqrt(3)/2*a['y'];rq=-np.sqrt(3)/2*a['x']+.5*a['y']
    mask=(abs(sq)<=.2)&(abs(rq)<=.05)
    fig,axs=plt.subplots(3,3,figsize=(13,9),sharex=True,sharey=True)
    for j,name in enumerate(('p','d','sigma')):
        low=min(min(a[name][mask]),min(b[name][mask]));high=max(max(a[name][mask]),max(b[name][mask]))
        for col,(title,values) in enumerate((('A',a[name]),('B',b[name]),('A - B',a[name]-b[name]))):
            artist=axs[j,col].scatter(sq[mask],rq[mask],c=values[mask],s=12,
                vmin=low if col<2 else None,vmax=high if col<2 else None,cmap='coolwarm')
            axs[j,col].set_title(f'{title}: {name} (Pa)');fig.colorbar(artist,ax=axs[j,col])
    for ax in axs[-1]:ax.set_xlabel('s (m)')
    for ax in axs[:,0]:ax.set_ylabel('r (m)')
    fig.tight_layout();fig.savefig(root/'matched_raw_qps.png',dpi=160);plt.close(fig)
    print(json.dumps({'geometry':result['geometry'],'final_comparison':result['comparisons'][4],
        'final_tractions':{m:result['branches'][m]['steps'][4]['tractions'] for m in result['branches']}},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    main(p.parse_args().root)
