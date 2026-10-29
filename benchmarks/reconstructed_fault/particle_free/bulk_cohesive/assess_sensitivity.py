#!/usr/bin/env python3
"""Fixed-law cap-flank/domain diagnostics using exported native weak fields.

Nodal weak densities are g/m, not pointwise derivatives. No quadrature history
is interpolated. Independent assembly uses the actual exported positive JxW.
"""
import argparse,json
from pathlib import Path
import numpy as np
from assess_native import csv,gauss,initial_error,connected_front,KF,ELL
from compare_native import grid,interpolate,area


def fields(path,step):
    nodes=csv(path/f'phase-{step:04d}.csv');nodes.sort(order=['y','x'])
    x,y=np.unique(nodes['x']),np.unique(nodes['y']);shape=(len(y),len(x))
    initial=grid(path,0)[2]
    force=csv(path/f'phase-forces-{step:04d}.csv')
    order=np.argsort(force['dof']);force=force[order]
    ids=np.searchsorted(force['dof'],nodes['dof']);force=force[ids]
    assert np.array_equal(force['dof'],nodes['dof'])
    result={k:force[k].reshape(shape) for k in force.dtype.names if k!='dof'}
    result.update(x=x,y=y,initial=initial,delta=nodes['phi'].reshape(shape)-initial)
    result['Yweak']=-result['mechanical_gradient']/result['mass']
    result['Fweak']=result['fracture_gradient']/result['mass']
    result['Rweak']=result['residual']/result['mass']
    return result


def masks(x,y):
    X,Y=np.meshgrid(x,y);L=x[-1]
    return dict(lower_flank=(X>=.625)&(X<=.8125)&(Y>=.25)&(Y<=.4375),
                upper_flank=(X>=.625)&(X<=.8125)&(Y>=.5625)&(Y<=.75),
                core_tip=(X>=.45)&(X<=.85)&(abs(Y-.5)<=.05),
                old_tip_window=(X>=.45)&(X<=.85)&(abs(Y-.5)<=.2),
                right_face=X==L,right_strip=X>=L-1/32,left_face=X==0)


def peak(field,mask,x,y):
    flat=np.flatnonzero(mask);i=flat[np.argmax(field.ravel()[flat])];j,k=divmod(i,len(x))
    return dict(value=float(field[j,k]),x=float(x[k]),y=float(y[j]))


def connected_extents(g,level):
    """Threshold component containing the pre-existing band (four-edge graph).

    Centerline interpolated first crossing is separately authoritative for
    that cut. Graph extents measure off-axis connectivity at nodal resolution,
    not a claim to resolve subcell bilinear saddle topology.
    """
    x,y,p=g;allowed=p>=level;visited=np.zeros_like(allowed);stack=[]
    for j in np.flatnonzero(allowed[:,0]):stack.append((int(j),0));visited[j,0]=True
    while stack:
        j,i=stack.pop()
        for jj,ii in [(j-1,i),(j+1,i),(j,i-1),(j,i+1)]:
            if 0<=jj<len(y) and 0<=ii<len(x) and allowed[jj,ii] and not visited[jj,ii]:
                visited[jj,ii]=True;stack.append((jj,ii))
    X,Y=np.meshgrid(x,y)
    return dict(rightmost_node=float(np.max(X[visited])),ymin=float(np.min(Y[visited])),
                ymax=float(np.max(Y[visited])),touches_right=bool(np.any(visited[:,-1])))


def cut_front(x,p,level):
    if p[0]<level:return None
    below=np.flatnonzero(p<level)
    if not len(below):return float(x[-1])
    j=below[0];i=j-1
    return float(x[i]+(x[j]-x[i])*(p[i]-level)/(p[i]-p[j]))


def independent_weak(path,step):
    d=fields(path,step);x,y=d['x'],d['y'];p=d['phi'];nx,ny=len(x)-1,len(y)-1
    dx,dy=np.diff(x)[0],np.diff(y)[0];f=csv(path/f'bulk-{step:04d}.csv')
    cx=np.clip(np.searchsorted(x,f['x'],side='right')-1,0,nx-1)
    cy=np.clip(np.searchsorted(y,f['y'],side='right')-1,0,ny-1)
    u=(f['x']-x[cx])/dx;v=(f['y']-y[cy])/dy
    gm=np.zeros_like(p)
    for i,j,N in [(0,0,(1-u)*(1-v)),(1,0,u*(1-v)),(0,1,(1-u)*v),(1,1,u*v)]:
        np.add.at(gm,(cy+j,cx+i),-f['JxW']*N*f['driving'])
    gf=np.zeros_like(p);z,w=gauss(2)
    for u,wu in zip(z,w):
        for v,wv in zip(z,w):
            gx=((1-v)*(p[:-1,1:]-p[:-1,:-1])+v*(p[1:,1:]-p[1:,:-1]))/dx
            gy=((1-u)*(p[1:,:-1]-p[:-1,:-1])+u*(p[1:,1:]-p[:-1,1:]))/dy
            for i,j,N,Nu,Nv in [(0,0,(1-u)*(1-v),-(1-v),-(1-u)),(1,0,u*(1-v),1-v,-u),
                                 (0,1,(1-u)*v,-v,1-u),(1,1,u*v,v,u)]:
                gf[j:j+ny,i:i+nx]+=KF*dx*dy*wu*wv*(N+2*ELL**2*(gx*Nu/dx+gy*Nv/dy))
    errors={k:float(np.max(abs(v-d[n])/d['mass']))/KF for k,n,v in
            [('fracture','fracture_gradient',gf),('mechanical','mechanical_gradient',gm),('total','residual',gf+gm)]}
    assert max(errors.values())<1e-10,errors
    assert abs(float(np.sum(f['JxW']))-(x[-1]-x[0])*(y[-1]-y[0]))<1e-12
    return dict(normalized_density_errors=errors,coverage=float(np.sum(f['JxW'])),
                actual_JxW_range=[float(min(f['JxW'])),float(max(f['JxW']))])


def point_statistics(path,step,L):
    f=csv(path/f'bulk-{step:04d}.csv');X,Y=f['x'],f['y']
    regions=dict(lower_flank=(X>=.625)&(X<=.8125)&(Y>=.25)&(Y<=.4375),
                 upper_flank=(X>=.625)&(X<=.8125)&(Y>=.5625)&(Y<=.75),
                 core_tip=(X>=.45)&(X<=.85)&(abs(Y-.5)<=.05),right_strip=X>=L-1/32)
    return {name:{key:dict(minimum=float(min(f[key][mask])),maximum=float(max(f[key][mask])),
                          weighted_mean=float(np.sum(f['JxW'][mask]*f[key][mask])/np.sum(f['JxW'][mask])))
                  for key in ['driving','tauxy','q']} for name,mask in regions.items()}


def measure(path):
    fixture=csv(path/'fixture.csv');ledger=csv(path/'bulk-accepted.csv');nodes=csv(path/'phase-0000.csv')
    assert fixture['step'][-1]==64 and abs(fixture['time'][-1]-.16)<1e-12 and len(ledger)==64
    assert np.array_equal(fixture['publications'],fixture['step'])
    assert np.array_equal(ledger['step'],np.arange(1,len(ledger)+1))
    assert np.array_equal(ledger['publications'],ledger['step'])
    assert np.all(csv(path/'gates.csv')['passed']==1)
    assert max(ledger['mechanical_residual'])<=1e-9 and max(ledger['phase_KKT'])<=1e-7
    assert max(ledger['projection_error'])<=1e-12
    assert max(abs(ledger['ledger_error']))<=1e-10*max(1,max(ledger['bulk']+ledger['cohesive']+ledger['fracture']))
    initial=grid(path,0);xx,yy,p0=initial;region=masks(xx,yy);previous=p0;history=[]
    onset={key:None for key in region}
    for row in fixture:
        step=int(row['step']);g=grid(path,step);delta=g[2]-p0
        assert np.all(g[2]>=previous) and np.all(g[2]<1);previous=g[2]
        h=dict(step=step,time=float(row['time']))
        for key,mask in region.items():
            h[key]=float(np.max(delta[mask]))
            if onset[key] is None and h[key]>1e-10:onset[key]=float(row['time'])
        if row['time']<=.05+1e-12:assert np.max(abs(delta))<=1e-12
        history.append(h)
    end=fields(path,int(fixture['step'][-1]));last=grid(path,int(fixture['step'][-1]));delta=end['delta']
    stats={}
    for key,mask in region.items():
        growing=mask&(delta>1e-10);intact=mask&(p0==0)
        stats[key]=dict(peak_increment=peak(delta,mask,xx,yy),
                        maximum_initially_intact_increment=float(np.max(delta[intact])) if np.any(intact) else None,
                        growing_nodes=int(np.sum(growing)),
                        growth_integral_lumped=float(np.sum(end['mass'][mask]*delta[mask])),
                        final_density_ranges={k:[float(min(end[k][mask])),float(max(end[k][mask]))] for k in ['Yweak','Fweak','Rweak']},
                        maximum_final_residual_density_on_previously_grown_nodes=float(max(abs(end['Rweak'][growing]))) if np.any(growing) else None)
    step_stats={}
    for step in [56,60,64]:
        d=fields(path,step);free=d['phi']>d['obstacle'];active=~free
        step_stats[str(step)]=dict(free_nodes=int(np.sum(free)),
                                  free_residual_density=float(np.max(abs(d['Rweak'][free]))) if np.any(free) else None,
                                  minimum_obstacle_residual_density=float(np.min(d['Rweak'][active])),
                                  natural_KKT=float(np.max(abs(np.minimum(d['phi']-d['obstacle'],d['Rweak']/KF)))),
                                  regions={key:dict(free_nodes=int(np.sum(free&mask)),
                                                    minimum_residual_density=float(np.min(d['Rweak'][mask])),
                                                    maximum_free_residual_density=float(np.max(abs(d['Rweak'][free&mask]))) if np.any(free&mask) else None)
                                           for key,mask in region.items()})
    dgamma=np.where((ledger['time']>.05+1e-12)&(ledger['time']<=.15+1e-12),.018*ledger['dt'],0.)
    reaction=np.divide(ledger['Wext'],dgamma,out=np.zeros(len(dgamma)),where=dgamma>0.)
    W=float(sum(ledger['Wext']));D=float(sum(ledger['Db']+ledger['Dc']+ledger['Df']));E=float(sum(ledger['energy_change']))
    Wtrap=float(np.sum(.5*(reaction+np.r_[0.,reaction[:-1]])*dgamma))
    return dict(domain=[float(xx[-1]),float(yy[-1])],cells=[len(xx)-1,len(yy)-1],
                phase_DoFs=len(xx)*len(yy),DG_tensor_DoFs=int(fixture['DG_DoFs'][0]),local_history_points=int(fixture['points'][0]),
                accepted_steps=len(ledger),initial_phase_relative_L2=initial_error(nodes),onset=onset,regions=stats,
                phase_growth_area=area(last)-area(initial),
                flank_cut_contours={str(cut):{str(level):dict(initial=cut_front(xx,p0[np.argmin(abs(yy-cut))],level),
                                                            final=cut_front(xx,last[2][np.argmin(abs(yy-cut))],level))
                                              for level in [.001,.01,.1,.3,.5]} for cut in [.3125,.6875]},
                contours={str(level):dict(initial=connected_front(nodes,level),final=connected_front(csv(path/f'phase-{int(fixture["step"][-1]):04d}.csv'),level),
                                         initial_component=connected_extents(initial,level),final_component=connected_extents(last,level)) for level in [.001,.01,.1,.3,.5]},
                energy=dict(endpoint_work=W,storage_change=E,physical_dissipation=D,remainder=W-E-D,
                            relative_remainder=abs(W-E-D)/max(1,abs(W)),physical_screen_passed=abs(W-E-D)<=.01*max(1,abs(W)),
                            trapezoidal_work_diagnostic=Wtrap,trapezoidal_remainder_diagnostic=Wtrap-E-D,
                            discretization_remainder=float(sum(ledger['stress_loss']+ledger['traction_loss']+ledger['phase_remainder']+ledger['fracture_loss'])),
                            phase_residual_work=float(sum(ledger['phase_work'])),mechanical_residual_work=float(sum(ledger['mechanical_work'])),
                            maximum_exact_ledger_error=float(max(abs(ledger['ledger_error'])))),
                maximum_mechanical_residual=float(max(ledger['mechanical_residual'])),maximum_phase_KKT=float(max(ledger['phase_KKT'])),
                maximum_source_discrepancy=float(max(ledger['projection_error'])),
                independent_weak={str(s):independent_weak(path,s) for s in [56,60,64]},step_statistics=step_stats,
                actual_quadrature_statistics_at_ramp_end=point_statistics(path,60,xx[-1]),history=history,
                strict_reference_match_passed=False)


def compare(first,second,kind='spatial'):
    metrics=csv(first/'fixture.csv');other=csv(second/'fixture.csv');assert np.array_equal(metrics['time'],other['time'])
    assert metrics['step'][-1]==64 and other['step'][-1]==64
    g0a,g0b=grid(first,0),grid(second,0)
    L=min(g0a[0][-1],g0b[0][-1]);H=min(g0a[1][-1],g0b[1][-1])
    x=np.unique(np.r_[g0a[0][g0a[0]<=L],g0b[0][g0b[0]<=L]])
    y=np.unique(np.r_[g0a[1][g0a[1]<=H],g0b[1][g0b[1]<=H]]);X,Y=np.meshgrid(x,y)
    initial_difference=interpolate(g0a,X,Y)-interpolate(g0b,X,Y)
    region=masks(x,y);result=dict(kind=kind,common_domain=[float(L),float(H)],
                                initial_phase_Linf=float(np.max(abs(initial_difference))),phase_Linf=0.,increment_Linf=0.,
                                regions={key:dict(increment_Linf=0.,density_differences_at_ramp_end={}) for key in region})
    for row in metrics:
        step=int(row['step']);a=grid(first,step);b=grid(second,step);diff=interpolate(a,X,Y)-interpolate(b,X,Y)
        result['phase_Linf']=max(result['phase_Linf'],float(np.max(abs(diff))))
        inc=diff-initial_difference;result['increment_Linf']=max(result['increment_Linf'],float(np.max(abs(inc))))
        for key,mask in region.items():result['regions'][key]['increment_Linf']=max(result['regions'][key]['increment_Linf'],float(np.max(abs(inc[mask]))))
    for step in [56,60,64]:
        a,b=fields(first,step),fields(second,step)
        diff={k:interpolate((a['x'],a['y'],a[k]),X,Y)-interpolate((b['x'],b['y'],b[k]),X,Y) for k in ['Yweak','Fweak','Rweak']}
        density_differences={k:float(np.max(abs(v))) for k,v in diff.items()}
        result[f'density_difference_step_{step}']=density_differences
        if step==60:
            for key,mask in region.items():result['regions'][key]['density_differences_at_ramp_end']={k:float(np.max(abs(v[mask]))) for k,v in diff.items()}
    la,lb=csv(first/'bulk-accepted.csv'),csv(second/'bulk-accepted.csv')
    storage_work=max([abs(float(la[k][-1]-lb[k][-1]))/max(1,abs(float(lb[k][-1]))) for k in ['bulk','cohesive','fracture']]+[abs(sum(la['Wext'])-sum(lb['Wext']))/max(1,abs(sum(lb['Wext'])))])
    result['storage_work_difference']=float(storage_work)
    if kind=='quadrature':result.update(quadrature_phase_screen_passed=result['phase_Linf']<=.001,quadrature_storage_work_screen_passed=bool(storage_work<=.001))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--compare',type=Path);p.add_argument('--kind',choices=['quadrature','spatial','distance'],default='spatial');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=compare(a.run,a.compare,a.kind) if a.compare else measure(a.run)
    a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='history'},indent=2))
