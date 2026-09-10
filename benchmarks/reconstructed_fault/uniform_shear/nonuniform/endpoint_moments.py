#!/usr/bin/env python3
"""Manufactured endpoint moments on the physical K2 strip, without mechanics.

The reference is the analytic integral of a prescribed transverse polynomial.
Wall-clipped Voronoi cells are independently constructed with reflected seeds;
reflection enforces a finite wall, NOT a periodic boundary. Alternative rules
below are diagnostics only and never change production particles or histories.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import Voronoi

LENGTH = .25
HALF_WIDTH = .3088215939070757
BASE_STRESS = 1000.
AMPLITUDE = 5.


def cloud(nx, ratio):
    if not 0 <= ratio < .5:
        raise ValueError('This no-crossing fixture requires displacement/spacing < .5')
    ax=LENGTH/(3*nx)
    ny=round(2*HALF_WIDTH/ax)
    ay=2*HALF_WIDTH/ny
    x,y=np.meshgrid((np.arange(3*nx)+.5)*ax,
                    -HALF_WIDTH+(np.arange(ny)+.5)*ay,indexing='ij')
    x+=ratio*ax*y/HALF_WIDTH
    return np.column_stack((x.ravel(),y.ravel())),ax,ay


def voronoi_cells(points, ax, ay, periodic_x=False):
    ghosts=[]
    # Only seeds near a wall can have its reflection as a relevant neighbor.
    # A reflected competitor cannot beat its original on the physical side.
    for sx in (-1,0,1):
        for sy in (-1,0,1):
            if sx==sy==0:
                continue
            keep=np.ones(len(points),dtype=bool)
            if sx:
                keep &= (points[:,0]<3*ax) if sx<0 else (points[:,0]>LENGTH-3*ax)
            if sy:
                keep &= (points[:,1]<-HALF_WIDTH+3*ay) if sy<0 else (points[:,1]>HALF_WIDTH-3*ay)
            q=points[keep].copy()
            if sx:
                if periodic_x:
                    q[:,0]+=LENGTH if sx<0 else -LENGTH
                else:
                    q[:,0]=-q[:,0] if sx<0 else 2*LENGTH-q[:,0]
            if sy:
                q[:,1]=-2*HALF_WIDTH-q[:,1] if sy<0 else 2*HALF_WIDTH-q[:,1]
            ghosts.append(q)
    vor=Voronoi(np.vstack((points,*ghosts)))
    regions=[vor.regions[r] for r in vor.point_region[:len(points)]]
    if any(len(r)<3 or -1 in r for r in regions):
        raise ValueError('Unbounded/empty owned Voronoi cell')
    polygons=[vor.vertices[r] for r in regions]
    all_vertices=vor.vertices[np.concatenate(regions)]
    if ((not periodic_x and (all_vertices[:,0].min() < -1e-12 or all_vertices[:,0].max()>LENGTH+1e-12))
        or all_vertices[:,1].min() < -HALF_WIDTH-1e-12
        or all_vertices[:,1].max()>HALF_WIDTH+1e-12):
        raise ValueError('Reflected-seed construction did not enforce the same physical strip')
    return polygons


def polygon_moments(polygons, origins):
    """Exact area, first-x and second-x moments in translated coordinates."""
    count=np.array([len(p) for p in polygons])
    owner=np.repeat(np.arange(len(polygons)),count)
    start=np.r_[0,np.cumsum(count)[:-1]]
    end=np.cumsum(count)-1
    next_index=np.arange(sum(count))+1
    next_index[end]=start
    v=np.concatenate(polygons)-origins[owner]
    a,b=v,v[next_index]
    cross=a[:,0]*b[:,1]-b[:,0]*a[:,1]
    moment=np.column_stack((np.bincount(owner,weights=cross)/2,
        np.bincount(owner,weights=(a[:,0]+b[:,0])*cross)/6,
        np.bincount(owner,weights=(a[:,0]**2+a[:,0]*b[:,0]+b[:,0]**2)*cross)/12))
    moment*=np.sign(moment[:,0,None])
    return moment


def clip_x(polygon, boundary, keep_right):
    result=[]
    for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
        inside_a=(a[0]>=boundary) if keep_right else (a[0]<=boundary)
        inside_b=(b[0]>=boundary) if keep_right else (b[0]<=boundary)
        if inside_a:
            result.append(a)
        if inside_a!=inside_b:
            result.append(a+(b-a)*((boundary-a[0])/(b[0]-a[0])))
    return np.array(result)


def assemble_point(points, areas, stress, ds, n):
    segment=np.minimum((points[:,0]/ds).astype(int),n-2)
    xi=(points[:,0]-segment*ds)/ds
    shape=np.column_stack((1-xi,xi))
    mass=np.zeros((n,n));load=np.zeros(n)
    for i in (0,1):
        np.add.at(load,segment+i,areas*shape[:,i]*stress)
        for j in (0,1):
            np.add.at(mass,(segment+i,segment+j),areas*shape[:,i]*shape[:,j])
    return mass,load


def assemble_domains(polygons, points, stress, ds, n, selected):
    """Integrate Q1 products piecewise, splitting cells at fault-node planes."""
    # Assemble only selected domains; unselected ones may use the point rule.
    pieces=[];owners=[];segments=[]
    for p in np.flatnonzero(selected):
        polygon=polygons[p]
        first=max(0,min(n-2,int(np.floor(polygon[:,0].min()/ds))))
        last=max(0,min(n-2,int(np.floor(polygon[:,0].max()/ds))))
        for s in range(first,last+1):
            part=clip_x(clip_x(polygon,s*ds,True),(s+1)*ds,False)
            if len(part)>=3:
                pieces.append(part);owners.append(p);segments.append(s)
    owners=np.array(owners);segments=np.array(segments)
    origins=np.column_stack((segments*ds,points[owners,1]))
    area,x,xx=polygon_moments(pieces,origins).T
    x/=ds;xx/=ds**2
    mass=np.zeros((n,n));load=np.zeros(n)
    np.add.at(mass,(segments,segments),area-2*x+xx)
    np.add.at(mass,(segments,segments+1),x-xx)
    np.add.at(mass,(segments+1,segments),x-xx)
    np.add.at(mass,(segments+1,segments+1),xx)
    np.add.at(load,segments,(area-x)*stress[owners])
    np.add.at(load,segments+1,x*stress[owners])
    return mass,load


def reference(n):
    ds=LENGTH/(n-1)
    mass=np.zeros((n,n))
    for s in range(n-1):
        mass[s:s+2,s:s+2]+=2*HALF_WIDTH*ds/6*np.array([[2,1],[1,2]])
    # Integral of ((y/w)^2-1/3) is exactly zero on [-w,w].
    return mass,np.zeros(n)


def evaluate(nx, ratio, output=None, stress_mode='quadratic', check_periodic=False):
    start=time.monotonic()
    points,ax,ay=cloud(nx,ratio)
    polygons=voronoi_cells(points,ax,ay)
    moments=polygon_moments(polygons,points)
    areas=moments[:,0]
    stress=AMPLITUDE*((points[:,1]/HALF_WIDTH)**2-1/3 if stress_mode=='quadratic'
                     else points[:,1]/HALF_WIDTH)
    n=nx//2+1;ds=LENGTH/(n-1)
    exact_mass,exact_load=reference(n)
    all_domains=np.ones(len(points),dtype=bool)
    lower=np.array([p.min(axis=0) for p in polygons])
    upper=np.array([p.max(axis=0) for p in polygons])
    crosses=np.floor((lower[:,0]+1e-13)/ds)!=np.floor((upper[:,0]-1e-13)/ds)
    wall=(lower[:,0]<1e-13)|(upper[:,0]>LENGTH-1e-13)|(lower[:,1]<-HALF_WIDTH+1e-13)|(upper[:,1]>HALF_WIDTH-1e-13)
    selected=crosses|wall
    point=assemble_point(points,areas,stress,ds,n)
    integrated=assemble_domains(polygons,points,stress,ds,n,all_domains)
    hybrid_parts=assemble_domains(polygons,points,stress,ds,n,selected)
    remaining=assemble_point(points[~selected],areas[~selected],stress[~selected],ds,n)
    methods=dict(point_wall=point,
        interior_area_counterfactual=assemble_point(points,np.full(len(points),ax*ay),stress,ds,n),
        domain_integrated=integrated,
        boundary_and_crossing_integrated=tuple(a+b for a,b in zip(hybrid_parts,remaining)))
    periodic_audit={}
    if check_periodic:
        periodic_polygons=voronoi_cells(points,ax,ay,periodic_x=True)
        periodic_area=polygon_moments(periodic_polygons,points)[:,0]
        methods['periodic_domain_area_point']=assemble_point(points,periodic_area,stress,ds,n)
        # Whole periodic domains cross the seam; wrap their pieces back onto
        # the SAME physical strip before verifying coverage and exact moments.
        wrapped=[];wrapped_owner=[]
        for owner,poly in enumerate(periodic_polygons):
            for shift in (-LENGTH,0,LENGTH):
                q=poly+np.array([shift,0])
                if q[:,0].max()<0 or q[:,0].min()>LENGTH:
                    continue
                part=clip_x(clip_x(q,0,True),LENGTH,False)
                if len(part)>=3:
                    wrapped.append(part);wrapped_owner.append(owner)
        wrapped_owner=np.array(wrapped_owner)
        wrapped_mass,_=assemble_domains(wrapped,points[wrapped_owner],stress[wrapped_owner],ds,n,
                                        np.ones(len(wrapped),dtype=bool))
        periodic_audit=dict(area_error_m2=float(sum(periodic_area)-LENGTH*2*HALF_WIDTH),
            max_area_difference_from_interior_relative=float(max(abs(periodic_area/(ax*ay)-1))),
            wrapped_relative_mass_error=float(np.linalg.norm(wrapped_mass-exact_mass)/np.linalg.norm(exact_mass)))
    result=dict(nx=nx,ratio=ratio,stress_mode=stress_mode,particle_spacing_x_m=ax,particle_spacing_y_m=ay,
                max_displacement_m=ratio*ax,particles=len(points),
                area_error_m2=float(sum(areas)-LENGTH*2*HALF_WIDTH),
                min_area_m2=float(min(areas)),integrated_fraction=float(np.mean(selected)),periodic_audit=periodic_audit,methods={})
    probes=[0,1,n//2,n-2,n-1]
    matrices=dict(reference_mass=exact_mass,reference_centered_load=exact_load,points=points,areas=areas)
    for name,(mass,load) in methods.items():
        represented=np.linalg.solve(mass,load)
        constant_error=max(abs(np.linalg.solve(mass,BASE_STRESS*mass.sum(axis=1))-BASE_STRESS))
        full_load=BASE_STRESS*mass.sum(axis=1)+load
        full_reference=BASE_STRESS*exact_mass.sum(axis=1)
        result['methods'][name]=dict(
            constant_reproduction_error_Pa=float(constant_error),
            relative_mass_error=float(np.linalg.norm(mass-exact_mass)/np.linalg.norm(exact_mass)),
            probe_nodes=probes,probe_diagonal_m2=mass.diagonal()[probes].tolist(),
            probe_row_mass_m2=mass.sum(axis=1)[probes].tolist(),
            probe_full_weak_load_error_Pa_m2=(full_load-full_reference)[probes].tolist(),
            probe_centered_load_Pa_m2=load[probes].tolist(),
            probe_traction_error_Pa=represented[probes].tolist(),
            traction_max_error_Pa=float(max(abs(represented))),
            traction_rms_error_Pa=float(np.sqrt(represented@exact_mass@represented/(2*HALF_WIDTH*LENGTH))))
        matrices[name+'_mass']=mass;matrices[name+'_centered_load']=load
        matrices[name+'_traction']=BASE_STRESS+represented
    result['wall_area_contribution']=dict(
        mass_norm=float(np.linalg.norm(point[0]-methods['interior_area_counterfactual'][0])),
        load_norm_Pa_m2=float(np.linalg.norm(point[1]-methods['interior_area_counterfactual'][1])))
    result['point_coordinate_contribution']=dict(
        mass_norm=float(np.linalg.norm(point[0]-integrated[0])),
        load_norm_Pa_m2=float(np.linalg.norm(point[1]-integrated[1])))
    result['wall_volume_reference_error']=float(np.max(abs(areas-ax*ay))/(ax*ay))
    result['wall_seconds']=time.monotonic()-start
    if output:
        output.mkdir(parents=True,exist_ok=True)
        label=f'nx{nx}_r{ratio:.3f}_{stress_mode}'
        (output/(label+'.json')).write_text(json.dumps(result,indent=2)+'\n')
        np.savez_compressed(output/(label+'.npz'),**matrices)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nx',type=int,default=32)
    p.add_argument('--ratio',type=float,default=.4)
    p.add_argument('--stress',choices=('quadratic','linear'),default='quadratic')
    p.add_argument('--check-periodic',action='store_true',help='Independent tiled-domain diagnostic, never a production flag change')
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'endpoint/moments')
    args=p.parse_args()
    if args.nx<4 or args.nx%2:
        p.error('nx must be even and at least 4')
    print(json.dumps(evaluate(args.nx,args.ratio,args.output,args.stress,args.check_periodic),indent=2))


if __name__=='__main__':
    main()
