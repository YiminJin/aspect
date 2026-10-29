#!/usr/bin/env python3
"""Reevaluate a saved failed native/DG history cell at 70 decimal digits."""
import argparse,csv,json
from pathlib import Path
import mpmath as mp


def assess(path):
    mp.mp.dps=70
    def table(name):
        with (path/name).open() as stream:
            return [{k:mp.mpf(v) for k,v in row.items()} for row in csv.DictReader(stream)]
    vertices=table('bulk-source-failure-vertices.csv')
    memory=table('bulk-source-failure-memory.csv');tensors=table('bulk-source-failure-tensors.csv')
    with (path/'bulk-source-consistency-failure.csv').open() as stream:failure=next(csv.DictReader(stream))
    def L(t,a,derivative=False):
        if derivative:return 4*t-3 if a==0 else 4-8*t if a==mp.mpf('.5') else 4*t-1
        return 2*t*t-3*t+1 if a==0 else 4*t-4*t*t if a==mp.mpf('.5') else 2*t*t-t
    def basis(u,v,row):return L(u,row['u'])*L(v,row['v'])
    def strain(u,v,name):
        J=mp.matrix([[0,0],[0,0]])
        derivatives=[(-(1-v),-(1-u)),(1-v,-u),(-v,1-u),(v,u)]
        for vertex,dN in zip(vertices,derivatives):
            for i,c in enumerate(['x','y']):
                for j in range(2):J[i,j]+=vertex[c]*dN[j]
        gradient=mp.matrix([[0,0],[0,0]])
        for row in memory:
            d=mp.matrix([[L(u,row['u'],True)*L(v,row['v']),L(u,row['u'])*L(v,row['v'],True)]])*J**-1
            for j in range(2):gradient[int(row['component']),j]+=row[name]*d[j]
        trace=gradient[0,0]+gradient[1,1]
        return mp.matrix([2e6*(gradient[0,0]-trace/3),2e6*(gradient[1,1]-trace/3),1e6*(gradient[0,1]+gradient[1,0])])
    def interpolate(u,v,prefix):
        return mp.matrix([sum(basis(u,v,row)*row[prefix+c] for row in tensors) for c in ['xx','yy','xy']])
    u,v=mp.mpf(failure['u']),mp.mpf(failure['v'])
    beta=mp.mpf(float(1/(1+.0025*.01))) # Actual double beta used by this fixture.
    actual=interpolate(u,v,'new_DG')+strain(u,v,'next')-interpolate(u,v,'new_P')
    expected=beta*(interpolate(u,v,'old_DG')+strain(u,v,'previous')-interpolate(u,v,'old_P')+strain(u,v,'increment'))
    result=dict(cell=failure['cell'],step=int(failure['step']),time=float(failure['time']),
        position=[float(failure[k]) for k in ['x','y']],reference=[float(u),float(v)],
        native_error=float(failure['error']),bound=float(failure['bound']),ratio=float(failure['ratio']),
        component_order=['xx','yy','xy'],
        high_precision_recurrence_difference_Pa=list(map(float,actual-expected)),
        actual_evaluation_roundoff_Pa=list(map(float,mp.matrix([mp.mpf(failure[k]) for k in ['Txx','Tyy','Txy']])-actual)),
        expected_evaluation_roundoff_Pa=list(map(float,mp.matrix([mp.mpf(failure[k]) for k in ['expected_xx','expected_yy','expected_xy']])-expected)),
        FE_recurrence_gradient_roundoff_Pa=list(map(float,strain(u,v,'next')-beta*(strain(u,v,'previous')+strain(u,v,'increment')))))
    z=[(1-mp.sqrt(mp.mpf(3)/5))/2,mp.mpf('.5'),(1+mp.sqrt(mp.mpf(3)/5))/2]
    points=[(i,j) for j in z for i in z]
    inverse=mp.matrix([[basis(i,j,row) for row in tensors] for i,j in points])**-1
    # For nine DGQ2 coefficients and nine source points the Gauss3 discrete
    # mass projection equals interpolation: V^T W V c=V^T W f => V c=f.
    for name,prefix in [('next','new_P'),('previous','old_P')]:
        projected=inverse*mp.matrix([list(strain(i,j,name)) for i,j in points])
        residual=mp.matrix([[row[prefix+c] for c in ['xx','yy','xy']] for row in tensors])-projected
        result['projection_coefficient_roundoff_'+name+'_Pa']=float(max(abs(t) for t in residual))
    increment=inverse*mp.matrix([list(strain(i,j,'increment')) for i,j in points])
    residual=mp.matrix([[row['new_DG'+c]-beta*row['old_DG'+c] for c in ['xx','yy','xy']] for row in tensors])-beta*increment
    result['DG_update_coefficient_roundoff_Pa']=float(max(abs(t) for t in residual))
    result['FE_memory_coefficient_roundoff_m']=float(max(abs(row['next']-beta*(row['previous']+row['increment'])) for row in memory))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();result=assess(args.run)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
