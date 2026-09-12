"""Frozen-output I_h projection discrimination; no production state changes."""
import argparse
import csv
import json

import numpy as np
from pathlib import Path

from audit_profile_identity import phase_interpolator, read, h


def project(path, step, split_bulk, bulk_aligned_basis=False):
    phi, ys = phase_interpolator(path, step)
    surface = read(path, 'surface', step)
    nodes = surface['x']
    bx = np.unique(read(path, 'phase', step)['x'])
    if bulk_aligned_basis:
        nodes = bx
    gn, wn = np.polynomial.legendre.leggauss(12)
    gy = ys[:-1,None]+np.diff(ys)[:,None]*(1+gn)/2
    yw = np.diff(ys)[:,None]*wn/2
    gx, wx = np.polynomial.legendre.leggauss(6 if split_bulk else 3)
    mass = np.zeros((len(nodes),len(nodes)))
    rhs = np.zeros(len(nodes))
    for segment,(a,b) in enumerate(zip(nodes[:-1],nodes[1:])):
        boundaries = np.unique(np.r_[a,b,bx[(bx>a)&(bx<b)]]) if split_bulk else np.array([a,b])
        for left,right in zip(boundaries[:-1],boundaries[1:]):
            for point,weight in zip(left+(right-left)*(gx+1)/2,(right-left)*wx/2):
                shape = np.array([(b-point)/(b-a),(point-a)/(b-a)])
                points = np.c_[np.full(gy.size,point),gy.ravel()]
                integral = np.sum(yw*h(phi(points)).reshape(gy.shape))
                mass[segment:segment+2,segment:segment+2] += weight*np.outer(shape,shape)
                rhs[segment:segment+2] += weight*shape*integral
    return np.linalg.solve(mass,rhs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',type=Path)
    path = parser.parse_args().case
    summary = []
    previous_three = previous_split = previous_aligned = None
    for step in range(len(list(path.glob('time_*.csv')))):
        surface = read(path,'surface',step)
        previous = read(path,'surface',max(0,step-1))
        actual = read(path,'identity_audit',step)
        three = project(path,step,False)
        split = project(path,step,True)
        aligned = project(path,step,True,True)
        bx = np.unique(read(path,'phase',step)['x'])
        if step==0:
            previous_three,previous_split,previous_aligned=three,split,aligned
        dt = float(read(path,'time',step)[0]['dt'])
        factor = np.exp(-dt/100)*np.interp(actual['x'],previous['x'],previous['C'])/(-1e8*np.expm1(-dt/100))
        J,Jp = actual['Ih_independent'],actual['previous_Ih_independent']
        V,scale = actual['V'],np.maximum(abs(actual['V']),1e-5)
        # Recover the same clipped current/old h moments without modifying the
        # original Cp, V, histories, or solving any counterfactual mechanics.
        Js = actual['supported_instantaneous_m_per_s']*actual['Ih_projected']/V
        Jps = Js*actual['previous_Ih_projected']/actual['Ih_projected']-actual['supported_history_m_per_s']/factor
        columns = dict(x=actual['x'],actual_signed_defect=actual['actual_signed_defect'])
        for label,I,Ip in (
            ('independent_three_point_projection',np.interp(actual['x'],surface['x'],three),
             np.interp(actual['x'],previous['x'],previous_three)),
            ('bulk_split_L2_projection',np.interp(actual['x'],surface['x'],split),
             np.interp(actual['x'],previous['x'],previous_split)),
            ('offline_bulk_aligned_Q1',np.interp(actual['x'],bx,aligned),
             np.interp(actual['x'],bx,previous_aligned)),
            ('column_exact_Ih',J,Jp)):
            columns[label+'_full_defect'] = (V-V*J/I-factor*(J*Ip/I-Jp))/scale
            columns[label+'_clipped_defect'] = (V-V*Js/I-factor*(Js*Ip/I-Jps))/scale
        with (path/f'projection_audit_{step}.csv').open('w') as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(zip(*columns.values()))
        summary.append(dict(step=step,
            stored_vs_independent_three_point_nodal_max_m=float(max(abs(surface['Ih']-three))),
            stored_vs_bulk_split_L2_nodal_max_m=float(max(abs(surface['Ih']-split))),
            maxima={key:float(max(abs(value))) for key,value in columns.items() if key!='x'}))
        previous_three,previous_split,previous_aligned=three,split,aligned
    (path/'projection_audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
