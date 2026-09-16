"""Audit saved BP3 initialization without advancing or changing any history.

Normal-pressure CSV values are actual constitutive weak moments, not bulk
column averages. Older output lacks the mass matrix, so its pressure results
remain explicitly labeled Q1-test-weighted means. Visualization stresses are
old FE-history inputs, NOT accepted current Maxwell stresses.
"""
import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np
from scipy.linalg import solve_banded
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader


def table(path):
    return np.atleast_1d(np.genfromtxt(path, delimiter=',', names=True))


def summarize(values):
    return dict(min=float(np.min(values)), max=float(np.max(values)),
                mean=float(np.mean(values)))


def analyze(directory):
    surface = table(directory/'fault_0.csv')
    normal = table(directory/'constitutive_normal_0_rank0.csv')
    targets = table(directory/'initial_traction_target.csv')
    log = directory.with_suffix('.log').read_text()
    residuals = re.findall(r'after nonlinear iteration (\d+): ([^,\n]+), ([^\n]+)', log)
    if not residuals or max(map(float, residuals[-1][1:])) >= 1e-8:
        raise ValueError('No genuinely converged final nonlinear residual')
    linear = re.findall(r'Fault linear solve: iterations=(\d+), estimated=([^,]+), fresh=([^,]+), target=([^,]+)', log)
    if not linear or any(float(fresh) > float(target) for _, _, fresh, target in linear):
        raise ValueError('A returned direction failed its fresh linear residual check')
    free = surface['prescribed'] == 0
    xd = np.maximum(surface['xd'], 0.)
    initial_theta = np.interp(xd, targets['xd'], targets['Theta'])
    qtarget = np.interp(xd, targets['xd'], targets['q_target'])
    result = dict(
        nonlinear=dict(iteration=int(residuals[-1][0]),
                       bulk_relative=float(residuals[-1][1]), surface_relative=float(residuals[-1][2])),
        linear=[dict(iterations=int(n), estimated=float(e), fresh=float(f), target=float(t))
                for n,e,f,t in linear],
        free_V_over_official=summarize(surface['V'][free]/1e-9),
        prescribed_V_absolute_error=float(np.max(np.abs(surface['V'][~free]-1e-9))),
        retained_Theta_relative_error=float(np.max(np.abs(surface['Theta']/initial_theta-1))),
        fault_normal_distance_max=float(np.max(np.abs((78867.5134594813-surface['x'])*np.sqrt(3)/2
                                                      -(100000-surface['y'])*.5))),
        sampled_constitutive_sigma_Pa=dict(min=float(normal['sigma_min'].min()),
                                         max=float(normal['sigma_max'].max())),
        free_q_minus_airy_target_Pa=summarize((surface['q']-qtarget)[free]),
        initialization_verdict='NONPASSING: official Vinit is not reproduced; dynamics not authorized by this audit')
    bands = []
    for lo,hi in [(0,500),(500,2500),(2500,15000),(15000,18000),(18000,40000),(40000,120000)]:
        selected=(xd>=lo)&(xd<hi)
        bands.append(dict(xd_m=[lo,hi], V_over_official=summarize(surface['V'][selected]/1e-9)))
    result['down_dip_bands']=bands
    # Retain both represented traction and directly measured weak means. A
    # nonzero friction residual on a prescribed deep node is not a failed PDE.
    order=np.argsort(xd)
    fields={key:surface[key] for key in ['V','Theta','C','Ih','slip','q']}
    pressure_label='Q1_test_weighted_mean'
    for name in ['p','sigma','tauN']:
        fields[name]=normal[name+'_load']/normal['weight']
    if 'mass_diagonal' in surface.dtype.names:
        mass=np.zeros((3,len(surface)))
        mass[1]=surface['mass_diagonal']
        mass[0,1:]=mass[2,:-1]=surface['mass_upper'][:-1]
        for name in ['p','sigma','tauN']:
            fields[name]=solve_banded((1,1),mass,normal[name+'_load'])
        pressure_label='consistent_Q1_projection'
    result['pressure_output_semantics']=pressure_label
    stations=[0,2500,5000,7500,10000,12500,15000,17500,20000,25000,30000,35000]
    with (directory/'stations_initial_audit.csv').open('w') as stream:
        out=csv.writer(stream)
        out.writerow(['xd']+list(fields))
        for station in stations:
            out.writerow([station]+[np.interp(station,xd[order],values[order]) for values in fields.values()])
    result['surface_intersection']={name:float(values[-1]) for name,values in fields.items()}

    reader=vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(directory/'solution/solution-00000.0000.vtu'))
    reader.Update()
    grid=reader.GetOutput()
    data=grid.GetPointData()
    points=vtk_to_numpy(grid.GetPoints().GetData())
    velocity=vtk_to_numpy(data.GetArray('velocity'))
    speed=np.linalg.norm(velocity,axis=1)
    k=np.argmax(speed)
    result['bulk_visualization']=dict(
        max_speed_m_per_s=float(speed[k]),max_speed_position_m=points[k].tolist(),
        pressure_Pa=summarize(vtk_to_numpy(data.GetArray('p'))),
        phase=summarize(vtk_to_numpy(data.GetArray('phase_field'))),
        stress_array_semantics='tau_xx, tau_yy, tau_xy are transferred old-history inputs, not current stress')
    result['resource']=json.loads(directory.with_suffix('.resources.json').read_text())
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    result=analyze(args.directory)
    (args.directory/'initialization_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
