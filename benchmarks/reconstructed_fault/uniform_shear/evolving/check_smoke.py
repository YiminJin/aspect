"""Read-only, per-accepted-state K3 gates; no simulation launch or fitting."""
import argparse
import json
import re
from pathlib import Path
import time

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from reference import read_parameters


def read(path, name, step):
    return np.atleast_1d(np.genfromtxt(path/f'{name}_{step}.csv', delimiter=',', names=True))


def check(path, step):
    start = time.monotonic()
    surface = read(path, 'surface', step)
    initial_surface = read(path, 'surface', 0)
    phase = read(path, 'phase', step)
    bulk = read(path, 'bulk', step)
    particles = read(path, 'particles', step)
    initial_particles = read(path, 'particles', 0)
    segments = read(path, 'segments', step)
    checks = {}
    checks['unchanged_geometry'] = all(np.array_equal(surface[k], initial_surface[k]) for k in ('fault','node','x','y'))
    checks['admissible_phi'] = bool(np.min(phase['phi']) >= -1e-4 and np.max(phase['phi']) < .8)
    xs, ys = np.unique(phase['x']), np.unique(phase['y'])
    refinement_scope = None
    common_case = re.fullmatch(r'spatial0375_n(128|256|512)(?:_f(32|64))?(?:_periodic)?(?:_floor)?',path.name)
    if path.name == 'normal256' or common_case:
        baseline = path.parent/'smoke'
        old = read_parameters(baseline/'parameters.prm')
        new = read_parameters(path/'parameters.prm')
        refinement_scope = {key:[old.get(key),new.get(key)] for key in old.keys() | new.keys()
                            if old.get(key) != new.get(key)}
        allowed = {'Output directory','Geometry model/Box/Y repetitions'}
        if '_periodic' in path.name:
            # Same fixture, rebuilt against the corrected domain ABI. This
            # permits only the library path change, not a numerical parameter.
            allowed.add('Additional shared libraries')
        if common_case:
            allowed |= {'Maximum time step','Maximum first time step','End time',
                        'Termination criteria/End step'}
            normal_cells = int(common_case.group(1))
            if common_case.group(2):
                allowed.add('Fault reconstruction/Structural point spacing')
            time_data = read(path,'time',step)[0]
            checks['common_timestep_sequence'] = (time_data['time']==step*.375
                and time_data['dt']==(2. if step==0 else .375))
        else:
            normal_cells = 256
        checks['approved_parameter_changes_only' if common_case else 'normal_only_parameter_change'] = set(refinement_scope) <= allowed
        checks['realized_normal_mesh' if common_case else 'realized_32x256_mesh'] = len(xs)==33 and len(ys)==normal_cells+1
        old_surface = read(baseline,'surface',0)
        old_segments = read(baseline,'segments',0)
        if common_case and common_case.group(2):
            elements = int(common_case.group(2))
            expected_x = np.linspace(old_surface['x'][0],old_surface['x'][-1],elements+1)
            checks['approved_fault_resolution'] = (len(surface)==elements+1
                and len(segments)==elements
                and np.allclose(surface['x'],expected_x,rtol=0,atol=1e-12)
                and np.allclose(surface['y'],np.interp(expected_x,old_surface['x'],old_surface['y']),
                                rtol=0,atol=1e-12))
            # This fixed-profile fixture has constant prescribed support. A
            # finer fault partition must not change that physical strip.
            checks['unchanged_support_width'] = all(
                np.ptp(old_segments[key])<1e-12
                and np.allclose(segments[key],old_segments[key][0],rtol=0,atol=1e-12)
                for key in ('half_width_minus','half_width_plus'))
        else:
            checks['same_fault_discretization'] = len(surface)==len(old_surface) and all(
                np.allclose(surface[key],old_surface[key],rtol=0,atol=1e-12) for key in ('x','y'))
            checks['unchanged_support_width'] = len(segments)==len(old_segments) and all(
                np.allclose(segments[key],old_segments[key],rtol=0,atol=1e-12)
                for key in ('half_width_minus','half_width_plus'))
    values = np.zeros((len(xs), len(ys)))
    values[np.searchsorted(xs, phase['x']), np.searchsorted(ys, phase['y'])] = phase['phi']
    interpolator = RegularGridInterpolator((xs, ys), values, bounds_error=True)
    h = lambda phi: 128*np.maximum(phi,0)*(1+np.maximum(phi,0))/(1-np.maximum(phi,0))**2
    xi, weights = np.polynomial.legendre.leggauss(12)
    profiles = []
    # Resolve the saved Q1 profile independently, including surface endpoints
    # and every actual bulk quadrature column. This benchmark remains horizontal.
    for x in np.unique(np.r_[bulk['x'],surface['x']]):
        segment = np.clip(np.searchsorted(surface['x'],x,side='right')-1,0,len(segments)-1)
        center = np.interp(x,surface['x'],surface['y'])
        lower = center-segments['half_width_minus'][segment]
        upper = center+segments['half_width_plus'][segment]
        split = np.unique(np.r_[ys,lower,upper])
        qy = split[:-1,None]+np.diff(split)[:,None]*(1+xi)/2
        q = np.c_[np.full(qy.size,x),qy.ravel()]
        integrals = np.diff(split)[:,None]*weights*h(interpolator(q).reshape(qy.shape))/2
        full = float(np.sum(integrals))
        omitted = float(np.sum(integrals*((qy<lower)|(qy>upper)))/full)
        profiles.append((x,full,omitted))
    profiles = np.asarray(profiles)
    np.savetxt(path/f'normal_profiles_{step}.csv',profiles,delimiter=',',
               header='x,independent_full_Ih,omitted_fraction',comments='')
    checks['containment'] = bool(np.max(profiles[:,2]) <= 1e-4)
    columns = []
    for x in np.unique(bulk['x']):
        column = bulk[bulk['x']==x]
        # Divide actual 2-D assembler JxW by its tangential weight (box height
        # is one). Include zeros outside support, not a fitted/renormalized tail.
        normal_weights = column['weight']/np.sum(column['weight'])
        instantaneous = float(np.dot(normal_weights,column['chi']*column['V']))
        history = float(np.dot(normal_weights,column['history']))
        V = float(np.interp(x,surface['x'],surface['V']))
        columns.append((x,V,instantaneous,history,instantaneous+history,
                        abs(instantaneous+history-V)/max(abs(V),1e-5)))
    columns = np.asarray(columns)
    np.savetxt(path/f'crack_integrals_{step}.csv',columns,delimiter=',',
               header='x,V,instantaneous,history,total,normalization_error',comments='')
    checks['supported_normalization'] = bool(np.max(columns[:,-1]) <= 1e-4)

    # Group H by the stable initial particle row, not exact current y after
    # tiny transverse motion. Keep endpoint/seam samples in all ranges.
    particles = np.sort(particles,order='id')
    initial_particles = np.sort(initial_particles,order='id')
    checks['stable_particle_ids'] = np.array_equal(particles['id'],initial_particles['id'])
    row_H = []
    for y in np.unique(initial_particles['y']):
        row = particles[initial_particles['y']==y]
        row_H.append((y,np.min(row['H']),np.max(row['H']),np.mean(row['H'])))
    np.savetxt(path/f'H_rows_{step}.csv',row_H,delimiter=',',header='initial_y,min_H,max_H,mean_H',comments='')
    spread = dict(H=float(max(row[2]-row[1] for row in row_H)),
                  phi=float(np.max(np.ptp(values,axis=0))),
                  Ih=float(np.ptp(surface['Ih'])), C=float(np.ptp(surface['C'])), V=float(np.ptp(surface['V'])))
    # "Comparable" means reaching the predeclared reference signal, not a
    # threshold fitted to this run. Apply it conservatively to the whole fault.
    signal = dict(H=3.935554978786889,phi=.000953568418833567,Ih=.151855077492115,
                  C=28.9011223390664,V=6.25189921831375e-5)
    if common_case:
        # The same comparability rule uses the PRE-run reference response of
        # this common timestep sequence, not the larger two-step smoke signal.
        preflight = json.loads((path.parent/'timestep-support/dt0375/report.json').read_text())
        diagnostics = preflight['diagnostics']
        signal = dict(H=max(r['H_cumulative_increment_max_Pa'] for r in diagnostics),
                      phi=max(r['phi_cumulative_increment_max'] for r in diagnostics),
                      Ih=max(abs(r['Ih_cumulative_increment']) for r in diagnostics),
                      C=max(abs(r['C']-diagnostics[0]['C']) for r in diagnostics),
                      V=max(abs(r['V']-diagnostics[0]['V']) for r in diagnostics))
    checks['homogeneity_including_seam'] = all(spread[k] < signal[k] for k in spread)
    wraps = 0
    if step:
        previous = np.sort(read(path,'particles',step-1),order='id')
        wraps = int(np.count_nonzero(abs(previous['x']-particles['x'])>.125))
        entry = np.genfromtxt(path/f'phase_probe_{step}.csv',delimiter=',',names=True)
        exit_data = np.genfromtxt(path/f'phase_probe_exit_{step}.csv',delimiter=',',names=True)
        checks['phase_probe_restore'] = bool(entry['forced_failure_restored']==1 and entry['stable_ids']==1)
        if 'absolute_target' in exit_data.dtype.names:
            # Independently verify the authorized mixed target from the entry
            # probe; retain the original relative check for old saved evidence.
            target=max(1e-8*float(entry['R_old_phi_Hprevious']),float(entry['roundoff']))
            checks['phase_converged'] = bool(exit_data['absolute_target']==target
                and exit_data['R_new_phi_Hprevious']<=target)
        else:
            checks['phase_converged'] = bool(exit_data['relative'] <= 1e-8)
    report = dict(step=step,checks={k:bool(v) for k,v in checks.items()},
                  refinement_parameter_differences=refinement_scope,
                  passed=bool(all(checks.values())),max_omitted_fraction=float(np.max(profiles[:,2])),
                  max_supported_normalization_error=float(np.max(columns[:,-1])),
                  signed_integral_ranges={k:[float(np.min(columns[:,i])),float(np.max(columns[:,i]))]
                                          for i,k in ((2,'instantaneous'),(3,'history'),(4,'total'))},
                  phi_range=[float(np.min(values)),float(np.max(values))],
                  along_fault_ranges=spread,reference_signal_scales=signal,periodic_crossings=wraps,
                  elapsed_seconds=time.monotonic()-start)
    (path/f'guard_{step}.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--step',type=int,required=True)
    args=parser.parse_args()
    result=check(args.directory,args.step)
    print('K3 state gate:',json.dumps(result),flush=True)
    raise SystemExit(0 if result['passed'] else 1)
