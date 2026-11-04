#!/usr/bin/env python3
"""Bounded coupled material fixture; never imports a production checkpoint."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[4]
BUILD=ROOT/'build-particle-free/transported-objective/stage-c'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('label');p.add_argument('--speed',type=float,default=.4)
    p.add_argument('--dt',type=float,default=.1);p.add_argument('--end',type=float,default=.4)
    p.add_argument('--initial-dt',type=float,default=.1,help='Same initialization interval across temporal controls')
    p.add_argument('--schedule',help='Explicit time function for a matched rejection control')
    p.add_argument('--level',type=int,choices=[3,4,5,6],default=4)
    p.add_argument('--profile-method',choices=['Q1 remap','bounded MacCormack'],default='Q1 remap')
    p.add_argument('--representation',choices=['separated','direct physical stress'],default='separated')
    p.add_argument('--inflow',choices=['zero','accepted trace'],default='zero')
    p.add_argument('--shear',type=float,default=.01)
    p.add_argument('--support',choices=['stationary profile','full normal cross-section'],default='stationary profile')
    p.add_argument('--spacing',type=float,default=.1,help='Initial material-label spacing; no within-run resampling')
    p.add_argument('--eta',default='1');p.add_argument('--reject',action='store_true')
    p.add_argument('--stage-energy',action='store_true',help='Log direct advection/rotation/source projection energies')
    p.add_argument('--true-checks',action='store_true',help='Nonpublishing direct source/derivative checks at initialization')
    p.add_argument('--audit',action='store_true',help='Existing native residual/directional audit')
    p.add_argument('--isolation',action='store_true',help='Nonpublishing prescribed DG boundary/energy checks at initialization')
    p.add_argument('--energy',action='store_true',help='Separate checker additionally exports physical pressure for work accounting')
    p.add_argument('--entry-control',action='store_true',help='Fresh default fixture with the preserved incoming core/checker pair')
    p.add_argument('--phase',default='.35*exp(-((y-.3*x)/.22)^2)*(1+.2*cos(2*x))')
    p.add_argument('--ranks',type=int,choices=[1,2,4],default=2)
    p.add_argument('--restart',type=Path);p.add_argument('--seconds',type=int,default=180)
    a=p.parse_args();assert re.fullmatch(r'[A-Za-z0-9_-]+',a.label)
    a.requested_end=a.end
    # Match the representable clock reached by the requested short uniform
    # schedule. Avoid a spurious terminal interval of a few ulps, without
    # extending the requested end or changing any physical solver tolerance.
    if a.dt>0 and a.schedule is None and not a.reject and not a.restart:
        count=round(a.end/a.dt)
        if 0<count<=4096 and abs(count*a.dt-a.end)<=8*math.ulp(a.end):
            clock=0.
            for _ in range(count):clock+=a.dt
            a.end=min(a.end,clock)
    lock=(BUILD/'simulation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    out=BUILD/'material-runs'/a.label;out.mkdir(parents=True,exist_ok=False)
    fault=out/'fault.txt';fault.write_text('-.8 -.24 .4\n.8 .24 .4\n')
    binary=ROOT/'build-particle-free/native/aspect-release'
    plugin=BUILD/('checks/libtransported_isolation_native.release.so' if a.isolation else 'checks/libtransported_material_native.release.so')
    if a.energy:plugin=BUILD/'checks/libtransported_material_energy.release.so'
    if a.entry_control:
        assert a.profile_method=='Q1 remap' and a.inflow=='zero' and not a.restart and not a.energy and not a.isolation
        binary=BUILD/'controls/profile-entry-aspect-release'
        plugin=BUILD/'controls/profile-entry-libtransported_material_native.release.so'
    restart='false'
    if a.restart:
        source=a.restart.resolve();assert source.is_relative_to((BUILD/'material-runs').resolve())
        assert (source/'output/restart').is_dir()
        shutil.copytree(source/'output/restart',out/'output/restart')
        restart='true'
    text=f'''set Dimension = 2
set Additional shared libraries = {plugin}
set Output directory = {out/'output'}
set Use years instead of seconds = false
set End time = {a.end}
set Resume computation = {restart}
set Nonlinear solver scheme = single Advection, iterated Newton Stokes
set Nonlinear solver tolerance = 1e-8
set Max nonlinear iterations = 30
set Nonlinear solver failure strategy = abort program
set Pressure normalization = no
set Surface pressure = 0
set Maximum time step = {a.dt}
set Maximum first time step = {a.dt}
subsection Formulation
 set Enable phase field = true
 set Reconstruct faults from phase field = true
end
subsection Geometry model
 set Model name = box
 subsection Box
  set X extent = 4
  set Y extent = 4
  set Box origin X coordinate = -2
  set Box origin Y coordinate = -2
 end
end
subsection Mesh refinement
 set Initial global refinement = {a.level}
 set Initial adaptive refinement = 0
 set Time steps between mesh refinement = 0
end
subsection Boundary velocity model
 set Prescribed velocity boundary indicators = left:function, right:function, bottom:function
 subsection Function
  set Function expression = {a.speed}+({a.shear})*y; 0
 end
end
subsection Gravity model
 set Model name = vertical
 subsection Vertical
  set Magnitude = 0
 end
end
subsection Initial temperature model
 set Model name = function
 subsection Function
  set Function expression = 293
 end
end
subsection Temperature field
 set Temperature method = static
end
subsection Compositional fields
 set Number of fields = 3
 set Names of fields = tau_xx, tau_yy, tau_xy
 set Types of fields = stress, stress, stress
 set Compositional field methods = static, static, static
end
subsection Initial composition model
 set Model name = function
 subsection Function
  set Function expression = 0;0;0
 end
end
subsection Material model
 set Model name = phase field fault
 subsection Phase field fault
  set Evolve phase field = false
  set Reference viscosities = 10
  set Minimum viscosity = 1e-8
  set Maximum viscosity = 1e6
  set Elastic shear moduli = 10
  set Cohesions = 1
  set Critical energy release rates = 1
  set Initial time step = {a.initial_dt}
  set Phase field activation threshold = 0.1
  set Friction law = rate state
  set Reference friction coefficients = 0.6
  set Direct effect parameters = 0.025
  set Evolution effect parameters = 0.015
  set Characteristic slip distance = 0.008
  set Reference slip rate = 0.001
  set Minimum slip rate = 1e-20
  set Radiation damping coefficients = 1
  set Use regularized formulation = true
  set Use adiabatic pressure in fault friction = false
  set I h quadrature tolerance = 1e-10
  set I h tail tolerance = 1e-10
  set I h integration backend = cell intervals
  set I h surface quadrature subdivisions = 8
  set Fault constitutive mode = mature frictional
  set Bulk history representation = DG trial stress
  set Enable prescribed DG accepted history = true
  set Use homogeneous mesh data = true
  set Stress rotation = Jaumann DG material Lie
  set Material profile transport = {a.profile_method}
  set Material tensor inflow = {a.inflow}
{("  set Material stress representation = "+a.representation) if a.representation!="separated" else ""}
  set FE bulk normal quadrature points = 8
  subsection Material initial phase
   set Function expression = {a.phase}
  end
  subsection Material viscosity multiplier
   set Function expression = {a.eta}
  end
 end
end
subsection Phase field model
 set Length scale = .15
 set Geometric function type = AT1
 set Degradation curvature parameter = 1
end
subsection Fault reconstruction
 set Association support = {a.support}
 set Structural point spacing = {a.spacing}
 set Fit prescribed geometry to phase field = false
 set Prescribed faults file = {fault}
 set Boundary completion = legacy
end
subsection Solver parameters
 subsection Stokes solver parameters
  set Stokes solver type = block AMG
  set Linear solver tolerance = 1e-9
 end
end
subsection Time stepping
 set List of model names = function{', material transaction rejection' if a.reject else ''}
 subsection Function
  set Variable names = time
  set Function expression = {a.schedule or str(a.dt)}
 end
end
subsection Particles
 set Particle generator name = reference cell
 set Interpolation scheme = cell average
 set Integration scheme = material diagnostic samples
 set List of particle properties = crack driving force, maxwell stress
 set Generate particle domains = false
 set Generate CPDI data for particle domains = false
 set Generate face data for particle domains = false
 set Load balancing strategy = none
 subsection Generator
  subsection Reference cell
   set Number of particles per cell per direction = 2
  end
 end
end
subsection Postprocess
 set List of postprocessors = particles, material transaction checks
 subsection Particles
  set Data output format = none
 end
end
subsection Checkpointing
 set Steps between checkpoint = 2
 set Time between checkpoint = 0
end
subsection Termination criteria
 set Termination criteria = end time
 set Checkpoint on termination = true
end
'''
    if a.entry_control:
        text=text.replace(f'  set Material profile transport = {a.profile_method}\n','').replace(f'  set Material tensor inflow = {a.inflow}\n','')
    path=out/'input.prm';path.write_text(text)
    env=os.environ.copy();env.update(LD_LIBRARY_PATH='/opt/openmpi/5.0.6/lib',OMP_NUM_THREADS='1',DEAL_II_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMP_PROC_BIND='false',OMPI_MCA_mpi_yield_when_idle='1')
    if a.audit:env['ASPECT_K1_FLOOR_AUDIT']='1'
    if a.true_checks:env['ASPECT_TRUE_STRESS_CHECKS']='1'
    if a.stage_energy:env['ASPECT_TRUE_STRESS_ENERGY']='1'
    command=['/opt/openmpi/5.0.6/bin/mpirun','-n',str(a.ranks),str(binary),str(path)]
    sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    (out/'runner.py').write_bytes(Path(__file__).read_bytes())
    record=dict(arguments=vars(a)|{'restart':str(a.restart)},command=command,core_sha256=sha(binary),plugin_sha256=sha(plugin),input_sha256=sha(path),runner_sha256=sha(out/'runner.py'),head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),runtime_environment={k:v for k,v in env.items() if k.startswith('ASPECT_') or k in ['LD_LIBRARY_PATH','OMP_NUM_THREADS','DEAL_II_NUM_THREADS','OPENBLAS_NUM_THREADS']})
    start=time.perf_counter()
    with (out/'run.log').open('w') as log:
        run=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=run.wait(timeout=a.seconds)
        except subprocess.TimeoutExpired:
            os.killpg(run.pid,signal.SIGTERM)
            try:run.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(run.pid,signal.SIGKILL);run.wait()
            code=124
    record.update(exit_code=code,wall_s=time.perf_counter()-start)
    (out/'command.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(label=a.label,exit_code=code,wall_s=record['wall_s'])))
    raise SystemExit(code)

if __name__=='__main__':main()
