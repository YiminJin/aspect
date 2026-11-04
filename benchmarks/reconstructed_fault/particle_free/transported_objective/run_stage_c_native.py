#!/usr/bin/env python3
"""One bounded native Stage C fixture, never a production continuation."""
import argparse
import fcntl
import hashlib
import json
import os
import signal
import re
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[4]
BUILD=ROOT/'build-particle-free/transported-objective/stage-c'
RUNNER_SOURCE=Path(__file__).read_bytes()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('label');p.add_argument('--legacy',action='store_true');p.add_argument('--baseline',action='store_true')
    p.add_argument('--ranks',type=int,choices=[1,2,4],default=2);p.add_argument('--steps',type=int,default=3)
    p.add_argument('--restart',type=Path);p.add_argument('--reject',action='store_true');p.add_argument('--audit',action='store_true')
    p.add_argument('--schedule',default='200000');p.add_argument('--seconds',type=int,default=240)
    p.add_argument('--transport-check',action='store_true');p.add_argument('--refinement',type=int,choices=[3,4,5],default=4)
    p.add_argument('--layered-check',action='store_true')
    p.add_argument('--transfer-check',action='store_true')
    p.add_argument('--spin',type=float,default=0.,help='cohesive rigid boundary spin in 1/s; fixed-geometry reaction control only')
    p.add_argument('--cohesive-history',action='store_true');p.add_argument('--history-affine',action='store_true')
    p.add_argument('--history-level',type=int,choices=[0,1,2],default=0)
    p.add_argument('--cohesive',action='store_true',help='fresh bounded cohesive seed; never a production restart')
    p.add_argument('--cohesive-drive',type=float,default=0.,help='additional prescribed initial shear in Pa for the fresh cohesive fixture')
    p.add_argument('--cohesive-dt',type=float,default=.000125)
    p.add_argument('--tip-airy',type=float,default=0.,help='amplitude in Pa of a localized, analytically specified Airy stress fixture')
    p.add_argument('--signed-trace',action='store_true',help='Record unchanged signed Newton/backtrack diagnostics')
    p.add_argument('--probe-old-cohesive-import',action='store_true')
    args=p.parse_args()
    assert re.fullmatch(r'[A-Za-z0-9_-]+',args.label),'Use a single local fixture label, not a path.'
    # Several requested cases may be queued, but native simulations are serial.
    BUILD.mkdir(parents=True,exist_ok=True)
    simulation_lock=(BUILD/'simulation.lock').open('a')
    fcntl.flock(simulation_lock,fcntl.LOCK_EX)
    out=BUILD/'runs'/args.label;out.mkdir(parents=True,exist_ok=False)
    # Already resolved bounded input, read-only: retain its scientifically chosen
    # geometry/material/loading; override library/output/history and local bounds.
    template=ROOT/'build-particle-free/objective-stress/runs/legacy-current/input.prm'
    text=template.read_text().replace(', objective stress checks','')
    binary=ROOT/'build-particle-free/native/aspect-release'
    plugins=[BUILD/'bp3-plugin/libbp3_restore_150x50.release.so']
    if not args.legacy:plugins.append(BUILD/'checks/libtransported_objective_native.release.so')
    if args.baseline:
        assert args.legacy and not args.restart and not args.reject
        binary=BUILD/'controls/bp3-spin-control'
        plugins=[ROOT/'build-particle-free/objective-stress/bp3-plugin/libbp3_restore_150x50.release.so']
    text+=f'''
set Additional shared libraries = {', '.join(map(str,plugins))}
set Output directory = {out/'output'}
set Resume computation = false
subsection Material model
 subsection Phase field fault
  set Stress rotation = {'legacy' if args.legacy else 'Jaumann DG Lie'}
  set Bulk history representation = DG trial stress
  set Enable prescribed DG accepted history = true
  set Use homogeneous mesh data = true
  set FE bulk normal quadrature points = 8
 end
end
subsection Compositional fields
 set Compositional field methods = static, static, static, static, static
 set Mapped particle properties =
end
subsection Particles
 set Generate particle domains = false
 set Generate CPDI data for particle domains = false
 set Generate face data for particle domains = false
end
subsection Mesh refinement
 set Initial global refinement = {args.refinement}
end
subsection Postprocess
 set List of postprocessors = reconstructed fault BP3, BP3 output complete, BP3 restored monitor{', objective DG checks' if not args.legacy else ''}
 subsection BP3
  set Last accepted step = {args.steps}
  set Audit full state every step = false
 end
end
subsection Time stepping
 set List of model names = function{', objective DG rejection' if args.reject else ''}
 subsection Function
  set Function expression = {args.schedule}
 end
end
'''
    if args.cohesive_history:
        assert not (args.baseline or args.legacy or args.restart or args.cohesive)
        plugins=[BUILD/'cohesive-checks/libcohesive_rsf_native_checks.release.so']
        text=(ROOT/'benchmarks/reconstructed_fault/particle_free/cohesive_rsf_native/history.prm').read_text()
        text+=f'''\nset Additional shared libraries = {plugins[0]}
set Output directory = {out/'output'}
subsection Mesh refinement
 set Initial global refinement = {args.history_level}
end
'''
    elif args.cohesive:
        assert not (args.baseline or args.transport_check or args.transfer_check)
        plugins=[BUILD/'cohesive-checks/libcohesive_rsf_native_checks.release.so']
        text=f'''include benchmarks/reconstructed_fault/particle_free/cohesive_rsf_native/general.prm
set Additional shared libraries = {plugins[0]}
set Output directory = {out/'output'}
set End time = {args.steps*args.cohesive_dt}
set Maximum first time step = {args.cohesive_dt}
set Maximum time step = {args.cohesive_dt}
subsection Cohesive RSF pilot
 set Stress rotation = {'legacy' if args.legacy else 'Jaumann cut DGQ2 lagged Lie BE'}
 subsection Initial independent tensor
  set Function expression = 20*(x-6)^2/36;20*y^2/36;{args.cohesive_drive}-40*(x-6)*y/36
 end
end
subsection Time stepping
 set List of model names = function{', cohesive pilot rejection control' if args.reject else ''}
 subsection Function
  set Function expression = {args.cohesive_dt if args.schedule=='200000' else args.schedule}
 end
end
'''
    if args.tip_airy:
        assert args.cohesive
        u='((x-6)/.4)';v='(y/.4)';envelope=f'exp(-{u}^2-{v}^2)'
        xx=f'(2*{args.tip_airy}*{u}*{v}*(3-2*{v}^2)*{envelope})'
        yy=f'(2*{args.tip_airy}*{u}*{v}*(3-2*{u}^2)*{envelope})'
        xy=f'({args.tip_airy}*(1-2*{u}^2)*(1-2*{v}^2)*{envelope})'
        text+=f'''\nsubsection Cohesive RSF pilot
 subsection Initial independent tensor
  set Function expression = 20*(x-6)^2/36+{xx}-({xx}+{yy})/3;20*y^2/36+{yy}-({xx}+{yy})/3;{args.cohesive_drive}-40*(x-6)*y/36+{xy}
 end
end
'''
    if args.probe_old_cohesive_import:
        assert args.cohesive and not args.legacy and not args.restart
        text+='\nsubsection Cohesive RSF pilot\n set Initial state = checked prototype snapshot\n set Initial state file = intentionally-unread-old-trajectory.json\nend\n'
    if args.restart:
        assert args.restart.resolve().is_relative_to((BUILD/'runs').resolve()),'Only new Stage C local checkpoints may be resumed by this fixture.'
        shutil.copytree(args.restart.resolve(),out/'output/restart')
        text+='\nset Resume computation = true\n'
    if args.spin:
        assert args.cohesive,'BP3 validates its prescribed loading; rigid-spin boundary overrides are only supported by the cohesive reaction fixture.'
        text+=f'''\nsubsection Boundary velocity model
 set Prescribed velocity boundary indicators = left:function, right:function, bottom:function, top:function
 subsection Function
  set Function expression = -{args.spin}*y; {args.spin}*(x-6)
 end
end
'''
    if args.transport_check:
        text+=f'\nsubsection Mesh refinement\n set Minimum refinement level = {args.refinement}\nend\n'
    path=out/'input.prm';path.write_text(text)
    env=os.environ.copy();env.update(LD_LIBRARY_PATH='/opt/openmpi/5.0.6/lib',OMP_NUM_THREADS='1',DEAL_II_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMP_PROC_BIND='false',OMPI_MCA_mpi_yield_when_idle='1')
    if args.audit:env['ASPECT_K1_FLOOR_AUDIT']='1'
    if args.signed_trace:
        assert args.cohesive
        env['ASPECT_COHESIVE_SIGNED_TRACE']=str(out/'signed-trace.csv')
    if args.transport_check:env['ASPECT_OBJECTIVE_DG_TRANSPORT_CHECK']='1'
    if args.layered_check:
        assert args.transport_check
        env['ASPECT_OBJECTIVE_DG_LAYERED_CHECK']='1'
    if args.cohesive_history:
        env['ASPECT_COHESIVE_OBJECTIVE_CHECK']='1'
        if args.history_affine:env['ASPECT_COHESIVE_HISTORY_AFFINE']='1'
    if args.cohesive and args.reject:env['ASPECT_COHESIVE_FORCE_REJECT_STEP']='2'
    if args.cohesive:env['ASPECT_COHESIVE_WRITE_HISTORY_RECORDS']='1'
    if args.transfer_check:
        env['ASPECT_OBJECTIVE_DG_TRANSFER_CHECK']='1'
        text+='\nsubsection Mesh refinement\n set Strategy = BP3 fault support, objective DG transfer\nend\n'
        path.write_text(text)
    command=['/opt/openmpi/5.0.6/bin/mpirun','-n',str(args.ranks),str(binary),str(path)]
    if args.cohesive_history or args.cohesive:
        test_sources=[ROOT/'benchmarks/reconstructed_fault/particle_free/cohesive_rsf_native/checks.cc',
                      ROOT/'include/aspect/reconstructed_fault/cohesive_bulk_history.h']
        assert plugins[0].stat().st_mtime_ns>=max(f.stat().st_mtime_ns for f in test_sources),'Rebuild the cohesive checker before using this source revision.'
    elif not args.legacy:
        test_sources=[ROOT/'benchmarks/reconstructed_fault/particle_free/transported_objective/native/native_checks.cc',
                      ROOT/'include/aspect/material_model/phase_field_fault/dg_history_internal.h']
        assert plugins[-1].stat().st_mtime_ns>=max(f.stat().st_mtime_ns for f in test_sources),'Rebuild the mature DG checker before using this source revision.'
    sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    (out/'runner.py').write_bytes(RUNNER_SOURCE)
    record=dict(arguments=vars(args)|{'restart':str(args.restart)},command=command,runner_sha256=hashlib.sha256(RUNNER_SOURCE).hexdigest(),source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                core_sha256=sha(binary),plugin_sha256={str(f):sha(f) for f in plugins},input_sha256=sha(path),runtime_environment={k:v for k,v in env.items() if k in ['LD_LIBRARY_PATH','OMP_NUM_THREADS','DEAL_II_NUM_THREADS','OPENBLAS_NUM_THREADS','OMP_PROC_BIND'] or k.startswith('ASPECT_')})
    start=time.perf_counter()
    with (out/'run.log').open('w') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=process.wait(timeout=args.seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait()
            code=124
    record.update(exit_code=code,elapsed_s=time.perf_counter()-start)
    (out/'command.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(label=args.label,exit_code=code,elapsed_s=record['elapsed_s'])))
    raise SystemExit(code)


if __name__=='__main__':main()
