#!/usr/bin/env python3
"""Bounded local stationary-DG qualification; sequential cases, no server submission."""
import argparse,hashlib,json,os,resource,shutil,signal,subprocess,sys,time,threading,re
from pathlib import Path
from run_pf6_mature import flatten
ROOT=Path(__file__).resolve().parents[3]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    if len(sys.argv)>1 and sys.argv[1]=='--rank':
        out=Path(sys.argv[2]);rank=os.environ['OMPI_COMM_WORLD_RANK'];start=time.monotonic()
        code=subprocess.run([sys.argv[3],str(out/'input.prm')]).returncode
        r=resource.getrusage(resource.RUSAGE_CHILDREN)
        (out/f'resource-rank-{rank}.json').write_text(json.dumps(dict(max_rss_KiB=r.ru_maxrss,seconds=time.monotonic()-start,exit_code=code),indent=2)+'\n')
        raise SystemExit(code)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('label');p.add_argument('--mode',choices=['reference','optimized'],default='optimized')
    p.add_argument('--mesh',choices=['small','larger'],default='small')
    p.add_argument('--ranks',type=int,choices=[1,2,4],default=2)
    p.add_argument('--steps',type=int,default=2);p.add_argument('--dt',type=float,default=200000)
    p.add_argument('--seconds',type=int,default=240)
    p.add_argument('--operators',action='store_true')
    p.add_argument('--adaptive',action='store_true',help='Bounded local control with the unchanged server adaptive caps/initial interval')
    p.add_argument('--vary-bulk-composition',action='store_true',help='Diagnostic continuous bulk chemical label with unchanged homogeneous material/friction values')
    p.add_argument('--schedule',help='Explicit matching accepted-timestep sequence for the cutback control')
    p.add_argument('--restart',type=Path)
    p.add_argument('--outer-rejection',action='store_true');p.add_argument('--nonlinear-rejection',action='store_true');p.add_argument('--candidate-rejection',action='store_true')
    a=p.parse_args();work=ROOT/'build-particle-free/bp3-optimization';out=work/'runs'/a.label;out.mkdir(parents=True,exist_ok=False)
    binary=ROOT/'build-particle-free/native/aspect-release'
    bp3=work/'plugins/native/libbp3_restore_150x50.release.so';observer=work/'plugins/libbp3_coupled_checks.release.so'
    text=flatten(ROOT/'benchmarks/reconstructed_fault/bp3_native/section7b-fe-perturbed.prm',ROOT)
    text+=f'''
set Additional shared libraries = {bp3}, {observer}
set Output directory = {out/'run'}
set End time = 2000000
set Maximum time step = {a.dt:.17g}
set Maximum first time step = {a.dt:.17g}
subsection Mesh refinement
 set Initial global refinement = {6 if a.mesh=='small' else 7}
end
subsection Fault reconstruction
 set Structural point spacing = 10
end
subsection Material model
 subsection Phase field fault
  set Bulk history representation = DG trial stress
  set FE bulk normal quadrature points = 6
  set Initial time step = 200000
  set Particle stress strain convention = three dimensional deviator
  set Use homogeneous mesh data = true
  set Enable prescribed DG accepted history = true
  set Optimize stationary mature DG = {'true' if a.mode=='optimized' else 'false'}
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
subsection Time stepping
 set List of model names = function
 subsection Function
  set Function expression = {a.dt:.17g}
 end
end
subsection Postprocess
 subsection BP3
  set Last accepted step = {a.steps}
  set Graceful wall seconds = 1000000000
  set Heavy output slip interval = 1
  set Heavy output time interval = 1000000000
  set Profile slip interval = 1
  set Profile time interval = 1000000000
  set Audit full state every step = true
 end
 subsection BP3 coupled checks
  set Mature lifecycle checks = true
  set Prescribed schedule check = false
  set Write authoritative stress = true
  set Write FE coefficients = true
  set Scheduled field output = true
 end
end
subsection Termination criteria
 set Checkpoint on termination = true
end
'''
    if a.restart:
        shutil.copytree(a.restart.resolve(),out/'run/restart');text+='\nset Resume computation = true\n'
    if a.outer_rejection:text+='\nsubsection Time stepping\n set List of model names = function, BP3 one outer rejection\nend\n'
    if a.nonlinear_rejection:text+='\nset Nonlinear solver failure strategy = cut timestep size\nsubsection Time stepping\n set List of model names = function, repeat on nonlinear solver failure\nend\nsubsection Postprocess\n subsection BP3 coupled checks\n  set Inject nonlinear failure = true\n end\nend\n'
    if a.candidate_rejection:text+='\nsubsection Postprocess\n subsection BP3 coupled checks\n  set Inject candidate failure = true\n end\nend\n'
    if a.schedule:text+=f'\nsubsection Time stepping\n subsection Function\n  set Variable names = time\n  set Function expression = {a.schedule}\n end\nend\n'
    if a.adaptive:
        assert not a.schedule
        text+='''
set End time = 47336400000
set Maximum time step = 4e7
set Maximum first time step = 4e6
set Maximum relative increase in time step = 91
subsection Material model
 subsection Phase field fault
  set Initial time step = 4e6
 end
end
subsection Time stepping
 set List of model names = convection time step, reconstructed fault time step, repeat on nonlinear solver failure
 subsection Reconstructed fault time step
  set Maximum logarithmic state change = 0.2
 end
 subsection Repeat on nonlinear solver failure
  set Cut back factor = 0.5
 end
end
'''
    if a.vary_bulk_composition:
        # The truncated fixture has a constant native chemical label. Exercise
        # mixed fractions without changing its homogeneous bulk or surface laws.
        text+='''
subsection Initial composition model
 set Model name = unspecified
 set List of model names = reconstructed fault BP3, function
 set List of model operators = add, subtract
 subsection Function
  set Function expression = 0; 0; 0; 0; 0.8-0.6*(x+1234.5)/2000
 end
end
'''
    (out/'input.prm').write_text(text)
    env=os.environ.copy();env.update(LD_LIBRARY_PATH='/opt/openmpi/5.0.6/lib',OMP_PROC_BIND='false',OMP_NUM_THREADS='1',DEAL_II_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMPI_MCA_mpi_yield_when_idle='1',ASPECT_FAULT_PERFORMANCE='1',ASPECT_FAULT_LINEAR_PERFORMANCE='1')
    env.pop('ASPECT_FAULT_COMPARE_COUPLING',None)
    if a.operators:env['ASPECT_FAULT_COMPARE_COUPLING']='1'
    command=['/opt/openmpi/5.0.6/bin/mpirun','--map-by','core','--bind-to','core','-n',str(a.ranks),sys.executable,str(Path(__file__).resolve()),'--rank',str(out),str(binary)]
    record=dict(arguments=vars(a)|{'restart':str(a.restart) if a.restart else None},command=command,head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),sha256={str(f):sha(f) for f in [binary,bp3,observer,out/'input.prm']},timeout_seconds=a.seconds)
    start=time.monotonic();events=[]
    with (out/'stdout.log').open('w') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True,text=True)
        def capture():
            for line in process.stdout:
                log.write(line)
                if re.match(r'\*\*\* Timestep \d+:',line) or line.startswith('Termination requested by criterion:') or line.startswith('*** Snapshot '):
                    events.append(dict(seconds=time.monotonic()-start,line=line.strip()))
        reader=threading.Thread(target=capture,daemon=True);reader.start()
        try:code=process.wait(timeout=a.seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            code=124
        reader.join(timeout=10)
    record.update(exit_code=code,seconds=time.monotonic()-start)
    record['timing_events']=events
    (out/'record.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2));raise SystemExit(code)

if __name__=='__main__':main()
