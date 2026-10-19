#!/usr/bin/env python3
"""Prepare/finalize the selected fixed-mesh P2 physical-response controls.

The recorded command launches the native integrator; this does not control
constitutive updates, phase equilibrium, time integration or acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from run_fixed_profile import fixture_text


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','finalize'])
    p.add_argument('label')
    p.add_argument('--case',choices=['unloaded','primary','half'],default='primary')
    a=p.parse_args()
    root=Path(__file__).resolve().parents[3]
    out=root/'build-particle-free/runs'/a.label
    if a.action=='finalize':
        r=json.loads((out/'record.json').read_text())
        log=root/'benchmarks/reconstructed_fault/refactoring_boundary/evidence'/a.label
        r.update(json.loads(log.with_suffix('.json').read_text()))
        shutil.copy2(log.with_suffix('.log'),out/'run.log')
        r['gate_passed']=r['exit_code']==0
        r['outputs']={str(f.relative_to(out/'run')):sha(f) for f in (out/'run').rglob('*.csv')}
        (out/'record.json').write_text(json.dumps(r,indent=2)+'\n')
        print(json.dumps({k:r[k] for k in ['exit_code','seconds','gate_passed']}))
        return
    out.mkdir(parents=True,exist_ok=False)
    divisor=256 if a.case=='half' else 128
    dt=1/divisor
    end=.015625 if a.case=='unloaded' else .59375
    budget={'unloaded':180,'primary':1200,'half':2400}[a.case]
    binary=root/'build-particle-free/evolving-phase/native/aspect-release'
    plugin=root/'build-particle-free/evolving-phase/plugin/libpf6a_checks.release.so'
    for f in [binary,plugin]:shutil.copy2(f,out/f.name)
    times=[.0078125,.25,.5,.515625,.53125,.546875,.5625,.5703125,.578125,.59375]
    if a.case=='unloaded':times=[.0078125,.015625]
    velocity='0; 0' if a.case=='unloaded' else f'if(t<=0.5625,0.256*max(0,t-{.5*dt})*(y-0.5),0); 0'
    text=f'''include $ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/particle_free/pf6a-common.prm
set End time = {end}
set Maximum first time step = {dt}
set Maximum time step = {dt}
subsection Geometry model
 subsection Box
  set Y repetitions = 256
 end
end
subsection Time stepping
 subsection Function
  set Function expression = {dt}
 end
end
subsection Material model
 subsection Phase field fault
  set Use homogeneous mesh data = true
  set FE bulk normal quadrature points = 12
 end
end
subsection Boundary velocity model
 subsection Function
  set Function expression = {velocity}
 end
end
subsection Compositional fields
 set Compositional field methods = static, static, static, static
 set Mapped particle properties =
end
subsection Particles
 set Integration scheme = rk2
 set Generate particle domains = false
 set Generate CPDI data for particle domains = false
 set Generate face data for particle domains = false
 set List of particle properties = crack driving force, maxwell stress
end
subsection Postprocess
 subsection PF6a checks
  set Operator and rejection checks = false
  set Batch diagnostic stress evaluation = true
  set Verify batched diagnostic stress = false
  set Full snapshot times = {', '.join(map(str,times))}
 end
end
'''
    (out/'fixture.prm').write_text(text)
    text=fixture_text(out/'fixture.prm',root,'PF6a cohesive endpoint')
    text+=f'\nset Additional shared libraries = {out/plugin.name}\nset Output directory = {out/"run"}\n'
    (out/'input.prm').write_text(text)
    command=['env','ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION=1','OMP_PROC_BIND=false',
        '/opt/openmpi/5.0.6/bin/mpirun','--map-by','core','--bind-to','core','--rank-by','slot','--report-bindings','-n','1',str(out/binary.name),str(out/'input.prm')]
    (out/'command.json').write_text(json.dumps(command,indent=2)+'\n')
    paths=subprocess.check_output(['git','ls-files','-c','-o','--exclude-standard','source','include'],cwd=root,text=True).splitlines()
    record=dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        budget_seconds=budget,threads=1,ranks=1,case=a.case,divisor=divisor,end_time=end,
        normal_cells=512,load_stop=.5625,quadrature=12,source_quadrature=3,
        executable_sha256=sha(out/binary.name),plugin_sha256=sha(out/plugin.name),
        input_sha256=sha(out/'input.prm'),source_hashes={name:sha(root/name) for name in sorted(set(paths)) if (root/name).is_file()},
        benchmark_sources={str(f.relative_to(root)):sha(f) for f in [Path(__file__).resolve(),Path(__file__).with_name('pf6a_checks.cc')]},
        selection_sha256=sha(root/'build-particle-free/p2-response/selection-before-runs.md'),
        physical_seed_sha256=sha(Path(__file__).with_name('pf6a-initial-phase.txt')),
        scope='Bounded fixed-mesh cohesive P2 physical response; unloaded or ramp/hold; native integrator and unchanged acceptance')
    (out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
    shutil.copy2(Path(__file__).with_name('pf6a_checks.cc'),out/'observer-source.cc')
    print(json.dumps(dict(label=a.label,budget_seconds=budget,command=command)))

if __name__=='__main__':main()
