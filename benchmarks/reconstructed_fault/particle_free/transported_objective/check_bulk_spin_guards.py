#!/usr/bin/env python3
"""Fail early for unselected particle/cohesive/growth/legacy combinations."""
from pathlib import Path
import os,subprocess,json,hashlib
ROOT=Path(__file__).resolve().parents[4];B=ROOT/'build-particle-free/transported-objective/stage-c'
BASE=B/'material-runs/bulk-cd3';OUT=B/'bulk-spin/guard-runs2'
if __name__=='__main__':
    OUT.mkdir(exist_ok=False);records=[]
    env=os.environ.copy();env.update(LD_LIBRARY_PATH='/opt/openmpi/5.0.6/lib',OMP_NUM_THREADS='1',DEAL_II_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    for name,changes in [
      ('particle',{'Bulk history representation':'particle stress','Enable prescribed DG accepted history':'false','Stress rotation':'Jaumann lagged Lie','Material stress representation':'separated'}),
      ('cohesive',{'Fault constitutive mode':'cohesive'}),
      ('growth',{'Evolve phase field':'true'}),
      ('legacy',{'Stress rotation':'legacy','Material stress representation':'separated'})]:
        folder=OUT/name;folder.mkdir()
        if name in ['particle','legacy']:changes.update({'Material profile transport':'Q1 remap','Material tensor inflow':'zero'})
        text=(BASE/'input.prm').read_text().replace(str(BASE/'output'),str(folder/'output'))
        text+='\nsubsection Material model\n subsection Phase field fault\n'+''.join('  set '+k+' = '+v+'\n' for k,v in changes.items())+' end\nend\n'
        path=folder/'input.prm';path.write_text(text)
        command=['/opt/openmpi/5.0.6/bin/mpirun','-n','1',str(B/'bulk-spin/aspect-release'),str(path)]
        with (folder/'run.log').open('w') as log:run=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=30)
        message='Bulk slip-subtracted spin requires growth-disabled mature accepted DG'
        output=(folder/'run.log').read_text(errors='replace')
        assert run.returncode!=0 and message in output,(name,output[-4000:])
        records.append(dict(case=name,command=command,exit_code=run.returncode,input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),expected_message=message))
        print(name,'refused',flush=True)
    (ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/bulk_spin/guards.json').write_text(json.dumps(records,indent=2)+'\n')
