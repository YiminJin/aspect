#!/usr/bin/env python3
"""Small fresh mature-frozen check, at default and explicitly enabled selectors."""
from pathlib import Path
import os,subprocess,json,hashlib,time
root=Path(__file__).resolve().parents[3];b=root/'build-propagation-integration';e=Path(__file__).parent/'irreversibility_evidence';records=[]
env={**os.environ,'ASPECT_SOURCE_DIR':str(root),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','DEAL_II_NUM_THREADS':'1'}
for enabled in [False,True]:
 label='irreversibility_mature_'+('enabled' if enabled else 'default');out=b/('output-'+label);log=e/(label+'.log');prm=b/(label+'.prm');assert not out.exists() and not log.exists()
 text=(b/'compat_mature.prm').read_text().replace('output-compat_mature','output-'+label)
 if enabled:text+='\nset Nonlinear solver failure strategy = abort program\nsubsection Solver parameters\n subsection Phase field solver parameters\n  set Use bound constraints = true\n  set Enforce phase irreversibility = true\n end\nend\n'
 prm.write_text(text);start=time.monotonic()
 with log.open('w') as f:code=subprocess.run([str(b/'aspect'),str(prm)],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=120).returncode
 assert code==0
 for old in (b/'output-compat_mature').glob('*.csv'):assert old.read_bytes()==(out/old.name).read_bytes(),old.name
 records.append(dict(label=label,enabled=enabled,code=code,seconds=time.monotonic()-start,identical_csvs=2,core_sha256=hashlib.sha256((b/'aspect').read_bytes()).hexdigest()))
(e/'mature-compatibility.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records,indent=2))
