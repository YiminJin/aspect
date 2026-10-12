#!/usr/bin/env python3
"""Two unchanged normalization/frozen-field fixtures with fresh output paths."""
import hashlib,json,os,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parents[3]
build=root/'build-propagation-integration'
evidence=Path(__file__).resolve().parent/'normalization_evidence'
env=os.environ.copy();env.update(ASPECT_SOURCE_DIR=str(root),OMP_NUM_THREADS='1',DEAL_II_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
for case in ['mature','ih_budget']:
 label='normalization_compat_'+case
 prm=build/(label+'.prm');log=evidence/(label+'.log')
 assert not log.exists() and not (build/('output-'+label)).exists()
 source=build/('compat_'+case+'.prm')
 prm.write_text(source.read_text().replace('output-compat_'+case,'output-'+label))
 command=[str(build/'aspect'),str(prm)]
 hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [prm,build/'aspect']}
 start=time.monotonic()
 with log.open('w') as f:
  try:code=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=120).returncode
  except subprocess.TimeoutExpired:code=124
 result=dict(command=command,code=code,seconds=time.monotonic()-start,hashes=hashes)
 log.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
 print(case,code,result['seconds'],flush=True)
 assert code==0
