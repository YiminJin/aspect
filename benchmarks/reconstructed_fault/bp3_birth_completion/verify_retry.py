from pathlib import Path
import numpy as np,json
def population(folder,step):
 a=np.concatenate([np.atleast_1d(np.genfromtxt(p,delimiter=',',names=True))
                   for p in sorted(folder.glob(f'birth_identity_{step}_rank*.csv'))])
 assert len(np.unique(a['id']))==len(a)
 return a[np.argsort(a['id'])]
r=Path(__file__).resolve().parent;checks=[]
for ranks,source in [(1,'transport-A-serial'),(2,'transport-A-final')]:
 a=r/('output-'+source);b=r/f'output-transport-retry-{ranks}'
 log=(r/f'evidence/transport-retry-{ranks}-np{ranks}.log').read_text(errors='replace')
 checks.append(dict(check=f'{ranks}/native-retry-marker',passed='REUSED_ID_NATIVE_RETRY_PASS' in log))
 for step in (3,4):
  checks.append(dict(check=f'{ranks}/{step}/retry-particles-H-stress-births-exact',passed=np.array_equal(population(a,step),population(b,step))))
(r/'results/retry_reuse_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print(sum(c['passed'] for c in checks),'/',len(checks),'reuse retry checks')
assert all(c['passed'] for c in checks)
