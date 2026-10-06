#!/usr/bin/env python3
"""Exact comparisons; elapsed times, output paths and VTK/gnuplot creation timestamps excluded."""
from pathlib import Path
import hashlib, json, re, struct, zlib, xml.etree.ElementTree as ET
r=Path(__file__).resolve().parent
checks={}
def check(name, value): checks[name]=bool(value)
def log(v,n): return (r/f'evidence/{v}-{n}.log').read_text(errors='replace')
def out(v,n): return r/f'output-{v}-{n}'
def normalize(s):
    return re.sub(r'output-(reference|candidate)-[a-z-]+', 'OUTPUT', s)
def decisions(s):
    markers=('Iteration ', 'Fault linear solve:', 'Relative nonlinear residual',
             'line search accepted after', 'Solving temperature system', 'Solving Stokes system',
             'Phase-field residual target:', 'Stage-J history resolution:',
             'Stage-J feedback step', 'line search exhausted', 'Errors u_L1')
    result=[]
    for line in s.splitlines():
        if any(m in line for m in markers):
            line=re.sub(r'\s+', ' ', line).strip()
            line=re.sub(r'\s+([,:])', r'\1', line)
            if 'Phase-field residual target:' in line:
                line=re.sub(r'(?<==)[0-9.eE+\-]+', lambda m:float(m[0]).hex(),line)
            result.append(line)
    return result

def history_fingerprint(directory):
    data=zlib.decompress((directory/'restart/01/resume.z').read_bytes()[16:])
    key=b'StageJRestart';assert data.count(key)==1
    start=data.index(key)+len(key);size=struct.unpack_from('<Q',data,start)[0]
    blob=data[start+8:start+8+size];assert len(blob)==size
    return blob

def timers(s):
    return [(m[1].strip(),int(m[2])) for m in re.finditer(r'^\| ([^|]+)\|\s+(\d+)\s+\|',s,re.M)]

names=['amg','bfbt','melt','gmg','particles','particles-resume','cache','empty-owner-qualified','singular','cohesive','cohesive-resume']
for n in names:
    for v in ['reference','candidate']:
        meta=json.loads((r/f'evidence/{v}-{n}.json').read_text())
        expected=1 if n in ['singular','cohesive','cohesive-resume'] else 0
        check(f'{v}/{n}/exit',meta['exit_code']==expected and meta['limit_reason'] is None)
    a,b=log('reference',n),log('candidate',n)
    check(n+'/decisions',decisions(a)==decisions(b))
    check(n+'/work-counters',timers(a)==timers(b))
    sa,sb=out('reference',n)/'statistics',out('candidate',n)/'statistics'
    if sa.exists():check(n+'/statistics',normalize(sa.read_text())==normalize(sb.read_text()))
    files=[p for p in out('reference',n).rglob('*') if p.suffix in ['.vtu','.gnuplot'] or p.name.startswith(('ordinary-','particle-cache-rank-'))]
    other=[p.relative_to(out('candidate',n)) for p in out('candidate',n).rglob('*') if p.suffix in ['.vtu','.gnuplot'] or p.name.startswith(('ordinary-','particle-cache-rank-'))]
    check(n+'/field-file-set',set(p.relative_to(out('reference',n)) for p in files)==set(other))
    for p in files:
        q=out('candidate',n)/p.relative_to(out('reference',n))
        if p.suffix=='.vtu':equal=ET.tostring(ET.parse(p).getroot())==ET.tostring(ET.parse(q).getroot())
        elif p.suffix=='.gnuplot':
            clean=lambda text:re.sub(r'^# (?:Date|Time) = .*$', '', text, flags=re.M)
            equal=clean(p.read_text())==clean(q.read_text())
        else:equal=p.read_bytes()==q.read_bytes()
        check(n+'/'+str(p.relative_to(out('reference',n))),equal)

for v in ['reference','candidate']:
    check(v+'/actual-gmg','Solving Stokes system (GMG)' in log(v,'gmg'))
    for n in ['amg','bfbt','melt']:
        check(v+'/'+n+'/actual-amg',('Solving Stokes system (AMG-BFBT)' if n=='bfbt' else 'Solving Stokes system (AMG)') in log(v,n))
    check(v+'/singular-marker','Verified Stage-F singular K_V factorization diagnostic.' in log(v,'singular'))
    for n in ['cache','empty-owner-qualified']:
        counts=[]
        for rank in range(2):
            text=(out(v,n)/f'particle-cache-rank-{rank}.txt').read_text()
            check(f'{v}/{n}/rank{rank}/rebuilds',text.startswith('cold rebuilds 1 warm rebuilds 0 regeneration rebuilds 1 local particles '))
            counts.append(int(text.splitlines()[0].split()[-1]))
        check(f'{v}/{n}/ownership',min(counts)==0 and max(counts)>0 if n=='empty-owner-qualified' else min(counts)>0)
    for n in ['cohesive','cohesive-resume']:
        text=log(v,n)
        check(f'{v}/{n}/known-failure','Newton line search exhausted all admissible candidates' in text and 'Nonlinear solver failed to converge' in text and 'Segmentation fault' not in text)
    resumed=log(v,'cohesive-resume')
    check(v+'/cohesive/resume-history-marker','Stage-J checkpoint histories, V, geometry, and bulk: verified' in resumed)
    full=log(v,'cohesive');check(v+'/cohesive/restart-decisions',decisions(full[full.index('*** Timestep 2:'):])==decisions(resumed[resumed.index('*** Timestep 2:'):]))
    for step in [4,5]:
        for rank in range(2):
            a=out(v,'particles')/f'ordinary-{step}-{rank}.txt';b=out(v,'particles-resume')/a.name
            check(f'{v}/ordinary/restart-step{step}-rank{rank}',a.read_bytes()==b.read_bytes() and b.stat().st_mtime_ns>a.stat().st_mtime_ns)
    check(v+'/ordinary/native-output',len(list((out(v,'particles')/'solution').glob('*.vtu')))==12 and len(list((out(v,'particles')/'particles').glob('*.gnuplot')))==12)
    check(v+'/ordinary/resume-executed',len(re.findall('Ordinary feature-disabled audit:.*verified',log(v,'particles-resume')))==2)
    # The following are rejected fixture constructions, not successful empty-owner probes.
    check(v+'/rejected-empty-ridge','has no phase-field support' in log(v,'empty-cache'))
    check(v+'/rejected-empty-constitutive','Generic fault-property interpolation did not reject its uninitialized' in log(v,'empty-fixed-cache'))

for name in ['mesh','mesh.info','mesh_fixed.data','mesh_variable.data']:
    check('cohesive/checkpoint/'+name,(out('reference','cohesive')/'restart/01'/name).read_bytes()==(out('candidate','cohesive')/'restart/01'/name).read_bytes())
check('cohesive/history-V-geometry-bulk-fingerprint',history_fingerprint(out('reference','cohesive'))==history_fingerprint(out('candidate','cohesive')))
# Load the immutable reference checkpoints with the candidate executable too.
for n in ['particles','cohesive']:
    cross=n+'-cross-resume'
    meta=json.loads((r/f'evidence/candidate-{cross}.json').read_text())
    check(cross+'/exit',meta['exit_code']==(0 if n=='particles' else 1) and meta['limit_reason'] is None)
    check(cross+'/decisions',decisions(log('reference',n+'-resume'))==decisions(log('candidate',cross)))
    if n=='particles':
        for step in [4,5]:
            for rank in range(2):
                a=out('reference',n)/f'ordinary-{step}-{rank}.txt';b=out('candidate',cross)/a.name
                check(f'{cross}/step{step}-rank{rank}',a.read_bytes()==b.read_bytes() and b.stat().st_mtime_ns>a.stat().st_mtime_ns)
        check(cross+'/fresh-audits',len(re.findall('Ordinary feature-disabled audit:.*verified',log('candidate',cross)))==2)
    else:
        check(cross+'/history-marker','Stage-J checkpoint histories, V, geometry, and bulk: verified' in log('candidate',cross))
        check(cross+'/known-nonconvergence','Newton line search exhausted all admissible candidates' in log('candidate',cross) and 'Nonlinear solver failed to converge' in log('candidate',cross))

(r/'evidence/comparisons.json').write_text(json.dumps(checks,indent=2)+'\n')
print(f'{sum(checks.values())}/{len(checks)} exact checks passed')
print('Failures:',[k for k,v in checks.items() if not v])
assert all(checks.values())
