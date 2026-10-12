#!/usr/bin/env python3
"""The affected full-support P1 gate; retains all previous numerical allowances."""
import json,re,sys
from pathlib import Path
import check_integration as c
serial,mpi=sys.argv[1:]
a=c.assess(serial,'straight');b=c.assess(mpi,'straight')
assert all('restart_error_'+str(k) in x for x in (a,b) for k in (3,4))
errors={str(i):c.compare(c.nodes(c.build/('output-'+serial),i),c.nodes(c.build/('output-'+mpi),i),('phi','old','older'),4.922e-12) for i in range(5)}
prior={str(i):c.compare(c.nodes(c.build/('output-'+serial),i),c.nodes(c.build/'output-straight_contact_fixed',i),('phi','old','older'),0) for i in range(5)}
def initial_H(label):
 rows=[r for f in (c.build/('output-'+label)).glob('initial-H-*.csv') for r in c.rows(f)]
 assert len(rows)==9216 and all(r['H']==r['native_H'] for r in rows)
 return {(r['x'],r['y']):r['H'] for r in rows}
assert initial_H(serial)==initial_H(mpi)
support={}
for label in (serial,mpi):
 lines=(c.here/'integration_evidence'/(label+'.log')).read_text().splitlines()
 data=[{k:float(v) for k,v in re.findall(r'(\w+)=([\deE+.-]+)',line)} for line in lines if line.startswith('P1 support:')]
 assert len(data)==5
 for r in data:
  assert r['full_cross_section']==1 and r['omitted_source']==0 and r['omitted_fraction']==0
  assert r['missed_strip_qp_phi']==0 and r['missed_strip_particle_phi']==0
 support[label]=data
print(json.dumps(dict(serial=a,two_ranks=b,MPI_phase_history_error=errors,prior_narrow_phase_history_error=prior,native_H_preserved=True,MPI_initial_H_identical=True,support=support,full_support_gate=True),indent=2))
