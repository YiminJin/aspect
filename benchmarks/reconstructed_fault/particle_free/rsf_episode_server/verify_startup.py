#!/usr/bin/env python3
"""Strict same-state/native-restart diff reporter. It never relaxes native checks."""
from pathlib import Path
import argparse,csv,json,numpy as np
H=['jump_m','damage','theta_normalized','reference_m','relaxed_jump_m','absolute_slip_m']
def index(path):
 rows=list(csv.DictReader((path/'cohesive-index.csv').open()))
 for r in rows:assert int(r['step'])==int(r['publications'])
 return {int(r['step']):path/Path(r['stem']).name for r in rows}
def compare(a,b):
 aa=list(csv.DictReader(Path(str(a)+'-surface.csv').open()));bb=list(csv.DictReader(Path(str(b)+'-surface.csv').open()))
 assert [r['id'] for r in aa]==[r['id'] for r in bb]
 err=max(abs(float(x[k])-float(y[k])) for x,y in zip(aa,bb) for k in H)
 result=dict(surface_history_max_abs=err,geometry_exact=Path(str(a)+'-geometry.csv').read_bytes()==Path(str(b)+'-geometry.csv').read_bytes())
 def records(stem):
  r=list(csv.reader(Path(str(stem)+'-physical-history.csv').open()));return [x[0] for x in r],np.array([x[1:] for x in r],float)
 ia,x=records(a);ib,y=records(b);assert ia==ib and np.isfinite(x).all() and np.isfinite(y).all()
 result['bulk_history_max_abs']=float(abs(x-y).max());result['persistent_records_exact']=bool(np.array_equal(x,y) and err==0. and result['geometry_exact'])
 def fields(stem):
  r=np.concatenate([np.genfromtxt(f,delimiter=',',names=True,dtype=None,encoding='utf8') for f in stem.parent.glob(stem.name+'-bulk-r*.csv')]);return np.sort(r,order='cell')
 x,y=fields(a),fields(b);assert np.array_equal(x['cell'],y['cell'])
 result['reconstructed_stress_max_abs_Pa']=max(float(max(abs(x[k]-y[k]))) for k in ['Txx_Pa','Tyy_Pa','Tzz_Pa','Txy_Pa'])
 return result
p=argparse.ArgumentParser(__doc__);p.add_argument('reference',type=Path);p.add_argument('candidate',type=Path);p.add_argument('--output',type=Path,default=Path('startup-verification.json'));args=p.parse_args()
a,b=index(args.reference),index(args.candidate);common=sorted(a.keys()&b.keys());assert common,'No common accepted states'
rows=[dict(step=k,**compare(a[k],b[k])) for k in common];args.output.write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
raise SystemExit(0 if all(r['persistent_records_exact'] for r in rows) else 1)
