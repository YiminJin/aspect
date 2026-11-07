#!/usr/bin/env python3
"""Selected sensitivity comparisons; retain absolute norms and all material-label histories."""
import argparse,json
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import numpy as np
from assess_layered_campaign import OUT,compare_records,compare_final,load_case,state_at

def compare(a,b):
 A=json.loads((OUT/(a+'.json')).read_text());B=json.loads((OUT/(b+'.json')).read_text())
 shift=A['arguments']['translation']!=B['arguments']['translation']
 pa,ca,ra=load_case(a);pb,cb,rb=load_case(b)
 same_mesh=ca['arguments'].get('depth_repetitions',1)==cb['arguments'].get('depth_repetitions',1) and ca['arguments']['level']==cb['arguments']['level'] and ca['arguments']['halfwidth']==cb['arguments']['halfwidth']
 init={}
 if same_mesh and not shift:
  sa=state_at(pa,ca,0);sb=state_at(pb,cb,0)
  init=dict(bulk_Linf=float(np.max(abs(sa.coeff-sb.coeff))),surface_Linf=float(np.max(abs(sa.surface-sb.surface))))
  assert max(init.values())<1e-11,init
 history=compare_records(A,B,shift)
 final=compare_final(A,B,shift)
 # Orientation is modulo pi; weight only by the squared principal-stress gap.
 ra=A['records'][-1]['spin_diagnostics'];rb=B['records'][-1]['spin_diagnostics']
 aa=np.array(ra['principal_angle']);ab=np.array(rb['principal_angle']);gap=np.array(rb['principal_gap'])
 angle=.5*np.arctan2(np.sin(2*(aa-ab)),np.cos(2*(aa-ab)))
 orientation=dict(weighted_rms_rad=float(np.sqrt(np.sum(gap**2*angle**2)/np.sum(gap**2))),
                  max_gap=float(gap.max()),angles_rad=angle.tolist(),reference_gaps=gap.tolist())
 out=dict(a=a,b=b,initialization=init,translation_removed=shift,
          bulk_quadrature='8-point tensor rule on the union of both shifted Cartesian cell partitions',
          history=history,final=final,orientation=orientation)
 (OUT/('pair-'+a+'--'+b+'.json')).write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(dict(a=a,b=b,bulk=final['common_domain'],central=final['surface_common_quadrature']['central'],orientation=orientation['weighted_rms_rad'])),flush=True)
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('a');p.add_argument('b');a=p.parse_args();compare(a.a,a.b)
