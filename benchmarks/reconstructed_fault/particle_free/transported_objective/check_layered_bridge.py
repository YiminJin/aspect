#!/usr/bin/env python3
"""Single corrected homogeneous bridge, compared only as a constitutive change."""
import json
import numpy as np
from assess_layered_campaign import OUT
import assess_exterior_stress as exterior
from compare_bounded_profile import load
from assess_sheared_moving_fault import compare
from assess_corner_isolation import regional
exterior.OUT=OUT
new='exterior-compatible0125-bulk';old='exterior-compatible0125'
exterior.reduce_one(new)
a=json.loads((OUT/(new+'.json')).read_text())
b=json.loads((OUT.parent/'exterior_stress'/(old+'.json')).read_text())
_,rowsa,_,sa=load(new);_,rowsb,_,sb=load(old)
assert np.array_equal(sa[0].coeff,sb[0].coeff) and np.array_equal(sa[0].surface,sb[0].surface)
last_a=sa[max(sa)];last_b=sb[max(sb)]
result=dict(new=new,old=old,initial_bitwise_equal=True,law_comparison_not_refinement=True,
 final=compare(last_a,last_b,a['endpoint_cumulative_slip'],b['endpoint_cumulative_slip']),regional=regional(last_a,last_b),
 corrected_peaks=a['stress_evolution'],old_peaks=b['stress_evolution'])
(OUT/'homogeneous-bridge.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(initial_equal=True,regional=result['regional'])),flush=True)
