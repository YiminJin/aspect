"""Independent scalar preflight of the selected startup heuristic, no ASPECT run."""
import json
import xml.etree.ElementTree as ET
import numpy as np
from run_small_startup import OUT, STUDY
from analyze_small_startup import state_change
from check_startup_30km import table

path=STUDY/'startup'
s=table(path/'state_work_0.csv')
arrays={a.attrib['Name']:np.fromstring(a.text,sep=' ') for a in ET.parse(
    path/'reconstructed_faults/reconstructed_faults-00000.vtu').findall('.//PointData/DataArray')}
f=np.clip(arrays['composition_strengthening'],0,1);ratio=.03/(.004*(1-f)+.04*f)

def measure(dt):return ratio*abs(np.log(state_change(s['V'],s['Theta_out'],dt)/s['Theta_out']))

rows=[]
for dt in [150.,300.,4e6]:
    e=measure(dt);i=np.argmax(e)
    rows.append(dict(dt=dt,measure=float(e[i]),limiting_node=int(i),xd=float(s['xd'][i])))
lower,upper=0.,4e6
for _ in range(60):
    middle=.5*(lower+upper)
    if max(measure(middle))<=.1:lower=middle
    else:upper=middle
# Independent analytic crossing of the same aging map. Exclude states whose
# equilibrium is inside the allowed interval (they impose no finite bound).
theta=s['Theta_out'];eq=.1/s['V'];target=theta*np.exp(np.sign(eq-theta)*.1/ratio)
crossing=((eq>theta)&(target<eq))|((eq<theta)&(target>eq))
exact=-.1/s['V'][crossing]*np.log1p((target[crossing]-theta[crossing])/(theta[crossing]-eq[crossing]))
assert abs(min(exact)/lower-1)<1e-11
assert rows[0]['measure']<.05 and rows[1]['measure']<.1 and rows[2]['measure']>30
report=dict(samples=rows,limiting_dt=lower,analytic_dt=float(min(exact)),passed=True)
(OUT/'predictor_preflight.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
