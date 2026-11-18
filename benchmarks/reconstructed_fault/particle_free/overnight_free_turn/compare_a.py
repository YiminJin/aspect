#!/usr/bin/env python3
"""Reduce completed quarter/half-step fields from the same accepted parent."""
from observations import *
import csv
label='overnight-free-turn-a';coarse='recover-free-turn-positive-half';parent='first-turn-positive-fourcore-end'
quarter_labels=[l for l in [label,label+'-tail'] if (RUNS/l/'output/bulk-accepted.csv').exists()]
locations={int(r.step):l for l in quarter_labels for r in read(RUNS/l/'output/bulk-accepted.csv')}
initial={'label':'first-turn-positive-prefix','field':seed('first-turn-positive-prefix',1)}
old=field(parent,15)
parent_result,_=endpoint(parent,15,.03125,initial)
result=dict(parent=dict(label=parent,step=15,time=.075),comparisons=[],quarter_work=work(quarter_labels),half_work=work([coarse]))
for step,other,time,preva,prevb in [(17,16,.0775,(parent,15),(parent,15)),(19,17,.08,(label,17),(coarse,16))]:
 if step not in locations:continue
 label=locations[step]
 if step==19:preva=(locations[17],17)
 a,fa=endpoint(label,step,.03125,initial);b,fb=endpoint(coarse,other,.03125,initial)
 assert abs(a['time']-time)<1e-13 and abs(b['time']-time)<1e-13
 r=dict(time=time,quarter=a,half=b,phase_error=l2(fb,fa,old),
   quarter_window_fracture={k:v-parent_result['regional_fracture'][k] for k,v in a['regional_fracture'].items()},
   half_window_fracture={k:v-parent_result['regional_fracture'][k] for k,v in b['regional_fracture'].items()},
   section_errors=section_difference(label,step,coarse,other,preva,prevb,.0025))
 for key,source in [('quarter_work_to_common_time','quarter_work'),('half_work_to_common_time','half_work')]:
  errors=[v['error'] for v in result[source]['steps'] if v['time']<=time+1e-12]
  r[key]=dict(signed=float(sum(errors)),absolute=float(sum(abs(e) for e in errors)))
 common=[]
 for x,y in zip(a['centerlines'],b['centerlines']):
  assert x['x']==y['x']
  if 'center' in x and 'center' in y:common.append(dict(x=x['x'],quarter_center=x['center'],half_center=y['center'],quarter_width=x['width'],half_width=y['width'],center_difference=x['center']-y['center'],width_difference=x['width']-y['width']))
 r['common_centerlines']=common
 result['comparisons'].append(r)
 for side,data in [('quarter',a),('half',b)]:
  with (OUT/f'A-{time:.4f}-{side}-centerlines.csv').open('w') as f:
   writer=csv.writer(f);writer.writerow(['x','lower','upper','center','width','lateral'])
   for row in data['centerlines']:
    if 'center' in row:writer.writerow([row[n] for n in ['x','lower','upper','center','width','lateral']])
(OUT/'A-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='comparisons'},indent=2))
for c in result['comparisons']:print(c['time'],c['phase_error'],c['quarter']['front'],c['common_centerlines'])
