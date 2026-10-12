#!/usr/bin/env python3
"""Bounded frozen-field follow-up: .0125 m, then .00625 only if needed.
Reuse audited analytic-Q1 reference and consistent operator; no physical solves.
"""
import ast,csv,hashlib,json,os
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize_scalar
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];BUILD=ROOT/'build-propagation-integration'
OUT=HERE/'review_checkpoint_evidence';OUT.mkdir(exist_ok=True)
assert not (OUT/'normalization.json').exists(), 'Preserve earlier follow-up evidence.'
# Load only the existing numerical definitions, not their artifact-writing runs.
ns=dict(np=np,csv=csv,leggauss=leggauss,minimize_scalar=minimize_scalar)
for filename,names in [('analyze_irreversibility.py',{'rows','field','at','integral'}),('accuracy_normalization.py',{'project','assess'})]:
    tree=ast.parse((HERE/filename).read_text())
    definitions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(definitions)==len(names)
    exec(compile(ast.Module(body=definitions,type_ignores=[]),str(HERE/filename),'exec'),ns)
rows,field,project,assess,integral=[ns[x] for x in ['rows','field','project','assess','integral']]
calibration=rows(BUILD/'output-irreversibility_dt03125/material-calibration.csv')
ns['MATERIAL_M']=calibration[0]['m'];assert all(r['m']==ns['MATERIAL_M'] for r in calibration)
inputs=[];records=[];spacings=[]
for ds in [.0125,.00625]:
    spacings.append(ds);new=[]
    for label in ['irreversibility_dt125','irreversibility_dt0625','irreversibility_dt03125']:
        folder=BUILD/('output-'+label);feedback=rows(folder/'feedback.csv')
        onset=next(int(r['step']) for r in feedback if r['phase_change']>4.922e-12)
        for name,step in [('onset',onset),('strong',int(feedback[-1]['step']))]:
            path=folder/f'material-phase-accepted-{step}.csv';inputs.append(path);v=field(path)
            nodes,I,conservation,constant=project(v,ds)
            _,reference,reference_conservation,reference_constant=project(v,ds,32,True)
            _,refined,_,_=project(v,ds,48,True)
            native=assess(v,nodes,I);accurate=assess(v,nodes,reference)
            uncertainty=float(max(abs(reference/refined-1)))
            assert native['min_I']>0 and accurate['min_I']>0 and uncertainty<2e-12
            native.update(label=label,step=step,time=feedback[step]['time'],state=name,spacing=ds,phi_max=float(v.max()),constant_error=constant,conservation_error=conservation,reference_projection=accurate,reference_constant_error=reference_constant,reference_conservation_error=reference_conservation,projection_quadrature_uncertainty=uncertainty,native_Q3_vs_split_Q32=float(max(abs(I/reference-1))),source_target=.005)
            native['frozen_field_target_pass']=max(native['max_source_error'],accurate['max_source_error'])<.005
            new.append(native);records.append(native)
            x=np.linspace(0,1,4001);direct=integral(v,x,ns['MATERIAL_M'])
            with (OUT/f'normalization-{label}-{step}-{ds:g}.csv').open('w') as f:
                writer=csv.writer(f);writer.writerow(['x','direct','consistent_native_Q3','consistent_split_Q32','source_error_native_Q3','source_error_split_Q32'])
                projected=np.interp(x,nodes,I);high=np.interp(x,nodes,reference)
                writer.writerows(zip(x,direct,projected,high,direct/projected-1,direct/high-1))
            print(label,name,ds,f"native {100*native['max_source_error']:.6f}%",f"spatial {100*accurate['max_source_error']:.6f}%",f"worst x={native['worst_x']:.8f}",flush=True)
    if all(a['frozen_field_target_pass'] for a in new):
        print(f'{ds} m passes all six retained fields; no smaller spacing needed.',flush=True)
        break
result=dict(source_target=.005,spacings_tested=spacings,selected_spacing=spacings[-1] if all(a['frozen_field_target_pass'] for a in new) else None,all_frozen_fields_pass=all(a['frozen_field_target_pass'] for a in new),qualification='six frozen accepted fields only',coupled_model_qualified=False,runtime_m=ns['MATERIAL_M'],records=records,input_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(inputs))})
(OUT/'normalization.json').write_text(json.dumps(result,indent=2)+'\n')
os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-review-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(11,3.4),constrained_layout=True)
for col,label in enumerate(['irreversibility_dt125','irreversibility_dt0625','irreversibility_dt03125']):
    for ds in spacings:
        r=next(a for a in records if a['label']==label and a['spacing']==ds and a['state']=='strong')
        data=np.genfromtxt(OUT/f"normalization-{label}-{r['step']}-{ds:g}.csv",delimiter=',',names=True)
        axes[col].plot(data['x'],100*data['source_error_split_Q32'],label=f'{ds:g} m')
    axes[col].axhline(.5,color='black',ls=':');axes[col].axhline(-.5,color='black',ls=':')
    axes[col].set(title=label.replace('irreversibility_','')+', t=1 s',xlabel='x (m)',ylabel='Source error (%)');axes[col].legend();axes[col].grid(alpha=.2)
fig.suptitle('Frozen Q1 phase, analytic transverse reference; accurate consistent projection')
fig.savefig(OUT/'normalization.png',dpi=160);fig.savefig(OUT/'normalization.pdf')
