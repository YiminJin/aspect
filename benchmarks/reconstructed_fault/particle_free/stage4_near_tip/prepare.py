#!/usr/bin/env python3
"""One prescribed near-tip slipping state; retained native mechanics unchanged."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-near-tip-mpl')
from pathlib import Path
import hashlib,json,subprocess,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
OLD=ROOT/'build-particle-free/stage4-preload';BUILD=ROOT/'build-particle-free/stage4-near-tip'
OUT=HERE.parent/'evidence/stage4_near_tip';OUT.mkdir(exist_ok=True)
target=BUILD/'input-256';target.mkdir(exist_ok=True)
prior=ROOT/'build-particle-free/stage4-pilot/input-256'
for name in ('phase.txt','sections.txt','transfer.json'):shutil.copy2(prior/name,target/name)
a=np.loadtxt(prior/'background.txt',skiprows=1);s=np.loadtxt(prior/'sections.txt',skiprows=1)
x=a[:,0];v=np.zeros_like(x)
inside=(x>3)&(x<4);v[inside]=np.sin(np.pi*(x[inside]-3)/2)**2
v[(x>=4)&(x<=6)]=1.
inside=(x>6)&(x<6.125);v[inside]=np.cos(np.pi*(x[inside]-6)/.25)**2
v*=2e-4
assert np.all(s[v>0,2]==1),'Nonzero prepared sliding must stay in formed sections.'
new=np.c_[x,v,a[:,2]]
np.savetxt(target/'background.txt',new,header=str(len(a)),comments='',fmt='%.17g')
np.savetxt(target/'operating.csv',new,header='x,V,theta',delimiter=',',comments='',fmt='%.17g')
np.savetxt(OUT/'prepared-rate-profiles.csv',np.c_[x,a[:,1],v,a[:,2]],header='x,old_target_V,new_target_V,initial_unperturbed_age',delimiter=',',comments='',fmt='%.17g')
fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for ax in axs:
 ax.plot(x,1e3*a[:,1],label='Old preparation');ax.plot(x,1e3*v,label='New preparation')
 ax.axvspan(5.5,6,alpha=.12,color='gray');ax.axvline(6,color='k',ls=':',label='Age perturbation ends')
 ax.set(xlabel='Section x (m)',ylabel='Prepared slip rate (mm/s)');ax.legend(fontsize=8)
axs[0].set(xlim=(1.5,6.5),title='Nominal state before the 1% age perturbation')
axs[1].set(xlim=(5.4,6.2),title='Appreciable sliding through the formed taper')
for ext in ('png','pdf'):fig.savefig(OUT/f'prepared-rate-profiles.{ext}',dpi=170)
source=OLD/'tested/checks.cc';text=source.read_text()
preload=(OLD/'tested/preload.h').read_text()
needle='coordinate>2.5 && coordinate<5.5 ? std::pow(std::sin(numbers::PI*(coordinate-2.5)/3.),2)'
assert preload.count(needle)==1
preload=preload.replace(needle,'coordinate>3.0 && coordinate<6.0 ? std::pow(std::sin(numbers::PI*(coordinate-3.0)/3.),2)')
(BUILD/'preload.h').write_text(preload)
assert str(OLD/'preload.h') in text
text=text.replace(str(OLD/'preload.h'),str(BUILD/'preload.h'))
(BUILD/'checks.cc').write_text(text)
old=json.loads((OLD/'build-command.json').read_text())
commands={key:[v.replace(str(OLD),str(BUILD)) if v in [str(OLD/'checks.cc'),str(OLD/'checks.o'),str(OLD/'libbulk_tip_checks.release.so')] else v for v in old[key]] for key in ('compile','link')}
for law in ('weakening','neutral'):shutil.copy2(OLD/f'256-{law}.prm',BUILD/f'256-{law}.prm')
dependencies=[source,BUILD/'checks.cc',BUILD/'preload.h',HERE.parent/'stage4_preload/scan.h',HERE.parent/'stage4_release/observe.h',ROOT/'build-particle-free/stage4-pilot/import.h',*target.iterdir()]
commands['sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
(BUILD/'build-command.json').write_text(json.dumps(commands,indent=2)+'\n')
with (BUILD/'build.log').open('w') as log:
 for key in ('compile','link'):subprocess.run(commands[key],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
print('Built separate near-tip checker and prepared fixed physical inputs.')
