#!/usr/bin/env python3
"""Build a separate synthetic-preload checker and fresh native pilot inputs."""
from pathlib import Path
import os,argparse,hashlib,json,shlex,subprocess,shutil
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];BUILD=ROOT/'build-particle-free/stage4-pilot'
p=argparse.ArgumentParser();p.add_argument('--inputs-only',action='store_true');a=p.parse_args()
BUILD.mkdir(exist_ok=True)
for nx in (256,512):
 source=ROOT/f'build-particle-free/stage4-wavelength/input-{nx}';target=BUILD/f'input-{nx}'
 if not target.exists():
  target.mkdir()
  for name in ('phase.txt','sections.txt','operating.csv','transfer.json'):shutil.copy2(source/name,target/name)
  v=np.genfromtxt(source/'operating.csv',delimiter=',',names=True)
  np.savetxt(target/'background.txt',np.c_[v['x'],v['V'],v['theta']],header=str(len(v)),comments='',fmt='%.17g')
 for law in ('weakening','neutral'):
  fixture=(source/'fixture.prm').read_text()
  edits={'set Characteristic slip distance = .0005':'set Characteristic slip distance = 3.2211126126299625e-7',
         'set Initial time step = .00125':'set Initial time step = .00004',
         'set Bulk tip initial age = .5':'set Bulk tip initial age = .0016105563063149812',
         'set Evolution effect parameters = .02':f'set Evolution effect parameters = {".02" if law=="weakening" else ".015"}',
         'set Reference friction coefficients = .6':f'set Reference friction coefficients = {".6" if law=="weakening" else ".6080471895621705"}',
         '0.01*(y-2)-if(t>0.04,0.0041954981558863999,0)*(x-4);if(t>0.04,0.0041954981558863999,0)*(y-2)':'0.01*(y-2)-0.0041954981558863999*(x-4);0.0041954981558863999*(y-2)'}
  for old,new in edits.items():assert fixture.count(old)==1,old;fixture=fixture.replace(old,new)
  (BUILD/f'{nx}-{law}.prm').write_text('# Fresh, mechanically preloaded fixed-phase synthetic pilot; not a growth-checkpoint continuation.\n'+fixture)
if a.inputs_only:raise SystemExit(0)
src=HERE.parent/'bulk_tip/checks.cc';text=src.read_text()
text=text.replace('#include <fstream>','#include <deal.II/lac/solver_gmres.h>\n#include <aspect/utilities.h>\n#include <fstream>',1)
importer=(HERE/'frozen.h').read_text().split('void frozen_wavelength_response()')[0]
(BUILD/'import.h').write_text(importer)
text=text.replace('        void export_step()',f'#include "{BUILD}/import.h"\n#include "{HERE}/preload.h"\n        void export_step()',1)
text=text.replace('            if (const auto file=std::getenv("ASPECT_TIP_IMPORT_PARENT"))import_parent(file);','            import_frozen_phase();',1)
needle='          const auto &s=*model().bulk_cohesive_state;\n          gate("single publication"'
assert text.count(needle)==1
text=text.replace(needle,'          if(this->get_timestep_number()==0)prepare_slipping_background();\n'+needle,1)
needle='          if (model().uses_integrated_bulk_tip())export_step();'
text=text.replace(needle,needle+'\n          observe_slipping_pilot();\n          if(s.step==0)return {"RSF preload", "equilibrated initial stress and section law checked"};',1)
needle='if (!stride || s.step%std::max(1,std::stoi(stride))==0)'
assert text.count(needle)==1
text=text.replace(needle,'if (!std::getenv("ASPECT_PILOT_COMPACT") && (!stride || s.step%std::max(1,std::stoi(stride))==0))')
(BUILD/'checks.cc').write_text(text)
original=ROOT/'build-particle-free/bulk-tip/points/CMakeFiles/bulk_tip_checks.dir';flags={}
for line in (original/'flags.make').read_text().splitlines():
 if line.startswith('CXX_'):
  k,v=line.split('=',1);flags[k.strip()]=shlex.split(v.strip())
compile=['/opt/gcc/12.4.0/bin/g++']+flags['CXX_DEFINES']+flags['CXX_INCLUDES']+flags['CXX_FLAGS']+['-c',str(BUILD/'checks.cc'),'-o',str(BUILD/'checks.o')]
link=shlex.split((original/'link.txt').read_text());link=[s for s in link if not s.startswith('-Wl,--dependency-file=')]
link=[str(BUILD/'checks.o') if s=='CMakeFiles/bulk_tip_checks.dir/checks.cc.o' else str(BUILD/'libbulk_tip_checks.release.so') if s=='libbulk_tip_checks.release.so' else s for s in link]
(BUILD/'build-command.json').write_text(json.dumps(dict(compile=compile,link=link,sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [src,HERE/'preload.h',BUILD/'import.h',BUILD/'checks.cc']}),indent=2)+'\n')
subprocess.run(compile,cwd=ROOT,check=True);subprocess.run(link,cwd=ROOT,check=True)
print('Built separate native preload checker')
