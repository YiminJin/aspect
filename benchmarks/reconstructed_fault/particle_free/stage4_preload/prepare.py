#!/usr/bin/env python3
"""Separate benchmark checker; preserve all qualified binaries and source."""
from pathlib import Path
import hashlib,json,subprocess,shutil
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
OLD=ROOT/'build-particle-free/stage4-release';BUILD=ROOT/'build-particle-free/stage4-preload'
BUILD.mkdir(exist_ok=True)
source=OLD/'tested/checks.cc'
text=source.read_text()
preload=ROOT/'benchmarks/reconstructed_fault/particle_free/stage4_wavelength/preload.h'
p=preload.read_text().replace('const double gamma=2e-4,extension=',
 'const double gamma=std::stod(std::getenv("ASPECT_PRELOAD_GAMMA")),extension=')
assert p!=preload.read_text()
(BUILD/'preload.h').write_text(p)
assert str(preload) in text
text=text.replace(str(preload),str(BUILD/'preload.h'))
needle='        void export_step()';assert text.count(needle)==1
text=text.replace(needle,f'#include "{HERE}/scan.h"\n'+needle,1)
needle='          observe_phase_release();';assert text.count(needle)==1
text=text.replace(needle,needle+'\n          observe_preload_scan();',1)
(BUILD/'checks.cc').write_text(text)
old=json.loads((OLD/'build-command.json').read_text())
commands={key:[v.replace(str(OLD),str(BUILD)) if v in [str(OLD/'checks.cc'),str(OLD/'checks.o'),str(OLD/'libbulk_tip_checks.release.so')] else v for v in old[key]] for key in ('compile','link')}
for law in ('weakening','neutral'):shutil.copy2(OLD/f'256-{law}.prm',BUILD/f'256-{law}.prm')
commands['sha256']={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in
                   (source,BUILD/'checks.cc',BUILD/'preload.h',HERE/'scan.h',
                    HERE.parent/'stage4_release/observe.h',ROOT/'build-particle-free/stage4-pilot/import.h')}
(BUILD/'build-command.json').write_text(json.dumps(commands,indent=2)+'\n')
with (BUILD/'build.log').open('w') as log:
 for key in ('compile','link'):subprocess.run(commands[key],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
print('Built preload calibration checker.')
