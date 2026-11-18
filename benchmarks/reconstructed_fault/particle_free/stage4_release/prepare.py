#!/usr/bin/env python3
"""Build only an observational checker; reuse the exact qualified pilot inputs."""
from pathlib import Path
import hashlib,json,subprocess,shutil
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
OLD=ROOT/'build-particle-free/stage4-pilot';BUILD=ROOT/'build-particle-free/stage4-release'
BUILD.mkdir(exist_ok=True)
source=OLD/'tested/checks.cc'
text=source.read_text()
needle='        void export_step()'
assert text.count(needle)==1
text=text.replace(needle,f'#include "{HERE}/observe.h"\n'+needle,1)
needle='          observe_slipping_pilot();'
assert text.count(needle)==1
text=text.replace(needle,needle+'\n          observe_phase_release();',1)
(BUILD/'checks.cc').write_text(text)
old=json.loads((OLD/'build-command.json').read_text())
compile=[v.replace(str(OLD/'checks.cc'),str(BUILD/'checks.cc')).replace(str(OLD/'checks.o'),str(BUILD/'checks.o')) for v in old['compile']]
link=[v.replace(str(OLD/'checks.o'),str(BUILD/'checks.o')).replace(str(OLD/'libbulk_tip_checks.release.so'),str(BUILD/'libbulk_tip_checks.release.so')) for v in old['link']]
for law in ('weakening','neutral'):shutil.copy2(OLD/f'256-{law}.prm',BUILD/f'256-{law}.prm')
files=[source,BUILD/'checks.cc',HERE/'observe.h',ROOT/'benchmarks/reconstructed_fault/particle_free/stage4_wavelength/preload.h',OLD/'import.h']
(BUILD/'build-command.json').write_text(json.dumps(dict(compile=compile,link=link,sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),indent=2)+'\n')
subprocess.run(compile,cwd=ROOT,check=True);subprocess.run(link,cwd=ROOT,check=True)
print('Built separate phase-release observer; qualified pilot checker unchanged.')
