#!/usr/bin/env python3
"""Generate a diagnostic-only copy of the existing checker; no production edits."""
from pathlib import Path
import hashlib,json,shlex,subprocess
ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
BUILD=ROOT/'build-particle-free/stage4-rsf'
BUILD.mkdir(exist_ok=True)
src=ROOT/'benchmarks/reconstructed_fault/particle_free/bulk_tip/checks.cc'
text=src.read_text()
text=text.replace('#include <fstream>','#include <deal.II/lac/solver_gmres.h>\n#include <aspect/utilities.h>\n#include <fstream>',1)
needle='        void export_step()'
assert text.count(needle)==1
text=text.replace(needle,'#include "'+str(HERE/'stiffness.h')+'"\n'+needle,1)
needle='          if (model().uses_integrated_bulk_tip())export_step();'
assert text.count(needle)==1
text=text.replace(needle,needle+'\n          if (s.step>0 && model().uses_integrated_bulk_tip())measure_restoring_response();',1)
(BUILD/'checks.cc').write_text(text)
original=ROOT/'build-particle-free/bulk-tip/points/CMakeFiles/bulk_tip_checks.dir'
flags={}
for line in (original/'flags.make').read_text().splitlines():
 if line.startswith('CXX_'):
  k,v=line.split('=',1);flags[k.strip()]=shlex.split(v.strip())
compile=['/opt/gcc/12.4.0/bin/g++']+flags['CXX_DEFINES']+flags['CXX_INCLUDES']+flags['CXX_FLAGS']+['-c',str(BUILD/'checks.cc'),'-o',str(BUILD/'checks.o')]
link=shlex.split((original/'link.txt').read_text())
link=[s for s in link if not s.startswith('-Wl,--dependency-file=')]
link=[str(BUILD/'checks.o') if s=='CMakeFiles/bulk_tip_checks.dir/checks.cc.o' else str(BUILD/'libbulk_tip_checks.release.so') if s=='libbulk_tip_checks.release.so' else s for s in link]
record=dict(original_checker_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),diagnostic_header_sha256=hashlib.sha256((HERE/'stiffness.h').read_bytes()).hexdigest(),generated_sha256=hashlib.sha256(text.encode()).hexdigest(),compile=compile,link=link)
(BUILD/'build-command.json').write_text(json.dumps(record,indent=2)+'\n')
subprocess.run(compile,cwd=ROOT,check=True)
subprocess.run(link,cwd=ROOT,check=True)
print('Built diagnostic-only checker',flush=True)
