#!/usr/bin/env python3
"""Reuse retained native restoring implementation with configurable patterns."""
from pathlib import Path
import hashlib,json,shlex,subprocess,argparse
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
p=argparse.ArgumentParser();p.add_argument('--build',type=Path,default=ROOT/'build-particle-free/stage4-wavelength');a=p.parse_args()
BUILD=a.build.resolve();BUILD.mkdir(exist_ok=True)
header=(HERE.parent/'stage4_rsf/stiffness.h').read_text()
header=header.replace('ns=state.sections.size(),nm=7;','ns=state.sections.size();\n  std::ifstream pattern_file(std::string(std::getenv("ASPECT_WAVELENGTH_INPUT"))+"/patterns.txt");\n  unsigned int nm;pattern_file>>nm;\n  std::vector<std::pair<unsigned int,unsigned int>> waves(nm);\n  for(auto &wave:waves)pattern_file>>wave.first>>wave.second;\n  AssertThrow(pattern_file,ExcMessage("Incomplete wavelength patterns."));')
header=header.replace('std::sin((j<6?j+1:48)*numbers::PI*(coordinate-2.5)/3.)','std::sin(waves[j].first*numbers::PI*(coordinate-2.5)/3.+waves[j].second*numbers::PI/2.)')
# Publish complete independent actions to disk before the next solve. No change
# to the operator, tolerances or numerical ordering; interrupted raw files remain.
header=header.replace('  struct Action\n', '  if(rank==0)sections.flush();\n  struct Action\n',1)
header=header.replace('          summary<<j', '          modes.flush();\n          summary<<j',1)
(BUILD/'restoring.h').write_text(header)
src=ROOT/'benchmarks/reconstructed_fault/particle_free/bulk_tip/checks.cc'
text=src.read_text().replace('#include <fstream>','#include <deal.II/lac/solver_gmres.h>\n#include <aspect/utilities.h>\n#include <fstream>',1)
text=text.replace('        void export_step()',f'#include "{BUILD}/restoring.h"\n#include "{HERE}/frozen.h"\n        void export_step()',1)
text=text.replace('            if (const auto file=std::getenv("ASPECT_TIP_IMPORT_PARENT"))import_parent(file);','            import_frozen_phase();',1)
text=text.replace('          const auto &s=*model().bulk_cohesive_state;\n          gate("single publication"','          frozen_wavelength_response();\n          return {"Frozen wavelength", "native restoring actions completed; no physical step"};\n          const auto &s=*model().bulk_cohesive_state;\n          gate("single publication"',1)
(BUILD/'checks.cc').write_text(text)
original=ROOT/'build-particle-free/bulk-tip/points/CMakeFiles/bulk_tip_checks.dir'
flags={}
for line in (original/'flags.make').read_text().splitlines():
 if line.startswith('CXX_'):
  k,v=line.split('=',1);flags[k.strip()]=shlex.split(v.strip())
compile=['/opt/gcc/12.4.0/bin/g++']+flags['CXX_DEFINES']+flags['CXX_INCLUDES']+flags['CXX_FLAGS']+['-c',str(BUILD/'checks.cc'),'-o',str(BUILD/'checks.o')]
link=shlex.split((original/'link.txt').read_text());link=[s for s in link if not s.startswith('-Wl,--dependency-file=')]
link=[str(BUILD/'checks.o') if s=='CMakeFiles/bulk_tip_checks.dir/checks.cc.o' else str(BUILD/'libbulk_tip_checks.release.so') if s=='libbulk_tip_checks.release.so' else s for s in link]
(BUILD/'build-command.json').write_text(json.dumps(dict(compile=compile,link=link,inputs={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [src,HERE/'frozen.h',HERE.parent/'stage4_rsf/stiffness.h',BUILD/'checks.cc',BUILD/'restoring.h']}),indent=2)+'\n')
subprocess.run(compile,cwd=ROOT,check=True);subprocess.run(link,cwd=ROOT,check=True)
print('Built frozen diagnostic checker')
