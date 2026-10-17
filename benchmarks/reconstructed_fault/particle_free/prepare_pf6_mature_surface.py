#!/usr/bin/env python3
import sys,shutil,json,hashlib,argparse
from pathlib import Path
from run_pf6_mature import flatten,sha
parser=argparse.ArgumentParser(description='Prepare the full-geometry, coarse initialization-only mature DG surface control; does not launch ASPECT.')
parser.add_argument('output_directory',type=Path)
parser.add_argument('--surface-spacing',type=float,choices=(10.,40.),default=10.)
a=parser.parse_args()
root=Path(__file__).resolve().parents[3];out=a.output_directory.resolve();out.mkdir(parents=True,exist_ok=False)
plugin=root/'build-particle-free/mature-pilot/ordinary/bp3-plugin/libbp3_restore_150x50.release.so';obs=root/'build-particle-free/mature-pilot/ordinary/observers/libbp3_coupled_checks.release.so';binary=root/'build-particle-free/native/aspect-release'
for p in [plugin,obs,binary]:shutil.copy2(p,out/p.name)
s=flatten(root/'benchmarks/reconstructed_fault/bp3_native/section7cd-transition.prm',root)
s=s.replace('set X extent = 2000','set X extent = 150000').replace('set Y extent = 1000','set Y extent = 50000').replace('set Box origin X coordinate = -1234.5','set Box origin X coordinate = -60000').replace('set Box origin Y coordinate = 137.25','set Box origin Y coordinate = 0').replace('set X repetitions = 4','set X repetitions = 75').replace('set Y repetitions = 2','set Y repetitions = 25').replace('set Minimum refinement level = 2','set Minimum refinement level = 1').replace('set Structural point spacing = 20',f'set Structural point spacing = {a.surface_spacing:.17g}')
s=s.replace(str(root/'benchmarks/reconstructed_fault/bp3_native/fault.txt'),str(root/'benchmarks/reconstructed_fault/bp3_native/server/fault.txt'))
s=s.replace('set Compositional field methods = particles, particles, particles, particles, particles','set Compositional field methods = static, static, static, static, static')
s='\n'.join('  set Mapped particle properties =' if l.strip().startswith('set Mapped particle properties') else l for l in s.splitlines())+'\n'
s+=f'''
set Additional shared libraries = {out/plugin.name}, {out/obs.name}
set Output directory = {out/'run'}
subsection Material model
 subsection Phase field fault
  set Bulk history representation = DG trial stress
  set Use homogeneous mesh data = true
  set FE bulk normal quadrature points = 6
 end
end
subsection Particles
 set Generate particle domains = false
 set Generate CPDI data for particle domains = false
 set Generate face data for particle domains = false
end
'''
(out/'input.prm').write_text(s)
(out/'manifest.json').write_text(json.dumps(dict(artifacts={p.name:sha(p) for p in [binary,plugin,obs]},input_sha256=sha(out/'input.prm'),command=[str(out/binary.name),str(out/'input.prm')],scope='Full physical geometry and transition; initialization only; coarse mesh is not a spatial accuracy reference'),indent=2)+'\n')
