#!/usr/bin/env python3
from pathlib import Path
r=Path(__file__).resolve().parent;repo=r.parents[2]
base='include $ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/refactoring_boundary/inputs/auto-bp3-base.prm\n'
base+='set Additional shared libraries = $ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/refactoring_r6b/plugin-build/bp3/libbp3_restore_150x50.release.so, $ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/particle_replenishment/build/libreplenishment.release.so, $ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/particle_replenishment/build/libcoupled_replenishment.release.so\n'
base+='''set End time = 200
subsection Postprocess
 set List of postprocessors = reconstructed fault BP3, particles, reconstructed faults, BP3 output complete, BP3 restored monitor, coupled replenishment observer
 subsection BP3
  set Audit full state every step = false
  set Last accepted step = 2
 end
 subsection Particles
  set Data output format = none
 end
end
subsection Particles
 set Load balancing strategy = remove and add particles
 set Minimum particles per cell = 12
 set Maximum particles per cell = 24
 set Particle addition algorithm = point density function
 set Particle removal algorithm = point density function
 set Interpolation scheme = observed native LLS
 subsection Generator
  subsection Reference cell
   set Number of particles per cell per direction = 4
  end
 end
 subsection Interpolator
  subsection Observed native LLS
   set Limit histories = true
  end
 end
end
'''
(r/'inputs/coupled-base.prm').write_text(base)
for case in ['coupled-regular','coupled-random-5433','coupled-regular-Hlimited','coupled-random-5433-Hlimited','crossing-after-create','crossing-after-resume','crossing-serial','crossing-mpi','crossing-create','crossing-resume','crossing-retry','crossing-direct','crossing-audit-original']:
 s='include $ASPECT_SOURCE_DIR/'+str((r/'inputs/coupled-base.prm').relative_to(repo))+'\n'
 s+='set Output directory = '+'$ASPECT_SOURCE_DIR/'+str((r/('output-'+case)).relative_to(repo))+'\n'
 if 'random' in case:s+='''subsection Particles
 set Particle generator name = random uniform
 subsection Generator
  subsection Random uniform
   set Number of particles = 30000
   set Random cell selection = true
   set Random number seed = 5433
  end
 end
end
'''
 if case.startswith('crossing'):
  s+='''subsection Particles
 set Particle generator name = crossing reference cell
end
'''
  if case!='crossing-audit-original':s+='''subsection Postprocess
 subsection Coupled replenishment
  set Track birth audit = true
 end
end
'''
 if case=='crossing-create':s+='set End time = 0\nsubsection Postprocess\n subsection BP3\n set Last accepted step = 0\n end\nend\n'
 if case in ('crossing-resume','crossing-after-resume'):s+='set Resume computation = true\n'
 if case in ('crossing-direct','crossing-retry'):
  s+='set End time = 50\nsubsection Postprocess\n subsection BP3\n set Last accepted step = 1\n end\nend\n'
 if case=='crossing-direct':s+='set Maximum time step = 50\nset Maximum first time step = 50\n'
 if case=='crossing-retry':
  s=s.replace('set End time = 50','set End time = 100')
  s+='''set Nonlinear solver failure strategy = cut timestep size
subsection Postprocess
 subsection Coupled replenishment
  set Reject first step = true
 end
end
'''
 if case=='crossing-after-create':s+='set End time = 100\nsubsection Postprocess\n subsection BP3\n set Last accepted step = 1\n end\nend\n'
 if case.startswith('crossing') or case.endswith('-Hlimited'):
  s+='subsection Particles\n subsection Interpolator\n  subsection Observed native LLS\n   set Limit crack driving history = true\n  end\n end\nend\n'
 (r/f'inputs/{case}.prm').write_text(s)
