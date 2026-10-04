#!/usr/bin/env python3
from prepare_inputs import prm,root
import subprocess
for name,dt,last,end in [('A-replay','0.05','11','0.55'),('A-small-dt','0.0125','17','0.55')]:
 data={};prm.parse(root/'inputs/A.prm',data,[])
 output=root/('output-'+name)
 subprocess.run(['bash',str(root.with_name('bp3')/'branch_output.sh'),str(root/'output-A'),'1',str(output)],check=True)
 data[('Additional shared libraries',)]+=', $ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/bp3_local_tests/build/clock/libbp3_local_clock.release.so'
 data[('Output directory',)]='$ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/bp3_local_tests/output-'+name
 data[('Resume computation',)]='true'
 data[('Maximum time step',)]=dt;data[('Maximum first time step',)]=dt
 data[('End time',)]=end
 data[('Postprocess','BP3','Last accepted step')]=last
 data[('Postprocess','BP3','Profile time interval')]='0.001'
 data[('Postprocess','List of postprocessors')]+=', local restart clock'
 prm.write(data,root/'inputs'/(name+'.prm'))
