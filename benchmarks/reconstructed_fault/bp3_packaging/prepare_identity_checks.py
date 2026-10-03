from prepare_inputs import root,parse,write
import subprocess,sys
base={};parse(root/'inputs/resume.prm',base,[])
for name,key,value in [
 ('material',('Material model','Phase field fault','Characteristic slip distance'),'0.009'),
 ('mesh',('Mesh refinement','Initial global refinement'),'1'),
 ('limiter',('Particles','Maximum particles per cell'),'25'),
 ('filter',('Postprocess','BP3 restored monitor','Normal filter length'),'21')]:
 case='reject-'+name;data=dict(base)
 data[key]=value;data[('Output directory',)]='$ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/bp3_packaging/output-'+case
 write(data,root/'inputs'/(case+'.prm'))
 subprocess.run([sys.executable,str(root/'prepare_restart.py'),str(root/'output-create'),str(root/('output-'+case))],check=True)
