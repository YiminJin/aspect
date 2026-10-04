"""Create new branches only after their parent runs finish; never modify parents."""
from pathlib import Path
import subprocess,shutil
r=Path(__file__).resolve().parent
for case in ['resume','direct2-final','retry2']:
 subprocess.run(['bash',str(r.parent/'bp3/branch_output.sh'),str(r/'output-create'),'2',str(r/('output-'+case))],check=True)
for parent,branch in [('transport-A-final','transport-A-resume'),('transport-A-serial','transport-A-serial-resume')]:
 shutil.copytree(r/('output-'+parent)/'restart',r/('output-'+branch)/'restart')
