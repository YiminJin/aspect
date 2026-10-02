#!/usr/bin/env python3
"""Copy only this experiment's completed checkpoint run to its restart target."""
from pathlib import Path
import shutil,sys
r=Path(__file__).resolve().parent
before_after=sys.argv[1] if len(sys.argv)>1 else 'before'
assert before_after in ('before','after')
prefix='crossing-'+('after-' if before_after=='after' else '')
src=r/('output-'+prefix+'create');dst=r/('output-'+prefix+'resume')
assert (src/'restart/last_good_checkpoint.txt').is_file()
assert not dst.exists(),f'Refusing to overwrite {dst}'
shutil.copytree(src,dst)
for name in ['lifecycle_rank0.csv','traction.csv']:
 (dst/name).rename(dst/(name+'.before_restart'))
