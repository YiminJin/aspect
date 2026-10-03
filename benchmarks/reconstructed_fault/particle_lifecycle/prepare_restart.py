#!/usr/bin/env python3
"""Clone a completed local run without overwriting any evidence."""
from pathlib import Path
import shutil, sys
root = Path(__file__).resolve().parent
source, target = (Path(x).resolve() for x in sys.argv[1:])
assert source.is_dir() and (source/'restart/last_good_checkpoint.txt').is_file()
assert target.parent == root and not target.exists()
shutil.copytree(source, target)
for pattern in ('lifecycle_rank*.csv', 'traction.csv', 'rng-*.txt'):
    for path in target.glob(pattern):
        path.rename(path.with_name(path.name+'.before_restart'))
