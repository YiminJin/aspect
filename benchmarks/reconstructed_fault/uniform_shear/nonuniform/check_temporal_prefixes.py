#!/usr/bin/env python3
"""Incremental byte checks only after the production export has completed."""
import hashlib
import json
from pathlib import Path
import re

base = Path(__file__).resolve().parent
destination = base/"domain-convergence-completion/prefix-checks.json"
report = json.loads(destination.read_text()) if destination.exists() else {}
def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()
for case, last in (("time025",7),("time0125",8)):
    log = (base/f"domain-convergence-completion/{case}.log").read_text()
    exported = [int(k) for k in re.findall(r"K1 accepted-state export:\s+(\d+)",log)]
    checked = report.setdefault(case,{})
    for k in exported:
        if k > last or str(k) in checked: continue
        old = base/f"domain-convergence/{case}"
        new = base/f"domain-convergence-completion/{case}"
        files = sorted(old.glob(f"*_{k}.csv"))
        assert files
        hashes = {}
        for path in files:
            expected, actual = digest(path), digest(new/path.name)
            assert actual == expected, f"Changed accepted prefix: {case}/{path.name}"
            hashes[path.name] = actual
        checked[str(k)] = hashes
destination.write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({case:dict(steps=sorted(map(int,rows)),files=sum(len(v) for v in rows.values()))
                  for case,rows in report.items()}))
