#!/usr/bin/env python3
"""Run the opt-in saved-state I_h kernel once, with a five-minute wall cap."""
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import time

directory = Path(__file__).resolve().parent
root = directory.parents[4]
binary = root / "build-pf-cpdi/aspect-release"
command = [str(binary), "--test", "[.fault_ih_performance]"]
start = time.monotonic()
with (directory / "fine-ih.log").open("w") as log:
    try:
        status = subprocess.run(command, cwd=root, stdout=log,
                                stderr=subprocess.STDOUT, timeout=300).returncode
    except subprocess.TimeoutExpired:
        status = 124
usage = resource.getrusage(resource.RUSAGE_CHILDREN)
report = dict(command=command, wall_seconds=time.monotonic()-start,
              user_seconds=usage.ru_utime, system_seconds=usage.ru_stime,
              peak_rss_KiB=usage.ru_maxrss, exit_status=status,
              executable_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
(directory / "fine-ih.resources.json").write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(report), flush=True)
raise SystemExit(status)
