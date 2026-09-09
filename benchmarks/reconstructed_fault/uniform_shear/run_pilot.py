#!/usr/bin/env python3
"""Run the one-rank pilot with a ten-minute cap and Linux resource accounting."""
import json
import resource
import subprocess
import sys
import time

if len(sys.argv) != 3:
    raise SystemExit("usage: run_pilot.py /path/to/aspect /path/to/pilot.prm")
start = time.monotonic()
try:
    result = subprocess.run(sys.argv[1:], timeout=600, check=False)
    status = result.returncode
except subprocess.TimeoutExpired:
    status = 124
print(json.dumps(dict(wall_seconds=time.monotonic()-start,
                      peak_rss_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                      exit_status=status)), file=sys.stderr, flush=True)
raise SystemExit(status)
