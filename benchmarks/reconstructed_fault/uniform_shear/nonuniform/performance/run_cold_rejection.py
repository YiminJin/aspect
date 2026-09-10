#!/usr/bin/env python3
"""One bounded nine-profile comparison run; never overwrite prior evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["baseline", "rejection"])
parser.add_argument("--cartesian", action="store_true")
args = parser.parse_args()
directory = Path(__file__).resolve().parent
root = directory.parents[4]
binary = root / "build-pf-cpdi/aspect-release"
prefix = directory / ("cold-"+("cartesian-" if args.cartesian else "")+args.mode)
env = os.environ.copy()
env["ASPECT_FAULT_PERFORMANCE_STATE"] = str(directory.parent/"domain-convergence/space128")
env.pop("ASPECT_IH_LOOKUP_BASELINE", None)
env.pop("ASPECT_IH_CARTESIAN", None)
if args.cartesian:
    env["ASPECT_IH_CARTESIAN"] = "1"
if args.mode == "baseline":
    env["ASPECT_IH_LOOKUP_BASELINE"] = "1"
command = [str(binary), "--test", "[.fault_ih_performance]"]
start = time.monotonic()
with prefix.with_suffix(".log").open("x") as log:
    try:
        status = subprocess.run(command, cwd=root, env=env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=120).returncode
    except subprocess.TimeoutExpired:
        status = 124
usage = resource.getrusage(resource.RUSAGE_CHILDREN)
report = dict(command=command, mode=args.mode, wall_seconds=time.monotonic()-start,
              user_seconds=usage.ru_utime, system_seconds=usage.ru_stime,
              peak_rss_KiB=usage.ru_maxrss, exit_status=status,
              executable_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
with prefix.with_suffix(".resources.json").open("x") as output:
    json.dump(report, output, indent=2)
print(json.dumps(report), flush=True)
raise SystemExit(status)
