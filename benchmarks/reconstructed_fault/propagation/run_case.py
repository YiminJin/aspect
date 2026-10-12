#!/usr/bin/env python3
"""Run one selected P1 control (120 s); preserve numerical qualification failure."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("case", choices=["finite", "padded", "straight", "straight_fine"])
parser.add_argument("--executable", type=Path, required=True)
parser.add_argument("--build", type=Path, required=True)
parser.add_argument("--evidence", type=Path, required=True)
args = parser.parse_args()
here = Path(__file__).resolve().parent
repo = here.parents[2]
build = args.build.resolve()
evidence = args.evidence.resolve()
executable = args.executable.resolve()
plugin = build / "libcontrolled_history.release.so"
assert executable.is_file() and plugin.is_file()
output = build / ("output-" + args.case)
prefix = evidence / args.case
assert not output.exists() and not list(evidence.glob(args.case + ".*")), "Refusing to overwrite evidence"
evidence.mkdir(parents=True, exist_ok=True)
overlay = build / (args.case + ".prm")
overlay.write_text(f"include {here / (args.case + '.prm')}\n"
                   f"set Additional shared libraries = {plugin}\n"
                   f"set Output directory = {output}\n")
env = os.environ.copy()
env.update(ASPECT_SOURCE_DIR=str(repo), OMP_NUM_THREADS="1",
           DEAL_II_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
env["PATH"] = "/opt/gcc/12.4.0/bin:/opt/openmpi/5.0.6/bin:" + env["PATH"]
command = [str(executable), str(overlay)]
start = time.monotonic()
with prefix.with_suffix(".log").open("w") as log:
    try:
        code = subprocess.run(command, cwd=repo, env=env, stdout=log,
                              stderr=subprocess.STDOUT, timeout=120).returncode
    except subprocess.TimeoutExpired:
        code = 124
record = dict(command=command, cwd=str(repo), exit_code=code,
              seconds=time.monotonic()-start,
              child_max_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
              environment={k: env[k] for k in ["ASPECT_SOURCE_DIR", "OMP_NUM_THREADS",
                                             "DEAL_II_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PATH"]},
              hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      [executable, plugin, overlay, repo / "tests/phase_field_controlled_history.cc"]})
prefix.with_suffix(".run.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2), flush=True)
if code not in (0, 1):
    sys.exit(code)
for name in ["controlled-history.csv", "initialization-nodes.csv", "initialization-particles.csv"]:
    shutil.copyfile(output / name, evidence / (args.case + "." + name))
check = subprocess.run([sys.executable, str(repo / "tests/phase_field_controlled_history_check.py"),
                        str(prefix.with_suffix(".controlled-history.csv")),
                        str(prefix.with_suffix(".log"))], capture_output=True, text=True)
prefix.with_suffix(".checks.json").write_text(check.stdout)
if check.stderr:
    print(check.stderr, file=sys.stderr)
assert record["child_max_rss_kib"] < 2*1024*1024, "2 GiB RSS budget exceeded"
assert sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) < 10*1024*1024, "10 MiB output budget exceeded"
if code == 1:
    assert "physical admissibility is not qualified" in " ".join(prefix.with_suffix(".log").read_text().split()), "Unexpected ASPECT failure"
    assert check.returncode == 2, "ASPECT/checker disagreement"
sys.exit(check.returncode)
