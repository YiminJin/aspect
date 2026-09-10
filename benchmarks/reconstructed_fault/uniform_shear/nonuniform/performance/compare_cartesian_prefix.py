#!/usr/bin/env python3
"""Require genuine acceptance and unchanged exports in the production prefix."""
import json
from pathlib import Path
import re
import sys

directory = Path(__file__).resolve().parent
sys.path.insert(0, str(directory.parent))
from measure_case import summarize

log = directory/"cartesian-prefix.log"
steps = summarize(log)
assert len(steps) == 2
for step in steps:
    final = step["nonlinear"][-1]
    assert final["bulk"] < final["bulk target"]
    assert final["surface"] < 1e-8*final["surface scale"]
    assert all(row["fresh"] <= row["target"] for row in step["linear"])
exports = {
    path.name: path.read_bytes() == (directory/"cartesian-prefix"/path.name).read_bytes()
    for path in (directory/"local").glob("*.csv")}
assert len(exports) == 19 and all(exports.values())
work = re.findall(
    r"hits=(\d+), rebuilds=(\d+), stored points=(\d+), rejected requests=(\d+), "
    r"mapping=([^,]+), rejection eligible=(\d+), preparation seconds=([^\n]+)", log.read_text())
assert len(work) == 2
assert int(work[0][1]) > 0 and int(work[1][0]) > 0 and int(work[1][1]) == 0
assert all(int(row[3]) > 0 and "MappingCartesian" in row[4] and row[5] == "1" for row in work)
resources = json.loads((directory/"cartesian-prefix.resources.json").read_text())
assert resources["exit_status"] == 0
report = dict(resources=resources, converged_states=len(steps), exports_identical=exports,
              lookup_work=work, final_residuals=[s["nonlinear"][-1] for s in steps],
              fresh_checks=sum(len(s["linear"]) for s in steps))
with (directory/"cartesian-prefix-comparison.json").open("w") as output:
    json.dump(report, output, indent=2)
print(json.dumps(report, indent=2))
