#!/usr/bin/env python3
"""Check the bounded cold-lookup comparison without running simulations."""
import json
from pathlib import Path
import re
import argparse

directory = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--cartesian", action="store_true")
args = parser.parse_args()
prefix = "cold-cartesian-" if args.cartesian else "cold-"
results = {}
for mode in ("baseline", "rejection"):
    log = (directory/f"{prefix}{mode}.log").read_text()
    resources = json.loads((directory/f"{prefix}{mode}.resources.json").read_text())
    assert resources["exit_status"] == 0 and "All tests passed" in log
    result = re.search(r"^Cold result:.*$", log, re.M)[0]
    cold = float(re.search(r"cold seconds=([^,]+)", log)[1])
    results[mode] = dict(cold_seconds=cold, resources=resources, result=result)
assert results["baseline"]["result"] == results["rejection"]["result"], \
    "Found/missing sequence digest, counts, or round-trip-precision integrals changed"
results["speedup"] = results["baseline"]["cold_seconds"]/results["rejection"]["cold_seconds"]
results["peak_rss_difference_KiB"] = (results["rejection"]["resources"]["peak_rss_KiB"]
                                      - results["baseline"]["resources"]["peak_rss_KiB"])
with (directory/f"{prefix}comparison.json").open("w") as output:
    json.dump(results, output, indent=2)
print(json.dumps(results, indent=2))
