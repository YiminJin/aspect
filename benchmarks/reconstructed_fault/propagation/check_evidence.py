#!/usr/bin/env python3
"""Check saved P1 controls without rerunning simulations; report diagnosis only."""
import csv
import json
import math
from pathlib import Path
import re
import subprocess
import sys

here = Path(__file__).resolve().parent
repo = here.parents[2]
evidence = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "evidence"

def table(path):
    rows = [{k: v if k == "stage" else float(v) for k, v in r.items()}
            for r in csv.DictReader(path.open())]
    assert rows and all(math.isfinite(v) for r in rows for k, v in r.items() if k != "stage")
    return rows

results = {}
for case in ["finite", "padded", "straight", "straight_fine"]:
    log = (evidence / (case + ".log")).read_text()
    run = json.loads((evidence / (case + ".run.json")).read_text())
    check = subprocess.run([sys.executable, str(repo / "tests/phase_field_controlled_history_check.py"),
                            str(evidence / (case + ".controlled-history.csv")),
                            str(evidence / (case + ".log"))], capture_output=True, text=True)
    assert check.returncode == 2 and run["exit_code"] == 1, (case, check.stderr)
    assert "physical admissibility is not qualified" in " ".join(log.split())
    assessment = json.loads(check.stdout)
    assert assessment == json.loads((evidence / (case + ".checks.json")).read_text())
    assert assessment["equality_converged"] and assessment["resolved_evolution"] and assessment["repeats_resolved"]
    rows = table(evidence / (case + ".controlled-history.csv"))
    nodes = table(evidence / (case + ".initialization-nodes.csv"))
    particles = table(evidence / (case + ".initialization-particles.csv"))
    assert run["seconds"] < 120 and run["child_max_rss_kib"] < 2*1024*1024
    tail = dict((k, float(v)) for k, v in re.findall(r"(\w+)=([\deE+.-]+)",
                 re.search(r"P1 tail (.*)", log)[1]))
    assert tail["clipped_residual"] > 80*tail["initialized_residual"]
    central = [r for r in nodes if r["x"] == 0.5]
    assert central and min(r["profile"] for r in nodes) >= 0
    outside = [r for r in particles if r["profile"] == 0]
    assert outside and all(r["H"] == 0.5 for r in outside)
    assert min(r["H"] for r in particles if r["profile"] > 0) < 0.02
    if case.startswith("straight"):
        assert max(r["healing"] for r in rows[2:4]) <= assessment["tolerance"]
    results[case] = dict(tail, profile_difference=float(re.search(r"profile difference=([\deE+.-]+)", log)[1]),
                        node_min=min(r["node_min"] for r in rows), cpdi_min=min(r["cpdi_min"] for r in rows),
                        max_load_healing=max(r["healing"] for r in rows[2:4]),
                        ramp_changes=[r["change"] for r in rows[2:4]],
                        uncertainty=assessment["uncertainty_estimate"], tolerance=assessment["tolerance"],
                        repeat=assessment["repeat_drift"],
                        residual_range=[min(r["final_residual"] for r in rows), max(r["final_residual"] for r in rows)],
                        target_range=[min(r["target"] for r in rows), max(r["target"] for r in rows)],
                        seconds=run["seconds"], rss_mib=run["child_max_rss_kib"]/1024,
                        central_initial_min=min(r["initialized"] for r in central),
                        physically_admissible=assessment["physically_admissible"])

# New observations must not change the original solve sequence.
assert (evidence / "finite.controlled-history.csv").read_bytes() == (repo / "doc/reconstructed_fault/propagation/P1_refined.csv").read_bytes()
print(json.dumps(dict(saved_evidence_checks_pass=True, physical_qualification=False,
                     finite_replay_bitwise_csv_match=True, cases=results), indent=2))
