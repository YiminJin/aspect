#!/usr/bin/env python3
"""Run the existing bounded coupled/retry/restart cases in isolated directories."""
from pathlib import Path
import os
import subprocess
import sys

root = Path(__file__).resolve().parent
repo = root.parents[2]
variant = sys.argv[1]
binaries = {
    "baseline": "build-refactor-r6b/aspect-birth-identity-qualified",
    "unity": "build-refactor-post-r6-unity/aspect",
    "independent": "build-refactor-post-r6-independent/aspect",
    "gcc12-unity-final": "build-refactor-post-r6-gcc12-unity/aspect",
    "gcc12-qualified-unity": "build-refactor-post-r6-gcc12-unity/aspect",
    "gcc12-unity": "build-refactor-post-r6-gcc12-unity/aspect",
    "gcc12-independent": "build-refactor-post-r6-gcc12-independent/aspect",
}
binary = repo / binaries[variant]
environment = {key: value for key, value in os.environ.items()
               if not key.startswith("ASPECT_")}
environment["ASPECT_SOURCE_DIR"] = str(repo)
for case, ranks in [("direct", 1), ("retry", 1), ("staggered", 2),
                    ("create", 2), ("resume", 2), ("direct2", 2), ("retry2", 2)]:
    if len(sys.argv) > 2 and case not in sys.argv[2:]:
        continue
    destination = root / f"output-{variant}-{case}"
    assert not destination.exists(), destination
    if case in ("resume", "direct2", "retry2"):
        subprocess.run(["bash", str(root.parent / "bp3/branch_output.sh"),
                        str(root / f"output-{variant}-create"), "2", str(destination)],
                       check=True)
    subprocess.run([sys.executable, str(root / "run_logged.py"),
                    f"{variant}-{case}", "300", "mpirun", "-np", str(ranks),
                    str(binary), str(root / "inputs" / f"{variant}-{case}.prm")],
                   cwd=repo, env=environment, check=True)
