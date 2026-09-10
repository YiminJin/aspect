#!/usr/bin/env python3
"""Record the accepted working-tree source and executable without git mutation."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

root = Path(__file__).resolve().parents[4]
destination = Path(__file__).resolve().parent/"performance/accepted-cartesian-baseline"
destination.mkdir(exist_ok=False)
files = set()
for directory in ("source", "include", "unit_tests", "tests", "cmake"):
    files.update(p for p in (root/directory).rglob("*") if p.is_file())
files.update(root/p for p in ("CMakeLists.txt", "doc/reconstructed_fault/current_design.md",
                              "doc/reconstructed_fault/specification.tex"))
# Store benchmark implementation/inputs, not duplicate the large raw results.
for p in (root/"benchmarks/reconstructed_fault").rglob("*"):
    if p.is_file() and p.suffix in (".cc", ".h", ".py", ".sh", ".prm"):
        if p.name not in ("parameters.prm", "original.prm") and not any("build" in s for s in p.parts):
            files.add(p)
manifest = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(files)}
with tarfile.open(destination/"source.tar.gz", "w:gz") as archive:
    for p in sorted(files):
        archive.add(p, arcname=str(p.relative_to(root)), recursive=False)
(destination/"working-tree.patch").write_bytes(subprocess.check_output(
    ["git", "diff", "--binary", "HEAD"], cwd=root))
binary = root/"build-pf-cpdi/aspect-release"
plugin = root/"benchmarks/reconstructed_fault/uniform_shear/diagnostics/pilot-build/libuniform_shear.release.so"
for path in (binary, plugin):
    (destination/path.name).write_bytes(path.read_bytes())
report = dict(head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              files=manifest,
              artifacts={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in destination.iterdir() if p.is_file()})
(destination/"manifest.json").write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(dict(head=report["head"], source_files=len(files), artifacts=report["artifacts"])))
