#!/usr/bin/env python3
"""Run one PF2 frozen-profile control with exact local artifact identities."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import re
import shutil
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_text(path, root, postprocessor="PF2 fixed profile checks", particle_observer=True):
    """Reuse physical inputs without loading unrelated historical observers."""
    lines = []
    for line in path.read_text().replace("$ASPECT_SOURCE_DIR", str(root)).splitlines():
        if line.strip().startswith("include "):
            lines.append(fixture_text(Path(line.strip()[8:].strip()), root, postprocessor, particle_observer))
        elif re.match(r"\s*set (List of postprocessors|Additional shared libraries)\s*=", line):
            # The native parser validates each assignment, including one later
            # overridden. Old observer names are not registered in this plugin.
            if "List of postprocessors" in line:
                lines.append(f"  set List of postprocessors = {'particles, ' if particle_observer else ''}{postprocessor}")
        else:
            lines.append(line)
    return "\n".join(lines)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=Path("build-particle-free/native/aspect-release"))
    parser.add_argument("--plugin", type=Path, default=Path("build-particle-free/plugin/libfixed_profile_checks.release.so"))
    parser.add_argument("--seconds", type=float, default=180.)
    parser.add_argument("--postprocessor", default="PF2 fixed profile checks")
    parser.add_argument("--expect-failure-marker")
    parser.add_argument("--no-particle-observer", action="store_true")
    parser.add_argument("--restart-from", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    out = root / "build-particle-free/runs" / args.label
    out.mkdir(parents=True, exist_ok=False)
    binary, plugin = args.binary.resolve(), args.plugin.resolve()
    shutil.copy2(binary, out / "aspect-release")
    shutil.copy2(plugin, out / plugin.name)
    prm = out / "input.prm"
    prm.write_text(fixture_text(args.input.resolve(), root, args.postprocessor, not args.no_particle_observer)+f"\nset Additional shared libraries = {out/plugin.name}\n"
                   f"set Output directory = {out/'run'}\n")
    if args.restart_from:
        shutil.copytree(args.restart_from, out/"run/restart")
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "DEAL_II_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[key] = "1"
    tracked = subprocess.check_output(["git", "ls-files", "source", "include", "unit_tests", "CMakeLists.txt"], cwd=root, text=True).splitlines()
    # Include all new production files, including private DG implementation data.
    tracked += subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "source", "include"],
                                       cwd=root, text=True).splitlines()
    record = dict(label=args.label, command=[str(out/"aspect-release"), str(prm)], cwd=str(root),
                  head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                  binary_source=str(binary), executable_sha256=sha(binary),
                  plugin_source=str(plugin), plugin_sha256=sha(plugin),
                  input_sha256=sha(prm), fixture=str(args.input.resolve()), fixture_sha256=sha(args.input),
                  source_hashes={p: sha(root/p) for p in sorted(set(tracked)) if (root/p).is_file()},
                  budget_seconds=args.seconds, threads=1, ranks=1,
                  scope=("native mechanical steps at frozen accepted phase; no phase evolution or transport"
                         if args.postprocessor=="PF2 fixed profile checks" else args.postprocessor))
    (out/"source.patch").write_text(subprocess.check_output(["git", "diff", "--", "source", "include"], cwd=root, text=True))
    start = time.monotonic()
    with (out/"run.log").open("w") as stream:
        try:
            code = subprocess.run(record["command"], cwd=root, env=env, stdout=stream,
                                  stderr=subprocess.STDOUT, timeout=args.seconds, check=False).returncode
        except subprocess.TimeoutExpired:
            code = 124
    record.update(exit_code=code, seconds=time.monotonic()-start,
                  max_rss_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    if args.expect_failure_marker:
        record["expected_failure_marker"] = args.expect_failure_marker
        record["gate_passed"] = code not in (0, 124) and args.expect_failure_marker in (out/"run.log").read_text()
    else:
        record["gate_passed"] = code == 0
    record["outputs"] = {p.name: sha(p) for p in (out/"run").glob("*.csv")}
    (out/"record.json").write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps({k: record[k] for k in ("label", "exit_code", "seconds", "max_rss_KiB")}))
    raise SystemExit(0 if record["gate_passed"] else (code or 1))


if __name__ == "__main__":
    main()
