#!/usr/bin/env python3
"""Run one bounded K1 case, retaining exit status, resource use and provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parameter", type=Path)
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--configuration", choices=("Debug", "Release"), default="Debug")
    parser.add_argument("--ranks", type=int, default=1)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    parameter = args.parameter.resolve()
    executable = root / ("build-pf-cpdi/aspect-"+args.configuration.lower())
    build = root / "benchmarks/reconstructed_fault/uniform_shear/diagnostics/pilot-build"
    prefix = parameter.with_suffix("")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    provenance = dict(command=[str(executable), str(parameter)], cwd=str(build),
                      executable_sha256=digest(executable), parameter_sha256=digest(parameter),
                      configuration=args.configuration,
                      plugin_sha256=digest(build / ("libuniform_shear."+args.configuration.lower()+".so")),
                      git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip())
    if args.ranks < 1:
        parser.error("--ranks must be positive")
    if args.ranks > 1:
        provenance["command"] = ["mpirun", "-np", str(args.ranks)] + provenance["command"]
    provenance["ranks"] = args.ranks
    start = time.monotonic()
    with prefix.with_suffix(".log").open("w") as output:
        try:
            status = subprocess.run(provenance["command"], cwd=build, stdout=output,
                                    stderr=subprocess.STDOUT, timeout=args.timeout).returncode
        except subprocess.TimeoutExpired:
            status = 124
    provenance.update(exit_status=status, wall_seconds=time.monotonic()-start,
                      peak_rss_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    prefix.with_suffix(".resources.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(json.dumps(provenance), flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
