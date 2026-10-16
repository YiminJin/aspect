#!/usr/bin/env python3
"""Rerun the retained native-history fixture in this worktree, with a hard budget.

The historical executable/plugins are copied, never rebuilt or run in their
original checkout. This is a retained-binary baseline, not a fresh source build.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retained-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=180)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    fixture = root / "benchmarks/reconstructed_fault/bp3_native/history-uniform-q6.prm"
    artifacts = ["aspect-release", "bp3-plugin/libbp3_restore_150x50.release.so",
                 "history-plugin/libbp3_native_checks.release.so"]
    record = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "kind": "retained-binary native-history diagnostic; no coupled trajectory",
              "budget_seconds": args.seconds, "fixture": str(fixture),
              "fixture_sha256": digest(fixture), "artifacts": []}
    for name in artifacts:
        source = args.retained_build / name
        target = output / Path(name).name
        shutil.copy2(source, target)
        record["artifacts"].append({"source": str(source.resolve()), "local": str(target),
                                    "sha256": digest(target)})
    text = fixture.read_text().replace("$ASPECT_SOURCE_DIR", str(root))
    text += ("\nset Additional shared libraries = " + ", ".join(str(output / Path(p).name) for p in artifacts[1:])
             + "\nset Output directory = " + str(output / "run") + "\n")
    prm = output / "baseline.prm"
    prm.write_text(text)
    record["input_sha256"] = digest(prm)
    cache = args.retained_build / "CMakeCache.txt"
    shutil.copy2(cache, output / "retained-CMakeCache.txt")
    record["retained_cache_sha256"] = digest(cache)
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "DEAL_II_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[key] = "1"
    record["environment"] = {k: v for k, v in env.items()
                             if k.startswith(("ASPECT_", "OMP_", "DEAL_II_", "OPENBLAS_", "OMPI_"))}
    command = [str(output / "aspect-release"), str(prm)]
    record["command"] = command
    record["cwd"] = str(root)
    start = time.monotonic()
    with (output / "run.log").open("w") as stream:
        try:
            code = subprocess.run(command, cwd=root, env=env, stdout=stream,
                                  stderr=subprocess.STDOUT, timeout=args.seconds, check=False).returncode
        except subprocess.TimeoutExpired:
            code = 124
    record.update(exit_code=code, seconds=time.monotonic() - start)
    record["outputs"] = {p.name: digest(p) for p in (output / "run").glob("native-*.csv")}
    (output / "baseline.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"exit_code": code, "seconds": record["seconds"], "record": str(output / "baseline.json")}))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
