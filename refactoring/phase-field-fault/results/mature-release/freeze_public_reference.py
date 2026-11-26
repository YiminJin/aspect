"""Archive one serial public reference into a new directory; this is separate from verification."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


REVISION = "ae676e24095909196f885073b1146babce50292b"
CASE = Path("benchmarks/fixed_fault/curved_mature")
INPUTS = (CASE/"curved_mature.prm", CASE/"vertices.txt", Path("benchmarks/fixed_fault/fixed_fault.prm"),
          Path("benchmarks/layered_moving_fault/layered_moving_fault.cc"))
STEPS = range(3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prefix", "observer", "run", "destination"):
        parser.add_argument(name, type=lambda s: Path(s).resolve())
    args = parser.parse_args()
    if args.destination.exists():
        raise FileExistsError(f"Reference destination already exists: {args.destination}")
    observed = args.run/"observed/output"
    public = args.run/"public/output"
    digests = {}
    for path in INPUTS:
        baseline = subprocess.check_output(["git", "show", f"{REVISION}:{path}"])
        if (args.prefix/path).read_bytes() != baseline:
            raise ValueError(f"Installed {path} differs from {REVISION}")
        digests[str(path)] = hashlib.sha256(baseline).hexdigest()
    for path in (args.prefix/"bin/aspect-release", args.observer):
        digests[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

    times = np.loadtxt(observed/"statistics", usecols=(0, 1, 2), ndmin=2)
    np.testing.assert_allclose(times, [[0, 0, 0], [1, .0125, .0125], [2, .025, .0125]], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(times, np.loadtxt(public/"statistics", usecols=(0, 1, 2), ndmin=2))
    for folder in (public, observed):
        log = (folder/"log.txt").read_text()
        if REVISION[:9]+")" not in log or "running with 1 MPI process\n" not in log:
            raise ValueError(f"Expected the serial {REVISION[:9]} run in {folder}")
        wrapper = [f"include {args.prefix/CASE/'curved_mature.prm'}", "set Output directory = output"]
        if folder == observed:
            wrapper += [f"set Additional shared libraries = {args.observer}", "subsection Postprocess",
                        "set List of postprocessors = mature fault statistics, moving fault samples", "end"]
        lines = [s.strip() for s in (folder.parent/"case.prm").read_text().splitlines() if s.strip()]
        if lines != wrapper:
            raise ValueError(f"Unexpected parameters in {folder.parent/'case.prm'}")
    files = {"clocks.csv": ("step,time_s,dt_s\n" +
                            "".join(f"{int(n)},{t:.17g},{h:.17g}\n" for n, t, h in times)).encode()}
    for step in STEPS:
        filename = f"fault-labels-{step:05}.txt"
        files[filename] = (public/filename).read_bytes()
        if files[filename] != (observed/filename).read_bytes():
            raise ValueError(f"Public and observed {filename} differ")
        filename = f"material-{step}-surface-0.csv"
        files[filename] = (observed/filename).read_bytes()
        filename = f"material-{step}-rank0.csv"
        files[filename+".gz"] = gzip.compress((observed/filename).read_bytes(), compresslevel=9, mtime=0)
    manifest = dict(source_revision=REVISION, mpi_ranks=1, numerical_threads=1,
                    inputs_and_binaries_sha256=digests,
                    files_sha256={k: hashlib.sha256(v).hexdigest() for k, v in files.items()})
    files["provenance.json"] = (json.dumps(manifest, indent=2)+"\n").encode()
    args.destination.mkdir()
    for name, contents in files.items():
        (args.destination/name).write_bytes(contents)
    print(f"Archived {len(times)} frames from {REVISION} in {args.destination}")


if __name__ == "__main__":
    main()
