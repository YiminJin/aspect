"""Validate and run a granite input, recording wall times and per-rank memory."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def run_rank(record, command):
    """Sample Linux's per-process high-water mark even if MPI aborts all ranks."""
    rank = os.environ["OMPI_COMM_WORLD_RANK"]
    rss_file = record / f"rank-{rank}.json"
    peak = 0
    with subprocess.Popen(command) as solver:
        while solver.poll() is None:
            try:
                lines = Path(f"/proc/{solver.pid}/status").read_text().splitlines()
            except FileNotFoundError:
                break
            for line in lines:
                if line.startswith("VmHWM:"):
                    value = int(line.split()[1])
                    if value > peak:
                        peak = value
                        rss_file.write_text(json.dumps(dict(sampled_peak_KiB=peak)) + "\n")
                    break
            time.sleep(0.1)
        status = solver.wait()
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    rss_file.write_text(json.dumps(dict(peak_KiB=usage.ru_maxrss, user_s=usage.ru_utime,
                                        system_s=usage.ru_stime)) + "\n")
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("input", type=Path)
    parser.add_argument("name", help="New timing directory beside the input")
    parser.add_argument("--ranks", type=int, default=4, choices=range(1, 5))
    args = parser.parse_args()
    exe = args.executable.resolve(strict=True)
    prm = args.input.resolve(strict=True)
    record = prm.parent / args.name
    record.mkdir()
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", DEAL_II_NUM_THREADS="1", BLIS_NUM_THREADS="1")
    with (record / "validate.log").open("w") as out:
        subprocess.run([exe, "--validate", prm.name], cwd=prm.parent, env=env,
                       stdout=out, stderr=subprocess.STDOUT, check=True)
    cmd = ["mpiexec", "-n", str(args.ranks), sys.executable, str(Path(__file__).resolve()),
           "--rank", str(record), str(exe), prm.name]
    (record / "command.json").write_text(json.dumps(cmd) + "\n")
    start = time.monotonic()
    with (record / "solver.log").open("w", buffering=1) as log, \
         (record / "wall.log").open("w", buffering=1) as times:
        with subprocess.Popen(cmd, cwd=prm.parent, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, errors="replace") as solver:
            for line in solver.stdout:
                log.write(line)
                if line.strip():
                    times.write(f"{time.monotonic()-start:.6f} {line}")
            status = solver.wait()
    (record / "exit.json").write_text(json.dumps(dict(status=status, wall_s=time.monotonic()-start)) + "\n")
    return status


if __name__ == "__main__":
    if sys.argv[1:2] == ["--rank"]:
        sys.exit(run_rank(Path(sys.argv[2]), sys.argv[3:]))
    sys.exit(main())
