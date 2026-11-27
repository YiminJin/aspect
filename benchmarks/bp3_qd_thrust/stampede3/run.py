"""Start or continue the production case within a Stampede3 Slurm allocation."""
import argparse
import fcntl
import hashlib
import json
import uuid
import shutil
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from startup import OUTPUT, VERTICES, checkpoint, require, verify

PARAMETERS = Path("production.prm")
SOLVER = Path("plugins/aspect-release")
PLUGIN = Path("plugins/libbp3_qd.release.so")
STOP = OUTPUT / "terminate-aspect"
COMPLETE = OUTPUT / "bp3-complete.txt"
MARGIN_SECONDS = 1800


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def end_epoch(job):
    epoch = os.environ.get("SLURM_JOB_END_TIME", "")
    if epoch.isdigit() and int(epoch) > 0:
        return int(epoch)
    stamp = subprocess.check_output(["squeue", "-h", "-j", job, "-o", "%e"], text=True).strip()
    require(stamp and stamp not in ("N/A", "Unknown"), "Cannot determine the Slurm allocation end time")
    return int(subprocess.check_output(["date", "-d", stamp, "+%s"], text=True))


def input_text(original, resume, target, hours):
    # Preserve the user's production settings. Extend only the lists needed for
    # this allocation, and save a complete audit copy for each solver invocation.
    lines = []
    sections = []
    managed = {"Output directory": str(OUTPUT), "Additional shared libraries": "./plugins/libbp3_qd.so",
               "Material model/Phase field fault/Fault vertex file": str(VERTICES)}
    for line in original.splitlines():
        body = line.partition("#")[0].strip()
        if body.startswith("subsection "):
            sections.append(body[len("subsection "):])
        elif body == "end":
            sections.pop()
        elif body.startswith("set "):
            key, value = (s.strip() for s in body[4:].split("=", 1))
            path = "/".join(sections + [key])
            if path in managed:
                require(value == managed[path], f"{path} must point to the file or directory tracked by this launcher")
            if path == "Resume computation":
                value = "true" if resume else "false"
            elif path == "Termination criteria/Termination criteria":
                value += ", wall time" + (", end step" if target else "")
            elif target and path == "Postprocess/List of postprocessors":
                value += ", BP3 startup cuts"
            elif target and path in ("Postprocess/BP3 profiles/Time interval",
                                    "Postprocess/Visualization/Time between graphical output"):
                value = "0"
            else:
                lines.append(line)
                continue
            line = "  " * len(sections) + f"set {key} = {value}"
        lines.append(line)
    lines += ["", "subsection Termination criteria", f"  set Wall time = {hours:.10f}"]
    if target:
        lines.append(f"  set End step = {target}")
    return "\n".join(lines + ["end", ""])


def launch(directory, original, resume, target, deadline):
    require(not STOP.exists(), f"Stop requested in {STOP}")
    seconds = deadline - time.time() - MARGIN_SECONDS
    if seconds < 1:
        print("Checkpoint margin reached before the next launch. End of this allocation.")
        return {"wall time"}
    phase = f"step-{target}" if target else "production"
    path = directory / f"{phase}-input.txt"
    path.write_text(input_text(original, resume, target, seconds / 3600))
    log = directory / f"{phase}.log"
    print(f"{phase}: resume={resume}, input={path}, solver log={log}", flush=True)
    with log.open("w") as stream:
        subprocess.run(["ibrun", str(SOLVER.resolve(strict=True)), str(path)], stdout=stream,
                       stderr=subprocess.STDOUT, check=True)
    shutil.copyfile(OUTPUT / "parameters.json", directory / f"{phase}-effective.json")
    prefix = "Termination requested by criterion: "
    with log.open() as stream:
        reasons = {line.partition(prefix)[2].strip() for line in stream if prefix in line}
    require(reasons, f"ASPECT exited without a recorded termination criterion. See {log}")
    print("ASPECT stopped on " + ", ".join(sorted(reasons)))
    checkpoint()
    return reasons


def ended(reasons):
    if reasons & {"end time", "BP3 first event complete"}:
        COMPLETE.write_text(", ".join(sorted(reasons)) + "\n")
        print(f"Production target reached. See {COMPLETE}.")
        return True
    return STOP.exists() or bool(reasons & {"wall time", "user request"})


def run(mode):
    job = os.environ["SLURM_JOB_ID"]
    deadline = end_epoch(job)
    require(not COMPLETE.exists(), f"The simulation is already complete. See {COMPLETE}")
    require(not STOP.exists(), f"Stop request is present: {STOP}. Move it aside to authorize a resume.")
    require(os.access(SOLVER, os.X_OK) and PLUGIN.is_file(), "Build the matching Release plugin in plugins/ first")
    require(mode != "start" or not OUTPUT.exists(), f"Fresh start requires an unused {OUTPUT}")
    next_step = checkpoint()[0] if mode == "resume" else 0
    if mode == "resume":
        print(f"Latest published checkpoint starts at step {next_step}")
    original = PARAMETERS.read_text()
    Path("jobs").mkdir(exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f"{job}-", dir="jobs"))
    print(f"Run directory {Path.cwd()}, allocation {job}, "
          f"nodes={os.environ.get('SLURM_JOB_NUM_NODES')}, tasks={os.environ.get('SLURM_NTASKS')}")
    print(f"Release executable {SOLVER.resolve(strict=True)}")
    inputs = (SOLVER, PLUGIN, VERTICES, PARAMETERS)
    provenance = "\n".join(f"{file_digest(p)}  {p}" for p in inputs) + "\n"
    (directory / "inputs.sha256").write_text(provenance)
    print(f"Input and executable hashes: {directory/'inputs.sha256'}")
    require(file_digest(VERTICES) == "710623919be54bd508ae87a8cd58d064742f252ca83fb35c110a04918f1c346a",
            "Vertex table differs from the historical production table")
    # Bind success markers and continuations to the original input, SDK binary
    # and loaded extension. Refuse a legacy or independently copied checkpoint.
    manifest = OUTPUT / "bp3-run.json"
    ledger = Path("jobs") / "bp3-run.json"
    if mode == "start":
        require(not ledger.exists(), "An earlier run owns jobs/. Preserve the old package directory.")
        OUTPUT.mkdir()
        identity = {"id": uuid.uuid4().hex, "inputs": provenance}
        for path in (manifest, ledger):
            with path.open("x") as out:
                json.dump(identity, out, indent=2)
    else:
        identity = json.loads(manifest.read_text())
        require(identity == json.loads(ledger.read_text()) and identity["inputs"] == provenance,
                "Checkpoint, startup markers or input/build identity belong to a different BP3 run")
    run_id = identity["id"]
    layout = {k: os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_JOB_NUM_NODES", "SLURM_NTASKS",
              "SLURM_TASKS_PER_NODE", "DEAL_II_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    layout.update(run_id=run_id, solver=str(SOLVER.resolve()), plugin=str(PLUGIN.resolve()),
                  vertex=str(VERTICES.resolve()), input=str(PARAMETERS.resolve()))
    shutil.copyfile(PLUGIN.parent/"CMakeCache.txt", directory/"plugin-CMakeCache.txt")
    (directory/"launch.json").write_text(json.dumps(layout, indent=2)+"\n")
    resume = mode == "resume"
    for target in (1, 3):
        passed = OUTPUT / f"startup-step-{target}.ok"
        if passed.exists():
            require(passed.read_text().split()[0] == run_id, f"Foreign startup marker {passed}")
            require(next_step >= target + 1, f"Checkpoint predates {passed}")
            continue
        require(next_step <= target + 1, f"Checkpoint is ahead of the unverified startup step {target}")
        reasons = set()
        if next_step <= target:
            reasons = launch(directory, original, resume, target, deadline)
            if "end step" not in reasons:
                if ended(reasons):
                    return
                raise ValueError(f"Missing end-step termination for step {target}")
        verify(target)
        next_step = target + 1
        passed.write_text(f"{run_id} checked by allocation {job}, see {directory}\n")
        print(f"Verified accepted steps 0..{target} and the terminal checkpoint.", flush=True)
        if ended(reasons):
            return
        resume = True
    reasons = launch(directory, original, True, None, deadline)
    require(ended(reasons), f"Unexpected production termination: {reasons}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("start", "resume"), default="start")
    args = parser.parse_args()
    try:
        with Path(".launch.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                sys.exit("Another allocation owns this run directory")
            run(args.mode)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        sys.exit(f"BP3 launch stopped: {error}")
