"""Replay loaded strip remeshing, migrated checkpoints and a pre-AMR uniform checkpoint."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from check_remesh_history import check, read_snapshot

import numpy as np

GROWING = Path(__file__).resolve().parents[1]/"growing_fault"
sys.path.insert(0, str(GROWING))
from compare import allowance, continuation, plane_tolerances, rows


def run(args, name, input_text, ranks, *, source=None, use_observer=True, checkpoint=None,
        expected_failure=None):
    case = args.destination/name
    case.mkdir()
    output = case/"output"
    shutil.copyfile(GROWING/"growing_seed.prm", case/"growing_seed.prm")
    shutil.copyfile(GROWING/"loaded_remesh.prm", case/"loaded_remesh.prm")
    shutil.copyfile(GROWING/"onecut.prm", case/"onecut.prm")
    if use_observer:
        input_text += f"\nset Additional shared libraries = {args.plugin}\n"
    if checkpoint:
        shutil.copytree(checkpoint, output/"restart"/checkpoint.name)
        input_text += f"\nset Resume computation = true\nsubsection Checkpointing\n set Resume checkpoint = {int(checkpoint.name)}\nend\n"
    (case/"input.prm").write_text(input_text+"\nset Output directory = output\n")
    env = dict(os.environ, OMP_NUM_THREADS="1", DEAL_II_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    rc = subprocess.run(["mpiexec", "-n", str(ranks), source or args.aspect, "input.prm"],
                        cwd=case, env=env, stdout=(case/"log").open("w"), stderr=subprocess.STDOUT)
    if expected_failure:
        if rc.returncode == 0 or expected_failure not in (case/"log").read_text():
            raise RuntimeError(f"Missing expected failure {expected_failure!r}: {case}/log")
    elif rc.returncode:
        raise RuntimeError(f"ASPECT failed: {case}/log")
    return output


def restored(source, destination, writer, reader):
    """Bitwise-by-value comparison before the first resumed nonlinear solve."""
    before, _, meta, ledger = read_snapshot(source,"remesh-1-1-after",writer)
    after, _, meta1, ledger1 = read_snapshot(destination,"remesh-1-0-restored",reader)
    assert sorted(before) == sorted(after) and meta == meta1 and ledger == ledger1
    moved = 0
    for cell, (state, owner) in before.items():
        received, other = after[cell]
        np.testing.assert_array_equal(state, received)
        moved += int(owner != other)
    assert writer == reader or moved > 0
    return moved


def trajectory(a,b,steps):
    """Report maxima, including every discrepancy in the established continuation budget."""
    result = {}
    for step in steps:
        failures = []
        angle_and_length_failures = []
        maximum = {}
        for pattern,keys in [(f"fields-{step:04d}-r*.csv", ("cell", "component", "local")),
                             (f"points-{step:04d}-r*.csv", ("cell", "q")),
                             (f"sections-{step:04d}.csv", ("section",)),
                             (f"balance-{step:04d}.csv", ("step",))]:
            left, right = [rows(src.glob(pattern), keys) for src in (a,b)]
            assert left.keys()==right.keys()
            ids = list(left)
            for col in left[ids[0]]:
                if col in keys:
                    continue
                # The FE block contains fields with different units.
                groups = {col:ids} if col != 'value' else {
                    f"field_{i}":[k for k in ids if int(k[1]) == i] for i in sorted({int(k[1]) for k in ids})}
                for name, group in groups.items():
                    l,r = (np.array([float(t[k][col]) for k in group]) for t in (left,right))
                    assert np.isfinite(l).all() and np.isfinite(r).all()
                    error = abs(r-l)
                    budget = 1e-12+1e-12*abs(l)
                    maximum[name] = float(max(error))
                    if col in ("beta","ds"):
                        physical_budget = allowance(l,*plane_tolerances(col))
                        for i in np.flatnonzero(error>physical_budget):
                            angle_and_length_failures.append(dict(column=col,strip=group[i][0],
                                                                  difference=float(error[i]),budget=float(physical_budget[i])))
                    bad = np.flatnonzero(error>budget)
                    if len(bad):
                        i=int(bad[0])
                        failures.append(dict(column=name,key=group[i],difference=float(error[i]),budget=float(budget[i])))
        result[str(step)] = dict(maximum=maximum,strict_failures=failures,
                                 angle_and_length_failures=angle_and_length_failures)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("aspect","plugin","pre_amr_aspect","destination"):
        p.add_argument(name, type=lambda x:Path(x).resolve())
    p.add_argument("--parents",type=lambda x:Path(x).resolve(),
                   help="Reuse completed serial, parallel and stateless parent runs for restart checks.")
    p.add_argument("--v1-generation", type=lambda x:Path(x).resolve(),
                   help="Optional original RSF v1 onecut checkpoint generation 03")
    args=p.parse_args()
    assert args.aspect.read_bytes() != args.pre_amr_aspect.read_bytes(), "Supply separate writer and reader binaries."
    args.destination.mkdir(parents=True)
    base = "include loaded_remesh.prm\n"
    state_free = ("subsection Material model\n set Friction normal stress = prescribed constant\n"
                  " subsection Phase field fault\n  set Resistance law = linear dashpot\n end\n"
                  " subsection Growing fault\n  set Prescribed fault compression = 200\n end\nend\n")
    outputs = {}
    report = {}
    def save():
        (args.destination/"measurements.json").write_text(json.dumps(report,indent=2)+"\n")
    stop_after_transfer = "subsection Termination criteria\n set Termination criteria = end step\n set End step = 1\nend\n"
    for name, nranks, extra in (
        ("serial",1,""),("parallel",2,""),("stateless",1,state_free),
        ("finer",2,"subsection Geometry model\n subsection Box\n set X repetitions = 16\n set Y repetitions = 8\n end\nend\n"+stop_after_transfer),
        ("gauss-four",1,"subsection Material model\n subsection Growing fault\n set Section quadrature order = 4\n end\nend\n"+stop_after_transfer)):
        if name in ("serial","parallel","stateless") and args.parents:
            case = args.parents/name
            assert "Total wallclock time elapsed" in (case/"log").read_text()
            outputs[name] = case/"output"
        else:
            outputs[name] = run(args,name,base+extra,nranks)
        result = report[name] = check(outputs[name],1,1,nranks)
        state_count = 0 if name == "stateless" else result["strip_count"]
        assert result["law_state_count"] == state_count and result["pressure_station_count"] == state_count
        if nranks > 1:
            assert result["cells_migrated"] > 0
        save()
    report["serial_vs_parallel"] = trajectory(outputs["serial"],outputs["parallel"],(1,2,3))
    for name, parent, writer, reader in (("same","serial",1,1), ("to-two","serial",1,2),
                                         ("to-one","parallel",2,1), ("free-to-two","stateless",1,2)):
        source = outputs[parent]
        out = run(args,name,base+(state_free if parent=="stateless" else ""),reader,checkpoint=source/"restart"/"02")
        moved = restored(source,out,writer,reader)
        if reader == writer:
            continuation(source,out,(2,3))
        physical = trajectory(source,out,(2,3))
        report[name] = dict(restored_cells_on_new_owners=moved, physical_endpoints=physical)
        save()
        assert all(not t["angle_and_length_failures"] for t in physical.values()), name
    for name,setting,checkpoint,diagnostic in (
        ("underintegrated","Section quadrature order = 3",None,"requires Section quadrature order >= 4"),
        ("changed-width","Target strip width = 0.124",outputs["serial"]/"restart"/"02",
         "Target strip width differs from the adaptive growing fault checkpoint")):
        run(args,name,base+f"subsection Material model\n subsection Growing fault\n  set {setting}\n end\nend\n",
            1,checkpoint=checkpoint,expected_failure=diagnostic)
        report[name] = diagnostic
        save()
    decks = [("v2","growing_seed.prm",None)]
    if args.v1_generation:
        decks.append(("v1","onecut.prm",args.v1_generation))
    for version,deck,parent in decks:
        command = f"include {deck}\nset End time = .015\nsubsection Checkpointing\nset Steps between checkpoint = 1\nset Number of checkpoints to keep = 4\nend\nsubsection Postprocess\nset List of postprocessors = growing fault history\nend\n"
        reference = run(args,version+"-old",command,1,source=args.pre_amr_aspect,use_observer=False,checkpoint=parent)
        start = parent or reference/"restart"/"02"
        candidate = run(args,version+"-new",command,1,use_observer=False,checkpoint=start)
        steps = (3,) if parent else (2,3)
        continuation(reference,candidate,steps)
        report[version] = trajectory(reference,candidate,steps)
        save()

if __name__ == "__main__":
    main()
