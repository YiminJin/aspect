"""Check the discarded interval, accepted histories, and clean replay of AMR.

Use run_advancing.py for the compact case. Its uniform control starts from
the original input on the final mesh. The historic 18-strip directories
written by recovery.py can also be checked.
"""
import argparse
import filecmp
import json
from pathlib import Path
import re

from trajectory import Case, check_interval, compare, field_names


def check_compact_parent(parent, reference):
    """Check both initial fields where identical and zero history on new fine cells."""
    import numpy as np

    old, old_fe, sections, _ = parent.load(0)
    new, new_fe, clean_sections, _ = reference.load(0)
    for name in sections:
        np.testing.assert_array_equal(clean_sections[name], sections[name])
    common = set(old["cell"]) & set(new["cell"])
    assert common
    for left, right in ((old, new), (old_fe, new_fe)):
        for name in left:
            np.testing.assert_array_equal(left[name][np.isin(left["cell"], list(common))],
                                          right[name][np.isin(right["cell"], list(common))])
    coarse = set(old["cell"])-common
    fine = set(new["cell"])-common
    assert coarse and fine
    assert {cid.split("_")[0] for cid in coarse} == {cid.split("_")[0] for cid in fine}
    for frame, changed in ((old, coarse), (new, fine)):
        select = np.isin(frame["cell"], list(changed))
        assert np.all(frame["x"][select] > 1.25)
        for name in ("phi", "tauxx", "tauyy", "tauzz", "tauxy", "Axx", "Ayy", "Azz", "Axy",
                     "signed_shear", "absolute_shear"):
            np.testing.assert_array_equal(frame[name][select], 0.)
    return len(coarse), len(fine)


def check(missed, clean, ranks, cut, steps, compact=False):
    log = (missed / "timing" / "solver.log").read_text()
    if log.count("Growing fault missed extension: refine and repeat") != 1:
        raise ValueError("Expected exactly one rejected mesh prediction")
    if log.count("Repeating the current time step") != 1 or log.count("*** Timestep 1:") != 2:
        raise ValueError("Expected exactly one replay of the first physical solve")
    if ("retry_reason=slip" in log) != cut:
        raise ValueError("Slip-budget rejection differs from the requested control")
    parent, reference = Case(missed / "output"), Case(clean / "output")
    for case in (parent, reference):
        if sorted(p.name for p in case.folder.glob("balance-*.csv")) != [
                f"balance-{i:04}.csv" for i in range(1, steps+1)]:
            raise ValueError("Expected exactly one work record per accepted physical interval")
        for step in range(steps+1):
            for kind in ("points", "fields"):
                for rank in range(ranks):
                    with (case.folder / f"{kind}-{step:04}-r{rank}.csv").open() as shard:
                        if not shard.readline() or not shard.readline():
                            raise ValueError(f"Empty {kind} on rank {rank} at step {step}")
    changed = check_compact_parent(parent, reference) if compact else None
    accepted = parent.load(1)
    clean_frame = reference.load(1)
    compare(clean_frame, accepted, field_names(parent.folder), strict=True)
    dt = float(accepted[3]["time"][0])
    initial_dt = .0025 if compact else .005
    if not (0. < dt < initial_dt if cut else dt == initial_dt):
        raise ValueError(f"Unexpected accepted duration {dt}")
    check_interval(parent, parent.load(0)[2], accepted[2], dt)
    check_interval(reference, reference.load(0)[2], clean_frame[2], dt)
    if steps == 2:
        next_frame = parent.load(2)
        compare(reference.load(2), next_frame, field_names(parent.folder), strict=True)
        check_interval(parent, accepted[2], next_frame[2], next_frame[3]["time"][0]-dt)
    physical_tables = [path for i in range(1, steps+1)
                       for path in parent.folder.glob(f"*-{i:04}*.csv")]
    count = re.search(r"connected coarse cells (\d+)", log)
    if count:
        exposure = dict(connected_coarse_cells=int(count[1]))
    else:
        exposure = dict(new_coarse_samples=int(re.search(r"newly crossed coarse samples (\d+)", log)[1]))
    return dict(ranks=ranks, first_dt_s=initial_dt, accepted_dt_s=dt, **exposure,
                initial_intact_parent_and_children=changed,
                accepted_cells=len(set(accepted[0]["cell"])), accepted_points=len(accepted[0]["cell"]),
                active_rows=len(accepted[2]["section"]), mesh_repeats=1, slip_cuts=int(cut),
                accepted_intervals=steps, relative_and_absolute_tolerance=1e-12,
                accepted_files=len(physical_tables),
                byte_identical_files=sum(filecmp.cmp(p, reference.folder/p.name, shallow=False)
                                          for p in physical_tables))


def check_mesh_passes(two_levels, blocked):
    """No intervening nonlinear solve while building levels. No publication on failure."""
    log = (two_levels/"timing/solver.log").read_text()
    before, after = log.split("Refining the mesh based on the time stepping manager")
    passes, retried = after.split("Repeating the current time step")
    cells = [int(x) for x in re.findall(r"Number of active cells: (\d+)", passes)]
    if (cells != [32, 128] or "Midpoint iteration" in passes or "*** Timestep" in passes or
            "missed extension" not in before or "*** Timestep 1:" not in retried):
        raise ValueError("Expected two consecutive mesh-only passes before the replay")
    if not (two_levels/"output/balance-0001.csv").exists():
        raise ValueError("The refined calculation did not publish the first physical interval")
    failure = (blocked/"timing/solver.log").read_text()
    status = json.loads((blocked/"timing/exit.json").read_text())["status"]
    if (status == 0 or "requested refinement target remains unresolved" not in failure or
            list((blocked/"output").glob("balance-*.csv"))):
        raise ValueError("A blocked refinement must stop before publication with a resolution diagnostic")
    return dict(cells_after_pass=cells, blocked_status=status, blocked_accepted_intervals=0)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("missed", type=Path)
    args.add_argument("clean", type=Path)
    args.add_argument("--ranks", type=int, required=True)
    args.add_argument("--cut", action="store_true")
    args.add_argument("--steps", type=int, choices=(1,2), default=1)
    args.add_argument("--compact", action="store_true", help="Check the initially intact compact-seed receiving cells")
    args.add_argument("--resumed", type=Path, help="Run resumed from the accepted parent before this miss")
    args.add_argument("--mesh-guards", type=Path, nargs=2, metavar=("TWO_LEVELS","BLOCKED"))
    opts = args.parse_args()
    result = check(opts.missed, opts.clean, opts.ranks, opts.cut, opts.steps, opts.compact)
    if opts.resumed:
        def reactions(folder):
            return [line for line in (folder/"timing/solver.log").read_text().splitlines()
                    if line.startswith(("Growing fault missed extension:", "Growing fault preventive refinement:",
                                        "Growing fault unresolved accepted damage:", "Repeating the current time step"))]
        resumed = Case(opts.resumed/"output")
        original = Case(opts.missed/"output")
        if reactions(opts.resumed) != reactions(opts.missed):
            raise ValueError("Restart changed the physical tip, rejection clock, coarse exposure or repeat decision")
        for step in range(1, opts.steps+1):
            compare(original.load(step), resumed.load(step), field_names(original.folder), strict=True)
        result["resumed_decisions"] = reactions(opts.resumed)
    if opts.mesh_guards:
        result["mesh_passes"] = check_mesh_passes(*opts.mesh_guards)
    print(json.dumps(result, indent=2))
