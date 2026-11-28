"""Exercise refinement admission and compare loaded material strips across two bulk targets."""
import argparse
import json
from pathlib import Path
import shutil
import struct
import sys
import zlib

import numpy as np

benchmarks = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(benchmarks/"granite_fault_growth"))
from run_transfer_cases import check, read_snapshot, restored, run, subsection


def fault(setting):
    return subsection("Material model", subsection("Phase field fault", "set "+setting))


def mesh(setting):
    return subsection("Mesh refinement", setting)


def marker(setting):
    return mesh(subsection("Growing fault", "set "+setting))


def check_admission(args):
    # The reaction-only case has no initial refinement, scheduled subdivision
    # or additional refinement time; all roots already meet the valid target.
    base = ("include growing_seed.prm\n"+fault("Target strip width = 0.25")
            +mesh("set Strategy = growing fault\nset Coarsening fraction = 0"))
    reaction = subsection("Time stepping", "set List of model names = growing fault refinement")
    result = {}
    for label, activation in (("reaction", reaction),
                              ("scheduled", mesh("set Time steps between mesh refinement = 2"))):
        for case, change, message in (
                ("spacing", fault("Target strip width = 0"), "requires a positive Target strip width"),
                ("order", fault("Section quadrature order = 3"), "requires Section quadrature order >= 4"),
                ("coarsen", mesh("set Coarsening fraction = 0.1"), "requires zero Coarsening fraction"),
                ("size", marker("Maximum cell edge = 0.124"), "Maximum cell edge is unreachable")):
            name = label+"-"+case
            output = run(args, name, base+activation+change, 1, use_observer=False, expected_failure=message)
            assert not list(output.glob("sections-*.csv")), name
            result[name] = message
    for name, input_text, message in (
            ("missing-marker", base+reaction+mesh("set Strategy = temperature"),
             "requires the growing fault mesh refinement criterion"),
            ("empty-reserve", base+marker("Tip disk replenishment margin = 0"),
             "must cover the direction-search radius"),
            ("wrong-material", (benchmarks/"phase_field_surface_energy"/"affine.prm").read_text()+
             mesh("set Strategy = growing fault"), "requires the two-dimensional midpoint strip material"),
            ("blocked-pass", base+mesh(
                "set Initial adaptive refinement = 1\nset Skip solvers on initial refinement = true\n"
                "set Strategy = growing fault, maximum refinement function\n" +
                subsection("Maximum refinement function", "set Function expression = 0"))+
             marker("Maximum cell edge = .125"), "target remains unresolved on the initialized mesh")):
        output = run(args, name, input_text, 1, use_observer=False, expected_failure=message)
        assert not list(output.glob("sections-0001.csv")), name
        result[name] = message
    return result


def material_rows(output, stage, ranks):
    cells, _, meta, _ = read_snapshot(output, stage, ranks)
    edges = np.array([float(v.split()[1]) for v in meta if v.startswith("edge ")])
    # Independently project the 2 m x 1 m domain onto the thirty-degree seed.
    span = 2.*np.cos(np.pi/6)+np.sin(np.pi/6)
    np.testing.assert_allclose(edges, np.linspace(0., span, 19), atol=1e-14, rtol=0.)
    slips = np.array([float(v.split()[3]) for v in meta if v.startswith("strip ")])
    ages = np.array([float(v.split()[1]) for v in meta if v.startswith("state ")])
    assert len(slips) == len(ages) == 18 and np.ptp(ages) > 0. and np.ptp(slips) > 0.
    if not stage.endswith("restored"):
        step = int(next(v for v in meta if v.startswith("clock ")).split()[1])
        published = np.sort(np.genfromtxt(output/f"sections-{step:04d}.csv", names=True, delimiter=","),
                            order="section")
        np.testing.assert_array_equal(published["section"], np.arange(len(slips)))
        np.testing.assert_array_equal(published["age"], ages)
        np.testing.assert_array_equal(published["slip"], slips)
        half_widths = .5*published["ds"]*np.cos(published["beta"])
        np.testing.assert_allclose(published["xi"]-half_widths, edges[:-1], atol=1e-14, rtol=0.)
        np.testing.assert_allclose(published["xi"]+half_widths, edges[1:], atol=1e-14, rtol=0.)
    return len(cells), np.column_stack((edges[:-1], edges[1:], ages, slips))


def check_loaded(args):
    base = "include loaded_remesh.prm\n"
    parent = run(args, "parent", base, args.ranks)
    transfer = check(parent, 1, 1, args.ranks)
    generation = parent/"restart"/"02"
    endpoint = ["remesh-2-1-"+stage for stage in ("before", "after")]
    results = {}
    old_count, old_history = material_rows(parent, "remesh-1-1-after", args.ranks)
    # At 10 ms, the scheduled pass acts on the same loaded, checkpointed mesh
    # in both continuations. Only the maximum bulk edge differs in their input.
    resumes = (base+mesh("set Strategy = growing fault\nset Time steps between mesh refinement = 2")
               +subsection("Termination criteria", "set Termination criteria = end step\nset End step = 2"))
    material_at_ten = []
    for name, spacing in (("coarse", ".25"), ("fine", ".125")):
        out = run(args, name, resumes+marker("Maximum cell edge = "+spacing),
                  args.ranks, checkpoint=generation)
        restored(parent, out, args.ranks, args.ranks)
        count, history = material_rows(out, "remesh-1-0-restored", args.ranks)
        assert count == old_count
        np.testing.assert_array_equal(history, old_history)
        before, incoming = material_rows(out, endpoint[0], args.ranks)
        if (out/(endpoint[1]+"-r0.txt")).exists():
            after, outgoing = material_rows(out, endpoint[1], args.ranks)
        else:
            # ASPECT skips the post-refinement signal when nothing was marked.
            assert name == "coarse" and "Skipping mesh refinement, because no cells" in (out.parent/"log").read_text()
            after, outgoing = before, incoming
        assert before == old_count
        np.testing.assert_array_equal(incoming, outgoing)
        material_at_ten.append(outgoing)
        results[name] = dict(max_edge_m=float(spacing), cells_before=before, cells_after=after)
    np.testing.assert_array_equal(*material_at_ten)
    assert results["fine"]["cells_after"] > results["coarse"]["cells_after"]
    results["fine"]["transfer"] = check(args.destination/"fine"/"output", 2, 1, args.ranks)

    # Requested strip spacing is an upper bound: .126 produces exactly the
    # same material edges as .125, whereas .124 introduces a nineteenth strip.
    equal = run(args, "equal-edges", base+fault("Target strip width = 0.126"),
                args.ranks, checkpoint=generation)
    restored(parent, equal, args.ranks, args.ranks)
    run(args, "different-edges", base+fault("Target strip width = 0.124"), args.ranks,
        checkpoint=generation, expected_failure="Physical strip edges differ from the growing fault checkpoint")
    # Perturb one physical edge in a private generation while retaining both
    # the requested spacing and the number of strips. Find the unique observed
    # native-double sequence in ASPECT's compressed Boost binary archive.
    private = args.destination/"shifted-edge-generation"/generation.name
    shutil.copytree(generation, private)
    saved = private/"resume.z"
    data = saved.read_bytes()
    blocks, size, block_size, packed_size = struct.unpack("=4I", data[:16])
    assert blocks == 1 and len(data) == 16+packed_size
    archive = zlib.decompress(data[16:])
    edges = np.r_[old_history[:,0], old_history[-1,1]]
    observed = edges.tobytes()
    assert len(archive) == size and archive.count(observed) == 1
    edges[1] += .001
    compressed = zlib.compress(archive.replace(observed, edges.tobytes()))
    saved.write_bytes(struct.pack("=4I", blocks, size, block_size, len(compressed))+compressed)
    run(args, "shifted-edge", base, args.ranks, checkpoint=private,
        expected_failure="Physical strip edges differ from the growing fault checkpoint")
    # This restart may use its earlier .006 s release (one level on .25 m
    # roots). The .1 s future release cannot supply level two at 10 ms.
    unavailable = run(args, "future-level", resumes+mesh("set Additional refinement times = .006,.1")+
                      marker("Maximum cell edge = .0625"), args.ranks, checkpoint=generation,
                      expected_failure="Maximum cell edge is unreachable")
    assert not list(unavailable.glob("sections-0002.csv"))
    return dict(parent_transfer=transfer, resumes=results, restored_clock_s=.005,
                compared_remesh_clock_s=.01, strip_ids=len(old_history),
                loaded_age_range_s=np.ptp(old_history[:,2]), loaded_slip_range_m=np.ptp(old_history[:,3]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("aspect", "plugin", "destination"):
        parser.add_argument(name, type=lambda arg: Path(arg).resolve())
    parser.add_argument("suite", choices=("admission", "loaded"))
    parser.add_argument("--ranks", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    args.destination.mkdir(parents=True)
    report = check_admission(args) if args.suite == "admission" else check_loaded(args)
    output = json.dumps(report, indent=2)
    (args.destination/"results.json").write_text(output+"\n")
    print(output)
