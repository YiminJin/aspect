"""Read fresh layered RSF and state-free checkpoints from a pre-boundary-restart branch installation."""
import argparse
from pathlib import Path
import zlib

import numpy as np

from check_restart import SAVE, check_clocks, compare, fault, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old_prefix", "prefix", "layered_plugin", "stateless_plugin", "destination"):
        parser.add_argument(name, type=lambda p: Path(p).resolve())
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=False)
    common = dict(root=args.destination)
    for law in ("rate and state", "linear dashpot"):
        layered = law == "rate and state"
        name = "layered-rsf" if layered else "stateless-interior"
        deck = "layered_moving_fault/layered.prm" if layered else "fixed_fault/fixed_fault.prm"
        relative_plugin = ("layered_moving_fault/liblayered_moving_fault.release.so" if layered else
                           "growing_fault/libgrowing_fault_checks.release.so")
        new_plugin = args.layered_plugin if layered else args.stateless_plugin
        settings = "" if layered else fault("""set Fault geometry = fixed interior
set Fit trace to phase = false
set Fault vertices = .5,.05; .5,.95
set Resistance law = linear dashpot
""") + """subsection Postprocess
set List of postprocessors = mature fault statistics, stateless fault archive check
end
"""
        writer = run(name+"-old", args.old_prefix/"benchmarks"/deck,
                     [args.old_prefix/"benchmarks"/relative_plugin], prefix=args.old_prefix,
                     settings=settings + SAVE + "set End time = .0125\n", **common)
        # The writer must exercise the legacy reader. Verify its actual wire
        # record as well as the lack of newer physical seed/sense metadata.
        blob = zlib.decompress((writer/"restart/02/resume.z").read_bytes()[16:])
        key = b"Material fault labels v1" if layered else b"Material fault labels linear dashpot"
        if key not in blob or b"Mature fault trace v1" in blob:
            raise ValueError(f"Writer did not produce the legacy {name} archive")
        clean = run(name+"-clean", args.prefix/"benchmarks"/deck, [new_plugin],
                    prefix=args.prefix, settings=settings, **common)
        resumed = run(name+"-resume", args.prefix/"benchmarks"/deck, [new_plugin], prefix=args.prefix,
                      settings=settings, checkpoint=writer, **common)
        if layered:
            check_clocks(resumed, [2])
            compare(clean, resumed, 2)
        else:
            file = "fault-labels-00002.txt"
            reference = (clean/file).read_text()
            if (resumed/file).read_text() != reference:
                raise ValueError("Restored stateless records differ from the uninterrupted endpoint")
            if "age" in reference.splitlines()[0] or (resumed/"fault-labels-00001.txt").exists():
                raise ValueError("Unexpected state-free history or publication")
            values = np.loadtxt(resumed/file)
            if values.shape != (2, 5) or not np.all(values[:, 3:] > 0.):
                raise ValueError("Expected two sliding labels with zero constitutive-history columns")
            print(name, "restored both accepted slips and speeds exactly; no age column", flush=True)


if __name__ == "__main__":
    main()
