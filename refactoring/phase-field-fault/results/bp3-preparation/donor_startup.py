"""Check the libraries and fault coordinates reported by the ordinary BP3 startup."""
import argparse
from pathlib import Path


def verify(log, fault):
    expected_libraries = ["./plugin/libbp3_restore_150x50.release.so",
                          "./plugin/libbp3_coupled_checks.release.so"]
    expected_points = [float(v) for line in fault.read_text().splitlines()
                       for v in line.partition("#")[0].split()]
    libraries = []
    anchors = None
    with log.open() as stream:
        for line in stream:
            if line.startswith("Loading shared library <"):
                libraries.append(line.split("<", 1)[1].split(">", 1)[0])
            if line.startswith("BP3 straight geometry v1 box "):
                coords, sense = line.split(" anchors ")[1].split(" thrust ")
                anchors = list(map(float, coords.split()))
                if int(sense.split()[0]) != -1:
                    raise ValueError("Expected the right-dipping thrust")
                break
    if libraries != expected_libraries:
        raise ValueError(f"Loaded libraries differ from the donor pair: {libraries}")
    if len(expected_points) != 15 or anchors != expected_points:
        raise ValueError(f"Startup geometry differs from {fault}: {anchors}")
    print(f"Startup loaded both BP3 plugins and all five thrust anchors from {fault.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("fault", type=Path)
    args = parser.parse_args()
    verify(args.log, args.fault)
