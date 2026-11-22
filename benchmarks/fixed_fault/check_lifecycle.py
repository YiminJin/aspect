"""Compare a clean step, a rejected attempt and a checkpoint continuation.

Build the tests.fixed_fault regression first to produce its observation plugin.
Each invocation uses a fresh destination. No donor files are required.
"""
import argparse
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import zlib

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("aspect", type=Path)
parser.add_argument("plugin", type=Path)
parser.add_argument("destination", type=Path)
args = parser.parse_args()
binary, plugin, dest = (p.resolve() for p in (args.aspect, args.plugin, args.destination))
dest.mkdir(parents=True, exist_ok=False)
root = Path(__file__).resolve().parents[2]
env = dict(os.environ, OMP_NUM_THREADS="1", DEAL_II_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
checkpoint = """
subsection Checkpointing
  set Steps between checkpoint = 1
end
"""
resume = """
set Resume computation = true
subsection Checkpointing
  set Resume checkpoint = 2
end
"""


def run(name, settings="", parent=False, corrupt=False):
    folder = dest / name
    folder.mkdir()
    output = folder / "output"
    if parent:
        slot = output / "restart/02"
        shutil.copytree(dest / "clean/output/restart/02", slot)
        if corrupt:
            archive = slot / "resume.z"
            data = archive.read_bytes()
            header = list(struct.unpack("=4I", data[:16]))
            body = zlib.decompress(data[16:])
            marker = b"constant section, DGQ2 stress v1"
            assert body.count(marker) == 1
            compressed = zlib.compress(body.replace(marker, marker[:-1] + b"2"))
            header[3] = len(compressed)
            archive.write_bytes(struct.pack("=4I", *header) + compressed)
    prm = folder / "input.prm"
    prm.write_text(f"include {root / 'tests/fixed_fault.prm'}\n"
                   f"set Output directory = {output}\nset Additional shared libraries = {plugin}\n"
                   + checkpoint + settings)
    with (folder / "run.log").open("w") as log:
        result = subprocess.run([binary, prm], cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT)
    log = (folder / "run.log").read_text()
    if corrupt:
        assert result.returncode and "Incompatible fixed-fault representation" in log
    else:
        result.check_returncode()
    return output, log


def compare(path, other):
    """Compare identities, clock, section history and unisolvent Q2 samples."""
    rows = path.read_text().splitlines()
    other_rows = other.read_text().splitlines()
    assert len(rows) == len(other_rows) == 37
    largest = 0.
    for i, (lhs, rhs) in enumerate(zip(rows, other_rows)):
        a, b = lhs.split(), rhs.split()
        first_value = 2 if i else 0
        assert a[:first_value] == b[:first_value] and len(a) == len(b)
        for x, y in zip(map(float, a[first_value:]), map(float, b[first_value:])):
            assert math.isclose(x, y, rel_tol=1.e-11, abs_tol=1.e-12), (path, x, y)
            largest = max(largest, abs(x-y))
    return largest


clean, _ = run("clean")
retry, log = run("retry", """
subsection Time stepping
  set List of model names = function, reject long first fault step
  subsection Function
    set Function expression = if(time==0,0.025,0.0125)
  end
end
""")
assert log.count("Repeating the current time step") == 1
assert (retry / "fault-checks").read_text() == (clean / "fault-checks").read_text() == "0 36\n1 36\n2 36\n"
for step in (0, 1, 2):
    field = f"fault-{step}-0.txt"
    print("Retry", step, "maximum absolute difference", compare(clean / field, retry / field))
restored, _ = run("restored", resume, parent=True)
assert (restored / "fault-checks").read_text() == "2 36\n"
print("Restart at B, maximum absolute difference at C",
      compare(clean / "fault-2-0.txt", restored / "fault-2-0.txt"))
run("incompatible", resume, parent=True, corrupt=True)
print("Incompatible checkpoint rejected")
