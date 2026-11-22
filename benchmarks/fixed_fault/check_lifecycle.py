"""Compare a clean step, a rejected attempt and a checkpoint continuation.

Build the tests.fixed_fault regression first to produce its observation plugin.
Each invocation uses a fresh destination. No donor files are required.
"""
import argparse
import os
from pathlib import Path
import shutil
import struct
import subprocess
import zlib

from compare_snapshots import compare

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("aspect", type=Path)
parser.add_argument("plugin", type=Path)
parser.add_argument("destination", type=Path)
parser.add_argument("--case", choices=("fixed_fault", "stationary_fault"), default="fixed_fault")
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


def run(name, settings="", parent=False, corrupt=False, expected_error=None):
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
            marker = b"stationary Q1 labels, DGQ2 stress v2"
            assert body.count(marker) == 1
            compressed = zlib.compress(body.replace(marker, marker[:-1] + b"9"))
            header[3] = len(compressed)
            archive.write_bytes(struct.pack("=4I", *header) + compressed)
    prm = folder / "input.prm"
    prm.write_text(f"include {root / 'tests' / (args.case + '.prm')}\n"
                   f"set Output directory = {output}\nset Additional shared libraries = {plugin}\n"
                   + checkpoint + settings)
    with (folder / "run.log").open("w") as log:
        result = subprocess.run([binary, prm], cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT)
    log = (folder / "run.log").read_text()
    if expected_error:
        assert result.returncode and expected_error in log, (name, log[-3000:])
    else:
        result.check_returncode()
    return output, log


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
run("incompatible", resume, parent=True, corrupt=True, expected_error="Incompatible fixed-fault representation")
print("Incompatible checkpoint rejected")

shifted = "0.25,0; 0.25,1" if args.case == "fixed_fault" else "0.25,0; 0.25,0.5; 0.25,1"
run("different_curve", resume + f"""
subsection Material model
  subsection Fixed fault
    set Fault vertices = {shifted}
  end
end
""", parent=True, expected_error="Incompatible stationary fault label")
print("Checkpoint with different label positions rejected")
