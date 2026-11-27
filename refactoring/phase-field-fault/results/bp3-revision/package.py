"""Build the standalone source package with its canonical event catalogue."""
import argparse
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

here = Path(__file__).resolve().parent
case = here.parents[3]/"benchmarks/bp3_qd_thrust"
names = ("README.md", "CMakeLists.txt", "bp3_qd.cc", "bp3.prm", "pilot.prm", "production.prm",
         "vertices.txt", "make_vertices.py", "check_pilot.py", "native_frames.py", "history.py",
         "prepare.sh", "stampede3.sh")
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("archive",type=Path)
args = p.parse_args()
with ZipFile(args.archive, "x") as archive:
    for src in [case/name for name in names]+[here.parent/"bp3-preparation/history_events.py"]:
        entry = ZipInfo("bp3_qd_thrust/"+src.name)
        entry.external_attr = (0o100755 if src.suffix==".sh" else 0o100644)<<16
        entry.compress_type = ZIP_DEFLATED
        archive.writestr(entry,src.read_bytes())
