"""Build a self-contained BP3 input and plugin-source archive from tracked files."""
import argparse
from pathlib import Path
import zipfile

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("new_zip",type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    catalogue = here.parents[1]/"refactoring/phase-field-fault/results/bp3-preparation/history_events.py"
    files = [here/name for name in (
        "README.md prepare.sh stampede3.sh CMakeLists.txt bp3_qd.cc historical_profile.cc "
        "startup_cuts.cc event_profiles.cc first_event.h bp3-common.prm bp3.prm production.prm pilot.prm "
        "contact.prm intermediate.prm make_vertices.py make_historical_vertices.py vertices.txt "
        "historical-vertices.txt check_pilot.py history.py native_frames.py").split()]
    files += [here/name for name in ("check_historical.py","check_event.cc") if (here/name).exists()]
    with zipfile.ZipFile(args.new_zip,"x",zipfile.ZIP_DEFLATED) as archive:
        for path in files+[catalogue]:
            archive.write(path, "bp3-first-event/"+path.name)
