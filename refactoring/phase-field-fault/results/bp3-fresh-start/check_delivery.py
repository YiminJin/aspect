"""Compare effective inputs from both installations and check the shipped signed traction conversion."""
import argparse
import csv
import io
import json
from pathlib import Path
import subprocess
import sys


def values(path):
    text=path.read_text()
    data,_=json.JSONDecoder().raw_decode(text[text.index("\n{")+1:])
    def walk(group,base):
        for key,entry in group.items():
            name=base+key
            if "alias" in entry:
                continue
            if "value" in entry:
                yield name,entry["value"]
            else:
                yield from walk(entry,name+"/")
    return dict(walk(data,""))


p=argparse.ArgumentParser(description=__doc__)
for name in ("old_json","new_json","package","output"): p.add_argument(name,type=Path)
args=p.parse_args()
old,new=values(args.old_json),values(args.new_json)
diff={k:(old.get(k),new.get(k)) for k in sorted(old.keys()|new.keys()) if old.get(k)!=new.get(k)}
expected={"Time stepping/Fault RSF/"+k for k in ("Maximum slip fraction","Maximum logarithmic state change",
                                                "Proposal safety factor","Slip resolution policy")}
expected.update(["Material model/Phase field fault/Report section convergence","Postprocess/BP3 profiles/Slip interval"])
assert diff.keys()==expected,diff
for key,(a,b) in diff.items(): print(key,repr(a),"->",repr(b))
sys.path.insert(0,str(args.package/"tools"))
from tractions import frictional_shear
from run import input_text

eta=float(new["Material model/Phase field fault/Radiation damping"])
for v in (-1.,1.):
    assert 10e6-frictional_shear(10e6,v,eta)==4624440*v
    print("Mechanical minus frictional shear at",v,"m/s:",eta*v,"Pa")
prm=(args.package/"production.prm").read_text()
try: input_text(prm.replace("historical-output","old-user-output"),False,1,1.)
except ValueError as err: assert "Output directory" in str(err)
else: raise AssertionError("Accepted an unmanaged output path")
path=args.output/"fault-profile-0000001.txt"
table=list(csv.DictReader(io.StringIO(subprocess.check_output([sys.executable,args.package/"tools/tractions.py",path],text=True))))
with path.open() as f:
    next(f);next(f)
    for converted,row in zip(table,csv.DictReader(f,delimiter=" ",skipinitialspace=True),strict=True):
        for key in ("id","s_m","raw_compression_Pa","friction_compression_Pa"):
            assert converted[key]==row[key]
        assert float(converted["frictional_shear_Pa"])==float(row["shear_Pa"])-eta*float(row["rate_m_s"])
print("Converted and checked",len(table),"material IDs at the first accepted endpoint.")

# Check the actual launcher's guards using a real checkpoint pointer and
# private scratch markers. No solver or scheduler invocation is needed.
import contextlib
import os
import tempfile
import time
import run as launcher

package,output=args.package.resolve(),args.output.resolve()
with tempfile.TemporaryDirectory() as tmp:
    previous=Path.cwd()
    try:
        os.chdir(tmp)
        for name in ("plugins","production.prm","historical-vertices.txt"):
            Path(name).symlink_to(package/name)
        launcher.OUTPUT.mkdir()
        (launcher.OUTPUT/"restart").symlink_to(output/"restart",target_is_directory=True)
        Path("jobs").mkdir()
        os.environ.update(SLURM_JOB_ID="marker-test",SLURM_JOB_END_TIME=str(int(time.time())+300))
        provenance="".join(f"{launcher.file_digest(f)}  {f}\n" for f in
                           (launcher.SOLVER,launcher.PLUGIN,launcher.VERTICES,launcher.PARAMETERS))
        identity={"id":"this-run","inputs":provenance}
        (launcher.OUTPUT/"bp3-run.json").write_text(json.dumps(identity))
        Path("jobs/bp3-run.json").write_text(json.dumps(identity))
        (launcher.OUTPUT/"startup-step-1.ok").write_text("another-run checked step 1\n")
        for mode,expected in [("start","unused"),("resume","Foreign startup marker"),
                              ("resume","different BP3 run")]:
            if expected=="different BP3 run":
                Path("jobs/bp3-run.json").write_text(json.dumps(dict(identity,id="different")))
            try:
                with contextlib.redirect_stdout(io.StringIO()): launcher.run(mode)
            except ValueError as err: assert expected in str(err), str(err)
            else: raise AssertionError(f"Launcher accepted {expected}")
            print("Launcher rejected",expected)
    finally:
        os.chdir(previous)
