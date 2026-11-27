"""Replay the standard-box MPI, restart, cadence and accuracy controls from a staged pilot."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
from xml.etree import ElementTree as ET

from small_controls import section, THREADS


def run(stage, work, name, mode="pilot", overrides="", ranks=1, parent=None, slot="02"):
    path = work/name
    path.mkdir()
    for source in ("bp3.prm","pilot.prm","production.prm","vertices.txt","libbp3_qd.release.so"):
        shutil.copy2(stage/source,path/source)
    if parent is not None:
        out = path/"output"
        restart = out/"restart"
        restart.mkdir(parents=True)
        shutil.copytree(parent/"restart"/slot,restart/slot)
        (restart/"last_good_checkpoint.txt").write_text(str(int(slot)))
        # The writer preserves the series in the checkpoint. Keep the previously
        # emitted pieces for both the visualization reader and its new PVD.
        if (parent/"solution.pvd").exists():
            # Snapshot metadata names the next solve. Copy only frames emitted
            # through the accepted parent, so stale future VTUs cannot pass.
            meta = dict(line.split(" ",1) for line in (restart/slot/"metadata.txt").read_text().splitlines())
            step = int(meta["timestep_number"])-1
            stats = [line.split() for line in (parent/"statistics").read_text().splitlines() if not line.startswith("#")]
            parent_time = float(stats[step][1])
            (out/"solution").mkdir()
            for frame in ET.parse(parent/"solution.pvd").findall(".//DataSet"):
                if float(frame.attrib["timestep"]) <= parent_time:
                    pvtu = parent/frame.attrib["file"]
                    for file in pvtu.parent.glob(pvtu.stem+".*"):
                        shutil.copy2(file,out/"solution")
        overrides += "\nset Resume computation = true\n"
    prm = path/"check.prm"
    prm.write_text(f"include {mode}.prm\n"+overrides)
    exe = (stage/"executable.path").read_text().strip()
    command = (["mpiexec","-n",str(ranks)] if ranks>1 else [])+[exe,prm.name]
    with (path/"run.log").open("w") as log:
        subprocess.run(command,cwd=path,check=True,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,**THREADS})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",type=Path,help="Staged, completed serial pilot")
    p.add_argument("work",type=Path)
    p.add_argument("cases",nargs="+")
    a=p.parse_args()
    stage=a.stage.resolve()
    work=a.work.resolve()
    work.mkdir(exist_ok=True)
    for case in a.cases:
        if case=="mpi":
            run(stage,work,case,ranks=2)
        elif case in ("split-fresh","repartition-fresh"):
            run(stage,work,case,parent=stage/"output",ranks=2 if case.startswith("repartition") else 1)
        elif case=="default-coarsen":
            run(stage,work,case,overrides="set End time = 0\n"+section("Mesh refinement",
                section("Maximum refinement function","set Coarsen cells above maximum = true")))
        elif case=="legacy":
            run(stage,work,case,parent=work/"baseline/serial",overrides=section("Postprocess",section("Visualization","set Fault slip interval = .1")))
        elif case=="off":
            run(stage,work,case,overrides=section("Postprocess","set List of postprocessors = BP3 stations, mature fault statistics, velocity statistics, point values"))
        elif case in ("tight","tighter"):
            cap = .025 if case=="tight" else .0125
            run(stage,work,case,overrides=section("Time stepping",section("Fault RSF",
                f"set Maximum slip fraction = {cap}\nset Maximum logarithmic state change = {cap}")))
        elif case in ("production","continuation"):
            run(stage,work,case,mode="production",overrides="set End time = 1525\n",
                parent=stage/"output" if case=="continuation" else None,slot="03")
        else:
            raise ValueError(case)
