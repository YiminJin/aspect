"""Run one manufactured DG endpoint sequence through the installed AMR criterion."""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run(exe, plugin, destination, ranks, generation=None, stop=None):
    here=Path(__file__).resolve().parent
    directory=destination.resolve()
    directory.mkdir(parents=True,exist_ok=False)
    benchmark=exe.resolve().parents[1]/"benchmarks"
    shutil.copy(benchmark/"growing_fault/growing_seed.prm",directory)
    shutil.copy(here/"identity.prm",directory)
    (directory/"plugin").mkdir()
    shutil.copy(plugin,directory/"plugin")
    text="include identity.prm\n"
    if stop is not None:
        text+=f"subsection Termination criteria\n set Termination criteria = end step\n set End step = {stop}\nend\n"
    if generation:
        generation=generation.resolve(strict=True)
        shutil.copytree(generation,directory/"output/restart"/generation.name)
        text+=f"set Resume computation = true\nsubsection Checkpointing\n set Resume checkpoint = {int(generation.name)}\nend\n"
    prm=directory/"run.prm"
    prm.write_text(text)
    subprocess.run([sys.executable,benchmark/"granite_fault_growth/run.py",exe.resolve(),
                    prm,"timing","--ranks",str(ranks)],check=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("executable","plugin","destination"):
        p.add_argument(name,type=Path)
    p.add_argument("--ranks",type=int,choices=(1,2),required=True)
    p.add_argument("--restart",type=Path)
    p.add_argument("--end-step",type=int)
    a=p.parse_args()
    run(a.executable,a.plugin,a.destination,a.ranks,a.restart,a.end_step)
