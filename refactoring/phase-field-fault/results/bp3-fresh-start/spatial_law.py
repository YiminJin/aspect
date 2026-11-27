"""Distinguish the spatial BP3 timestep law from using the shallow coefficients for every ID."""
import argparse
import math
from pathlib import Path
from controller import ROOT, attempts, run, section

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("executable",type=Path)
p.add_argument("plugin",type=Path)
p.add_argument("new_directory",type=Path)
a=p.parse_args()
dest=a.new_directory.resolve()
dest.mkdir()
source=ROOT/"benchmarks/bp3_qd_thrust"
lines=(source/"vertices.txt").read_text().splitlines()
table=dest/"ages.txt"
with table.open("w") as stream:
    print(*lines[:2],sep="\n",file=stream)
    for row in lines[2:]:
        i,x,y,age=row.split()
        s=2*float(x)
        # A broad, faster deep segment must have no influence on the weakening-zone minimum.
        # A smaller bump puts the eligible maximum at the crossover.
        shift=1.5*max(0.,1-abs(s-16000.)/1000.)+4*min(1.,max(0.,(s-16000.)/1800.))
        print(i,x,y,float(age)*math.exp(-shift),file=stream)
text=((source/"bp3-common.prm").read_text()+(source/"bp3.prm").read_text().split("\n",1)[1]+f"\nset Output directory = output\nset Additional shared libraries = {a.plugin.resolve()}\n"
      +section("Material model",section("Phase field fault",f"set Fault vertex file = {table}","set Report section convergence = true"))
      +section("Time stepping",section("Fault RSF","set Slip resolution policy = weakening dependent","set Maximum slip fraction = .5",
             "set Proposal safety factor = .99","set Maximum logarithmic state change = 0","set Report timestep limits = true"))
      +section("Postprocess","set List of postprocessors = mature fault statistics")
      +section("Termination criteria","set Termination criteria = end step","set End step = 0"))
rows,log=run(a.executable.resolve(),dest/"experiment",text)
results=[]
for row in (dest/"experiment/output/fault-labels-00000.txt").read_text().splitlines()[1:]:
    i,x,y,_,_,rate=map(float,row.split())
    s=math.hypot(x,y-34641.016151377546)
    if abs(s-16000)<1e-8: s=16000.0
    direct=.01+.015*min(1,max(0,(s-15000)/3000))
    bound=.5*(direct/.015)*.008/rate if s<=16000 else math.inf
    results.append((bound,int(i),s,rate))
best=min(results)
fast=max(results,key=lambda r:r[3])
v=attempts(log)[0]
print("eligible minimum",best,"fastest material ID",fast,"reported",v)
assert fast[2]>16000 and int(v["id"])==best[1] and abs(best[2]-16000)<1e-6
assert math.isclose(float(v["raw_limit"]),best[0],rel_tol=1e-13)
assert math.isclose(float(v["proposal"]),.99*best[0],rel_tol=1e-13)
