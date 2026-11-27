"""Compare balanced meshes, adaptive budgets, accepted histories and checkpointed native frames."""
import argparse
from pathlib import Path
import sys
import math
import numpy as np
from small_controls import ROOT

sys.path.insert(0,str(ROOT/"benchmarks/bp3_qd_thrust"))
from check_pilot import check_pilot, check_native, compare_labels, compare_production, labels, require
from native_frames import compare, frames, read
from history import read_statistics


p=argparse.ArgumentParser(description=__doc__)
p.add_argument("work",type=Path)
p.add_argument("--figure",type=Path)
args=p.parse_args()
work=args.work
# Raw corner intersection versus center-band refinement before p4est balances the mesh.
sine = math.sqrt(.75)
height = 40000*sine
count = 0
for level in range(4):
    n = 2**level
    dx,dy = 2000/n,height/(18*n)
    margin = (sine*dx + .5*dy)/2
    for i in range(40*n):
        for j in range(18*n):
            x,y = -30000+i*dx,j*dy
            low = sine*x+.5*(y-height)
            high = sine*(x+dx)+.5*(y+dy-height)
            r = sine*(x+dx/2)+.5*(y+dy/2-height)
            require((low<=1500 and high>=-1500) == (abs(r)<=1500+margin), "Different corner and center refinement decisions")
            count += 1
require(count==61200, "Incomplete level coverage")
serial=work/"qualified/output"
old_mesh = read(frames(serial)[0.])[0]
coarsened = read(frames(work/"default-coarsen/output")[0.])[0]
correct = set(map(tuple,old_mesh))
extra = np.array(sorted(set(map(tuple,coarsened))-correct))
require(len(old_mesh)-len(coarsened)==24 and len(extra)==8, "Unexpected default coarsening footprint")
require(set(extra[:,0])=={6000.,7000.,12000.,13000.}, "Coarsening in the wrong coarse rectangles")
print("Default coarsening control:",len(coarsened)//4,"cells; changed coarse rectangles",
      "[6000,8000] m at the surface and [12000,14000] m at the base.")
clocks, history = check_pilot(serial)
check_native(serial,clocks,history,True)
require(clocks==[0.,400.,900.], "Fixed startup clocks changed")
for source,new in (("old-serial","qualified"),("old-two","mpi")):
    compare(work/source/"output",work/new/"output",mesh_only=True)
    saved = work/"baseline"/("serial" if new=="qualified" else "two")/"restart/01/mesh"
    require(saved.read_bytes()==(work/new/"output/restart/01/mesh").read_bytes(), "Native p4est mesh changed")
require((serial/"fault-labels-00000.txt").read_bytes()==(work/"log-age/output/fault-labels-00000.txt").read_bytes(), "Serial initial label table changed")
require((work/"old-log-two/output/fault-labels-00000.txt").read_bytes()==(work/"mpi/output/fault-labels-00000.txt").read_bytes(), "Two-rank initial label table changed")
for run in ("mpi","split-fresh","repartition-fresh"):
    out = work/run/"output"
    compare(serial,out)
    compare_production(serial,out,exact=(run=="split-fresh"))
    compare_labels(history,out,range(len(history)) if run=="mpi" else [len(history)-1])
off=work/"off/output"
require(check_pilot(off)==(clocks,history), "Graphical output altered accepted history")
check_native(off,clocks,history,False)
compare_production(serial,off)
compare_production(work/"production/output",work/"continuation/output")
production = list(read_statistics(work/"production/output/statistics"))
require(float(production[-1]["BP3 clock (s)"])==1525., "Short production run did not reach its requested endpoint")
profiles = [(0.1, history[-1])]
for run in ("tight","tighter"):
    out=work/run/"output"
    times, rows=check_pilot(out)
    check_native(out,times,rows,False)
    require(times[-1]==clocks[-1], "Tightened run stopped early")
    rate=max(abs(r[5]-s[5]) for r,s in zip(rows[-1],history[-1]))
    print(run,"accepted steps",len(times)-1,"maximum common-endpoint rate difference from 0.1 budgets",rate,"m/s")
    if run=="tighter":
        print("Difference between tightened solutions",max(abs(r[5]-s[5]) for r,s in zip(rows[-1],prior)),"m/s")
    prior=rows[-1]
    profiles.append((.025 if run=="tight" else .0125, prior))
legacy=work/"legacy/output"
require(list(frames(legacy))==[900.], "Old checkpoint did not establish its first frame")
for r in labels(legacy/"fault-labels-00002.txt"):
    require(all(map(math.isfinite,r)), "Nonfinite resumed material history")
print("Legacy checkpoint without slip reference wrote first frame at 900 s.")
if args.figure:
    import matplotlib.pyplot as plt
    fig,(age,speed) = plt.subplots(2,1,sharex=True,figsize=(7.4,5.5),layout="constrained")
    distance = [row[1]/500 for row in history[0]]
    age.semilogy(distance,[row[3] for row in history[0]],color="#225577")
    age.set_ylabel("Initial age (s)")
    speed.plot(distance,[row[5]*1e9 for row in history[0]],":",color="black",label="Initial")
    for (budget,profile),style in zip(profiles,["-","--","-."]):
        speed.plot(distance,[row[5]*1e9 for row in profile],style,label=f"900 s, budget {budget}")
    for axis in (age,speed):
        axis.axvspan(15,18,color="gray",alpha=.14)
        axis.grid(axis="y",alpha=.2)
    speed.set(xlabel="Distance down dip (km)",ylabel="Slip speed (nm/s)",xlim=(0,40))
    speed.legend(ncol=2)
    fig.savefig(args.figure,dpi=170)
