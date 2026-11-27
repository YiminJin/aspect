"""Plot accepted peaks and the two rejected duration-budget trials in the small coupled calculation."""
import argparse
from pathlib import Path
from controller import attempts
from history import read_statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("controls",type=Path)
p.add_argument("png",type=Path)
args=p.parse_args()
def frames(case):
    return list(read_statistics(args.controls/case/"output/statistics"))

pulse=frames("pulse")
tight=frames("tight")
fig,(ax,bx)=plt.subplots(1,2,figsize=(9,3.7),layout="constrained")
for records,label,style in [(pulse,"Production controls","o-"),(tight,"2 ms ceiling",".--")]:
    ax.plot([float(r["BP3 clock (s)"]) for r in records],[1e3*float(r["BP3 peak speed (m/s)"]) for r in records],
            style,ms=3,label=label)
ax.axhline(1,color=".4",ls=":",label="Event threshold")
ax.set(xlabel="Accepted time (s)",ylabel="Maximum slip speed (mm/s)")
ax.legend(frameon=False,fontsize=8)
for row,line in zip(attempts((args.controls/"pulse/start.log").read_text()),
                    [s for s in (args.controls/"pulse/start.log").read_text().splitlines() if s.startswith("Fault timestep at ")],strict=True):
    t=float(line.split()[3])
    if not float(row["dt"]): continue
    failed="retry_reason" in row
    bx.plot(t,float(row["dt"])*1e3,"x" if failed else "o",color="#b63c1b" if failed else "#3573a0",ms=7 if failed else 4)
bx.plot([],[],"o",color="#3573a0",label="Accepted")
bx.plot([],[],"x",color="#b63c1b",label="Rejected")
bx.set(xlabel="Attempted endpoint time (s)",ylabel="Attempted duration (ms)")
bx.legend(frameon=False,fontsize=8)
fig.savefig(args.png,dpi=170)
