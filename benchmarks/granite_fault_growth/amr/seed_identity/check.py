"""Verify which excursion the installed timestep criterion actually refined."""
import argparse
from pathlib import Path
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from trajectory import Case, compare, field_names


def verify(folder,historical,resumed=None):
    log=(folder/"timing/solver.log").read_text()
    matches=re.findall(r"Manufactured seed step (\d+), source edge ([\d.]+) m, receiver edge ([\d.]+) m",log)
    visits=[(int(k),float(a),float(b)) for k,a,b in matches]
    case=Case(folder/"output")
    rows=[]
    for step in range(3):
        pts,fields,_,book=case.load(step)
        if book: assert book["time"].item()==step*.0025
        cells=case.phase_patches(pts,fields)
        sides=[]
        for name,x,y in (("original",2.8,.24),("gap",2.6,.24),("island",2.4,.24),("receiver",2.8,1.85)):
            cell,=[c for c in cells if all(c[0]<[x,y]) and all(c[0]+c[1]>[x,y])]
            value=.02 if name=="original" or (name=="island" and step>=1) or (name=="receiver" and step>=2) else .004
            if (abs(cell[2]-value)>2e-14).any():
                raise ValueError(f"Incorrect manufactured {name} at step {step}")
            sides.append(round(cell[1][0],12))
        rows.append(sides)
    # The receiver is 1.5 m above the seed line, beyond the 2 ell=1.2 m
    # geometric band, and disk replenishment is disabled in this fixture.
    assert rows[0][3]==.25 and rows[1][3]==.25
    if historical:
        assert rows[2][3]==.25 and "Growing fault missed extension" not in log
    else:
        assert rows[2][3]==.125 and log.count("Growing fault missed extension")==1
        assert log.count("Repeating the current time step")==1
        assert [b for step,_,b in visits if step==2]==[.25,.125]
        assert rows[0][0]==.125 and rows[2][0]==.0625
    print("source, gap, island, receiver edge at steps 0/1/2 (m):",rows)
    print("trial visits:",visits)
    if resumed:
        text=(resumed/"timing/solver.log").read_text()
        assert text.count("Growing fault missed extension")==1 and text.count("Repeating the current time step")==1
        compare(case.load(2),Case(resumed/"output").load(2),field_names(case.folder),strict=True)
        print("Checkpoint after the first physical step reproduced all second endpoint tables at 1e-12.")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("--historical",action="store_true")
    p.add_argument("--resumed",type=Path)
    a=p.parse_args()
    verify(a.run,a.historical,a.resumed)
