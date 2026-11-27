"""Exercise a near-budget increase, strong acceleration, age-only deep creep and incapable law."""
from controller import ROOT, attempts, fixture, run, section
import argparse
import math
from pathlib import Path


def main(exe, plugin, growing, dest):
    dest.mkdir()
    one=section("Termination criteria","set Termination criteria = end step","set End step = 1")
    for name,factor in [("exact",1.),("margin",.99)]:
        rows,log=run(exe,dest/name,fixture(plugin,factor=factor,load="if(t>0,-1.5e-7,0)",age_limit=0)+one)
        trials=attempts(log)[1:]
        assert len(trials)==(2 if name=="exact" else 1)
        assert ("retry_reason" in trials[0])==(name=="exact")
        print(name, "attempted/accepted", len(trials),len(rows)-1, "first slip/Dc",
              trials[0]["slip_fraction"],"budget 1/3; dt",rows[-1]["BP3 interval (s)"])

    for name,a,b,age in [("slip",.01,.015,0),("age",.025,.015,.2)]:
        rows,log=run(exe,dest/name,fixture(plugin,a=a,b=b,age_limit=age,maximum=.06)+one)
        trials=attempts(log)[1:]
        assert trials[0]["retry_reason"]==name and "retry_reason" not in trials[-1]
        print(name,"eligible budget",trials[0]["retry_slip_budget"],"change",
              trials[0]["retry_slip_fraction"],trials[0]["retry_abs_log_age"],
              "attempted/accepted",len(trials),len(rows)-1)
    loose=(fixture(plugin)+one+"set Nonlinear solver tolerance = 0.02\n"
           +section("Material model",section("Phase field fault","set Report section convergence = true")))
    _,log=run(exe,dest/"picard",loose)
    veto=next(l for l in log.splitlines() if "Fault Picard rate change: id=" in l)
    import re
    fields=dict(re.findall(r"(\w+)=([^ ]+)",veto))
    x,y=map(float,(fields["assembled"],fields["returned"]))
    assert math.isclose(abs(x-y),float(fields["abs_m_s"]),rel_tol=1e-14)
    assert math.isclose(abs(x-y)/max(x,y),float(fields["relative"]),rel_tol=1e-14) and float(fields["relative"])>1e-8
    print(veto)
    base=ROOT/"benchmarks/growing_fault/growing_seed.prm"
    text=(f"include {base}\nset Output directory = output\nset Additional shared libraries = {growing}\n"
          +section("Material model",section("Phase field fault","set Resistance law = linear dashpot"))
          +section("Time stepping","set List of model names = fault RSF",
                   section("Fault RSF","set Slip resolution policy = weakening dependent")))
    run(exe,dest/"unsupported",text,expect="requires local RSF coefficients")
    print("State-free law with weakening-dependent policy correctly diagnosed.")
    old=.10000002742/.1
    assert .99*old<1<old
    print("Archived 3660 rate factor",old,"projected slip/Dc with 0.99",.99*.10000002742)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("exe","plugin","growing","dest"): p.add_argument(k,type=Path)
    a=p.parse_args()
    main(*(x.resolve() for x in vars(a).values()))
