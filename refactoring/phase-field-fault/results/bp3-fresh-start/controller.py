"""Small, fully coupled manufactured loading cases for the production fault controller and event observer."""
import argparse
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/"benchmarks/bp3_qd_thrust"))
from history import read_statistics

THREADS = dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", DEAL_II_NUM_THREADS="1", MKL_NUM_THREADS="1")


def section(name, *lines):
    return "\n".join([f"subsection {name}", *lines, "end", ""])


def fixture(plugin, a=.01, b=.015, rate=.0005, dc=.0001, load="if(t>0,-1.5*min(1,max(0,(.2-t)/.08)),0)",
            coefficient=.5, age_limit=.2, maximum=.2, end=.8, event=True, factor=.99):
    age = dc/rate
    traction = 5*a*math.asinh(rate/.002*math.exp((.6+b*math.log(age*.001/dc))/a))+1000*rate
    return (f"include {ROOT}/benchmarks/fixed_fault/fixed_fault.prm\n"
            f"set Additional shared libraries = {plugin}\nset Output directory = output\n"
            f"set End time = {end}\nset Maximum time step = {maximum}\nset Maximum first time step = {maximum}\n"
            "set Maximum relative increase in time step = 25\nset Nonlinear solver tolerance = 1e-8\nset Nonlinear solver failure strategy = cut timestep size\n"
            + section("Material model", section("Phase field fault", "set Fault vertices = 0.5,0; 0.5,1",
                      f"set Direct effect = {a}", f"set Evolution effect = {b}", f"set Initial age = {age}",
                      f"set Initial slip rate = {rate}", f"set Slip distance = {dc}", "set Radiation damping = 1000",
                      f"set Background shear traction = {traction:.17g}", "set Record section tractions = true"))
            + section("Boundary traction model", section("Function", f"set Function expression = 0;{load}"))
            + section("Time stepping", "set List of model names = fault RSF",
                      section("Fault RSF", "set Slip resolution policy = weakening dependent",
                              f"set Maximum slip fraction = {coefficient}", f"set Maximum logarithmic state change = {age_limit}",
                              f"set Proposal safety factor = {factor}", "set Report timestep limits = true"))
            + section("Postprocess", "set List of postprocessors = visualization, BP3 first event profiles, mature fault statistics, BP3 stations, fault slip output",
                      section("BP3 profiles","set Time interval = 1e15","set Slip interval = 1"),
                      section("Visualization","set Time between graphical output = 1e15","set Fault slip interval = 1"))
            + section("Termination criteria","set Checkpoint on termination = true",
                      "set Termination criteria = end time"+(", BP3 first event complete" if event else "")))


def run(exe, dest, text, ranks=1, resume=False, expect=None):
    dest.mkdir(exist_ok=resume)
    path=dest/("resume.prm" if resume else "start.prm")
    with path.open("x") as out:
        out.write(text + ("set Resume computation = true\n" if resume else ""))
    with path.with_suffix(".log").open("x") as log:
        proc=subprocess.run((["mpiexec","-n",str(ranks)] if ranks>1 else []) + [str(exe),str(path)],cwd=dest,
                            stdout=log,stderr=subprocess.STDOUT,env={**os.environ,**THREADS})
    log=path.with_suffix(".log").read_text()
    if expect:
        assert proc.returncode and expect in log, log[-3000:]
        return
    assert proc.returncode==0, log[-3000:]
    return list(read_statistics(dest/"output/statistics")),log


def attempts(log):
    return [dict(re.findall(r"(\w+)=([^ ]+)",line)) for line in log.splitlines() if line.startswith("Fault timestep at ")]


def values(row, keys):
    return tuple(float(row[key]) for key in keys)


def check_event(path, rows):
    files = sorted((path/"output").glob("fault-profile-*.txt"))
    times = [float(p.open().readline().split()[-1]) for p in files]
    bulk = [float(r.get("timestep")) for r in ET.parse(path/"output/solution.pvd").findall(".//DataSet")]
    onset,completion=values(rows[-1],["BP3 event onset (s)","BP3 event completion (s)"])
    assert onset>0 and completion>onset and len(times)==len(bulk)==3
    assert all(math.isclose(a,b,rel_tol=5e-6) for a,b in zip(times,bulk)) and times==[0,onset,completion]
    clocks=[float(r["BP3 clock (s)"]) for r in rows]
    peaks=[float(r["BP3 peak speed (m/s)"]) for r in rows]
    i,j=clocks.index(onset),clocks.index(completion)
    assert i>0 and peaks[i-1]<.001<=peaks[i] and all(v<.001 for v in peaks[j-4:j+1]) and peaks[j-5]>=.001
    assert values(rows[-1],["BP3 accepted time before onset (s)","BP3 accepted time before completion (s)"])==(clocks[i-1],clocks[j-1])
    return times


def check_native_rollback(dest):
    from native_frames import frames, read
    import numpy as np
    original=frames(dest/"pulse/output")
    for case in ("clean","restart"):
        candidate=frames(dest/case/"output")
        expected=list(original)[:2] if case=="clean" else list(original)
        assert list(candidate)==expected, (case,candidate.keys(),expected)
        for clock in expected:
            cells,fields,_=read(original[clock])
            other,actual,_=read(candidate[clock])
            np.testing.assert_array_equal(cells,other)
            assert fields.keys()==actual.keys()
            for key in fields:
                assert np.isfinite(fields[key]).all() and np.array_equal(fields[key],actual[key]), (case,clock,key)
    print("Clean/retried and uninterrupted/resumed native fields match at every shared frame.")


def main(exe,plugin,dest,ranks):
    dest.mkdir()
    # Strengthening material at steady state slips five Dc in one global-ceiling step.
    for name,a,b in [("steady",.025,.015),("b-zero",.01,0.),("neutral",.015,.015),("weak",.01,.015)]:
        step=4e7
        text=fixture(plugin,a=a,b=b,rate=1e-9,dc=.008,load="0",maximum=step,end=2*step,event=False)
        one=section("Termination criteria","set Termination criteria = end step","set End step = 1")
        rows,log=run(exe,dest/name,text+one,ranks)
        assert len(rows)==2
        dt=float(rows[1]["BP3 interval (s)"])
        expect=min(step,.99*.5*(a/b)*.008/float(rows[0]["BP3 peak speed (m/s)"])) if 0<b and a<=b else step
        assert math.isclose(dt,expect,rel_tol=1e-12),(name,dt,expect)
        if a>b or not b:
            assert dt==step and float(rows[1]["BP3 0 m slip (m)"])>.008*4
        print(name,"duration",dt,"slip",rows[1]["BP3 0 m slip (m)"],flush=True)

    text=fixture(plugin)
    rows,log=run(exe,dest/"pulse",text,ranks)
    print("pulse",len(attempts(log))-1,"physical attempts",len(rows)-1,"accepted; event",check_event(dest/"pulse",rows),flush=True)
    rejected=[r for r in attempts(log) if "retry_dt" in r]
    assert rejected
    h=float(rows[1]["BP3 interval (s)"])
    one=section("Termination criteria","set Termination criteria = end step","set End step = 1")
    clean,_=run(exe,dest/"clean",text+one+f"set Maximum first time step = {h:.17g}\n",ranks)
    for a,b in zip(rows[:2],clean):
        assert all(a[k]==b[k] for k in a if k.startswith("BP3"))
    for p in (dest/"clean/output").glob("fault-labels*"):
        assert p.read_bytes()==(dest/"pulse/output"/p.name).read_bytes()
    split=section("Termination criteria","set Termination criteria = end step","set End step = 3")
    run(exe,dest/"restart",text+split,ranks)
    resumed,_=run(exe,dest/"restart",text,ranks,True)
    assert check_event(dest/"restart",resumed)==check_event(dest/"pulse",rows)
    for a,b in zip(rows,resumed,strict=True):
        assert all(a[k]==b[k] for k in a if k.startswith("BP3")), (a,b)

    check_native_rollback(dest)

    # End at a common physical time during the pulse for an independent timestep refinement.
    for name,cap in [("production",.2),("tight",.002)]:
        common,_=run(exe,dest/name,fixture(plugin,maximum=cap,end=.10,event=False)+
                     section("Postprocess",section("BP3 profiles","set Time interval = 0")),ranks)
        last=common[-1]
        print(name,"common clock, peak, location, slip, onset bracket",
              values(last,["BP3 clock (s)","BP3 peak speed (m/s)","BP3 peak distance (m)","BP3 0 m slip (m)",
                           "BP3 accepted time before onset (s)","BP3 event onset (s)"]),
              "mechanical shear Pa",sorted((dest/name/"output").glob("fault-profile*"))[-1].read_text().splitlines()[3].split()[7])
        assert math.isclose(float(last["BP3 clock (s)"]),.1,abs_tol=1e-15)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ["executable","plugin","directory"]: p.add_argument(name,type=Path)
    p.add_argument("--ranks",type=int,default=1)
    arg=p.parse_args()
    main(arg.executable.resolve(),arg.plugin.resolve(),arg.directory.resolve(),arg.ranks)
