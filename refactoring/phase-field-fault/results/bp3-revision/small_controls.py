"""Ordinary parameter-file controls for converged-trial limits and per-ID frame scheduling."""
import argparse
import math
import os
import sys
from pathlib import Path
import subprocess
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/"benchmarks/bp3_qd_thrust"))
from check_pilot import require

THREADS = {"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1","DEAL_II_NUM_THREADS":"1"}


def section(name, body):
    return f"subsection {name}\n{body}\nend\n"


def stepper(slip,state,models="function, fault RSF"):
    return section("Time stepping", f"set List of model names = {models}\n" +
                   section("Fault RSF", f"set Maximum slip fraction = {slip}\nset Maximum logarithmic state change = {state}"))


def run(exe,root,name,base,settings):
    path = root/name
    path.mkdir()
    prm = path/"case.prm"
    prm.write_text(f"include {base}\nset Output directory = output\n"+settings)
    with (path/"run.log").open("w") as log:
        subprocess.run([exe,prm],cwd=path,stdout=log,stderr=subprocess.STDOUT,
                       env={**os.environ,**THREADS},check=True)
    return path/"output"


def labels(path):
    return [[[float(x) for x in line.split()] for line in file.read_text().splitlines()[1:]]
            for file in sorted(path.glob("fault-labels-*.txt"))]


def times(path):
    return [float(e.attrib["timestep"]) for e in ET.parse(path/"solution.pvd").findall(".//DataSet")]


def mature(exe,dest):
    base = ROOT/"benchmarks/fixed_fault/fixed_fault.prm"
    def output(threshold,cadence=1000):
        return section("Postprocess","set List of postprocessors = mature fault statistics, visualization, fault slip output\n"+
                       section("Visualization",f"set Time between graphical output = {cadence}\nset Fault slip interval = {threshold}"))
    one = section("Termination criteria","set Termination criteria = end step\nset End step = 1")
    # Reducing imposed external shear raises the positive mature rate in this fixture.
    loading = section("Boundary traction model",section("Function","set Function expression = 0; if(t>0,-0.4,0.02)"))
    settings = one+loading+output(1e-9)
    baseline = run(exe,dest,"acceleration",base,settings)
    parent,candidate = labels(baseline)
    dt = .0125
    old,new = parent[0],candidate[0]
    dc = .008
    require(new[5] > old[5] > 0, "The driving pulse did not accelerate the fault")
    for kind in ("slip","age"):
        if kind == "slip":
            predicted = old[5]*dt/dc
            attempted = new[5]*dt/dc
            cap = (attempted+predicted)/2
            limiter = stepper(cap,0)
        else:
            age = old[3]*math.exp(-old[5]*dt/dc) - dc/old[5]*math.expm1(-old[5]*dt/dc)
            predicted = abs(math.log(age/old[3]))
            attempted = abs(math.log(new[3]/old[3]))
            cap = (attempted+predicted)/2
            limiter = stepper(0,cap)
        retry = run(exe,dest,"retry-"+kind,base,settings+limiter)
        after = labels(retry)
        require(len(after) == 2 and after[0] == parent, "Retry changed the parent or endpoint count")
        require("Fault slip/state budget exceeded" in (retry.parent/"run.log").read_text(), "The configured accuracy gate did not reject the trial")
        accepted = after[1][0]
        # The published increment and rate give the full-precision accepted duration.
        h = accepted[4]/accepted[5]
        require(0 < h < dt and len(times(retry)) == 2 and math.isclose(times(retry)[1],h,rel_tol=5e-6), "Rejected trial leaked into frames or did not shorten duration")
        require((h*accepted[5]/dc if kind=="slip" else abs(math.log(accepted[3]/old[3]))) <= cap, "Retry still exceeds the selected budget")
        clean = run(exe,dest,"clean-"+kind,base,settings+limiter+f"set Maximum first time step = {h:.17g}\n")
        for frame,expected in zip(labels(clean),after,strict=True):
            for row,ref in zip(frame,expected,strict=True):
                require(all(math.isclose(x,y,rel_tol=2e-13,abs_tol=1e-15) for x,y in zip(row,ref)), "Different clean and retried label history")
        from native_frames import compare
        compare(clean,retry)
        print(kind,"parent prediction",predicted,"attempted",attempted,"budget",cap,"accepted dt",h)

    three = section("Material model",section("Phase field fault","set Fault vertices = 0.5,0; 0.5,0.5; 0.5,1"))
    switch = section("Boundary traction model",section("Function","set Function expression = 0;0.02+if(t<0.02,0.45,-0.45)*(y-0.5)"))
    reference = run(exe,dest,"switch",base,three+switch+output(1.e-12))
    _,a,b = labels(reference)
    diffs = [y[4]-x[4] for x,y in zip(a,b)]
    first = max(a,key=lambda x:x[4])
    next_id = diffs.index(max(diffs))
    scalar = max(x[4] for x in b)-first[4]
    cap = (scalar+min(first[4],max(diffs)))/2
    require(next_id != first[0] and scalar < cap <= min(first[4],max(diffs)), "The fixture did not distinguish per-ID and global maximum slip")
    moving = run(exe,dest,"moving-maximum",base,three+switch+output(cap))
    require(times(moving) == [0.,.0125,.025] and labels(moving)==labels(reference), "Moving front missed a frame or changed the physics")
    require(max(diffs)>cap, "Second material maximum did not cross the threshold")
    print("Fastest ID switched",int(first[0]),"to",next_id,"per-ID increment",max(diffs),
          "change of scalar max",scalar,"threshold",cap)
    both = run(exe,dest,"coincident",base,three+switch+output(cap,.0125))
    require(times(both)==times(moving) and labels(both)==labels(reference), "Coincident triggers duplicated output or changed history")
    # Only the ordinary step trigger fires at step 2. Its acknowledgment
    # prevents a spurious slip frame at step 3.
    threshold = max(row[4] for row in b)*1.05
    reset = run(exe,dest,"step-reset",base,three+switch+output(threshold)+
        "set End time = 0.0375\n"+section("Postprocess",section("Visualization","set Time steps between graphical output = 2")))
    profile = labels(reset)
    require(max(r[4] for r in profile[3]) > threshold, "Step-reset fixture did not cross the reference from time zero")
    require(max(r[4]-q[4] for r,q in zip(profile[3],profile[2])) < threshold, "Step-reset fixture crossed the updated reference")
    require(times(reset)==[0.,.025], "An ordinary graphical frame failed to reset the slip reference")
    print("Ordinary step frame reset the reference at 25 ms; no additional frame at 37.5 ms.")


def growing(exe,dest,plugin):
    base = ROOT/"benchmarks/growing_fault/growing_seed.prm"
    for kind in ("rsf","linear"):
        settings = ""
        if kind == "linear":
            settings = (f"set Additional shared libraries = {plugin}\n"+
                        section("Material model",section("Phase field fault","set Resistance law = linear dashpot")))
        ordinary = run(exe,dest,"growth-"+kind,base,settings)
        cap = 1.e-20 if kind == "linear" else .1
        limited = run(exe,dest,"growth-budget-"+kind,base,settings+stepper(cap,cap,"fault RSF"))
        tables = sorted(ordinary.glob("*.csv"))
        require(tables and all(f.read_bytes()==(limited/f.name).read_bytes() for f in tables), "Enabling the timestep model changed accepted growing histories")
        print(kind,len(tables),"growth history files identical through 5 ms")
    import csv
    def sections(path,step):
        with (path/f"sections-{step:04}.csv").open() as f:
            return list(csv.DictReader(f))
    reverse = section("Boundary velocity model", section("Function",
        "set Function constants = xc=1,yc=0.5,tx=0.86602540378443871,ty=0.49999999999999994,extension=-0.003,shear=-0.020"))
    one = section("Termination criteria","set Termination criteria = end step\nset End step = 1")
    trial = run(exe,dest,"negative-rates",base,reverse+one)
    data = sections(trial,1)
    rates = [float(x["V"]) for x in data]
    require(min(rates) < 0, "Reversed wall motion produced no negative rate")
    cap = max(map(abs,rates))*.005/.01/2
    limited = run(exe,dest,"retry-strips",base,reverse+one+stepper(cap,0,"fault RSF"))
    require("Fault slip/state budget exceeded" in (limited.parent/"run.log").read_text(), "Growing slip cap did not reject the trial")
    with (limited/"balance-0001.csv").open() as f:
        h = float(next(csv.DictReader(f))["time"])
    require(0 < h < .005, "Growing slip retry did not shorten the step")
    for row in sections(limited,1):
        rate = float(row["V"])
        age = 10.*math.exp(-abs(rate)*h/.01) + (h if rate==0 else -.01/abs(rate)*math.expm1(-abs(rate)*h/.01))
        require(abs(rate)*h/.01 <= cap and math.isclose(float(row["slip"]),h*rate,rel_tol=1e-14), "Incorrect signed full-duration strip slip")
        require(math.isclose(float(row["age"]),age,rel_tol=1e-14), "Incorrect full-duration strip age")
    duration_case = (one+section("Material model",section("Phase field fault","set Radiation damping = 0.75"))+
                     "set Nonlinear solver failure strategy = cut timestep size\n"+stepper(.1,.1,"fault RSF"))
    recovered = run(exe,dest,"inadmissible-duration",base,duration_case)
    split = run(exe,dest,"clean-duration",base,duration_case+"set Maximum first time step = 0.0025\n")
    log = (recovered.parent/"run.log").read_text()
    require("Repeating the current time step" in log and "Fault slip/state budget exceeded" not in log, "Wrong cutback path for an inadmissible constitutive solve")
    require(all(f.read_bytes()==(split/f.name).read_bytes() for f in recovered.glob("*.csv")), "Recovered native tables differ from the clean half-step")
    print("Constitutively inadmissible 5 ms trial retried at 2.5 ms; all accepted tables matched clean.")
    print("Growing reversed trial",min(rates),"m/s, slip budget",cap,"retry",h,"s; all nine full-duration increments checked.")


def advected(exe,dest,plugin):
    base = ROOT/"benchmarks/layered_moving_fault/layered.prm"
    setting = (f"set Additional shared libraries = {plugin}\n"+
               section("Material model",section("Phase field fault",
                   f"set Fault vertex file = {base.parent/'layered_vertices.txt'}"))+
               section("Postprocess","set List of postprocessors = layered background, mature fault statistics"))
    ordinary = run(exe,dest,"ordinary",base,setting)
    controlled = run(exe,dest,"controlled",base,setting+stepper(1.,1.))
    a,b=labels(ordinary),labels(controlled)
    require(a==b and len(a)==3 and a[0][0][1:3]!=a[-1][0][1:3], "Advected parent, vertex motion or clocks differ")
    for old,new in zip(a,a[1:]):
        for x,y in zip(old,new):
            require(x[0]==y[0] and abs(y[4]-x[4])/.008<1 and abs(math.log(y[3]/x[3]))<1, "Advected trace changed ID or violated a budget")
    print("Advected trace matched by ID at three clocks on",len(a[0]),"moving vertices.")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("executable",type=Path)
    p.add_argument("directory",type=Path)
    p.add_argument("--growing-plugin",type=Path)
    p.add_argument("--moving-plugin",type=Path)
    a=p.parse_args()
    a.directory.mkdir(parents=True)
    if a.growing_plugin:
        growing(a.executable.resolve(),a.directory.resolve(),a.growing_plugin.resolve())
    elif a.moving_plugin:
        advected(a.executable.resolve(),a.directory.resolve(),a.moving_plugin.resolve())
    else:
        mature(a.executable.resolve(),a.directory.resolve())
