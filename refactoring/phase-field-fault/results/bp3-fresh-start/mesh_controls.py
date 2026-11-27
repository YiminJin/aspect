"""Perturb only the grading of the actual production box and audit each measured region."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("package",type=Path)
p.add_argument("inventory_plugin",type=Path)
p.add_argument("destination",type=Path)
a=p.parse_args()
stage=a.package.resolve()
sys.path.insert(0,str(stage/"tools"))
from startup import check_mesh
dest=a.destination.resolve()
dest.mkdir()
domain=(-60000.,90000.,0.,50000.)
for region,condition,kind in [
        ("support","abs(x-14434)<100 && abs(y-25000)<100","Maximum"),
        ("base","abs(x-29700)<100 && y<25","Maximum"),
        ("surface","abs(x-800)<100 && y>49975","Maximum"),
        ("outside","x< -30000 && y<30000","Minimum")]:
    text=(stage/"production.prm").read_text()
    prm=dest/(region+".prm")
    case=dest/region
    extra="maximum refinement function" if kind=="Maximum" else "minimum refinement function"
    expr=f"if({condition},8,9)" if kind=="Maximum" else f"if({condition},3,0)"
    prm.write_text(text + f"""
set Additional shared libraries = ./plugins/libbp3_qd.so, {a.inventory_plugin.resolve()}
set Output directory = {case}
subsection Mesh refinement
  set Strategy = BP3 historical footprint, {extra}
  subsection {kind} refinement function
    set Coordinate system = cartesian
    set Function expression = {expr}
  end
end
""")
    with prm.with_suffix(".log").open("w") as log:
        r=subprocess.run([stage/"plugins/aspect-release",prm],cwd=stage,stdout=log,stderr=subprocess.STDOUT,
                         env=dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",DEAL_II_NUM_THREADS="1"))
    assert r.returncode,"The mesh-only observer should have stopped before DoF allocation"
    path=case/"bp3-mesh.txt"
    with path.open() as f: row,=csv.DictReader(f,delimiter=" ")
    counts={int(row["cells"])}
    if region=="outside":
        with (stage/"measured-mesh/bp3-mesh.txt").open() as f: baseline,=csv.DictReader(f,delimiter=" ")
        assert counts!={int(baseline["cells"])}
        assert all(row[k]==baseline[k] for k in row if k!="cells")
        check_mesh(path,counts,domain)
    else:
        assert int(row["bad_"+region])>0 and "does not resolve the physical support" in prm.with_suffix(".log").read_text()
        try: check_mesh(path,counts,domain)
        except ValueError as error: assert region in str(error)
        else: raise AssertionError(f"Missed coarsened {region}")
    print(region, row,flush=True)
