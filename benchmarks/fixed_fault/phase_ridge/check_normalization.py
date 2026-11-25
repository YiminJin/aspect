"""Fresh, single-cell tests for association loss and near-singular DGQ2 integration."""
import argparse
from pathlib import Path
import os
import subprocess

import numpy as np

from check import gauss, near, require


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("aspect","plugin","directory"):
        parser.add_argument(name,type=Path)
    args=parser.parse_args()
    root=args.directory.resolve()
    root.mkdir(parents=True)
    for case, phase, width, failure in [
            ("discarded", ".35", .1, "discards significant normal weight"),
            ("constant", ".35", .5, None),
            ("scaled", ".35", .5, None),
            ("pole", ".99*(1-4*(x-.5)^2)", .5, None),
            ("unequal", ".55*(1-4*(x-.5)^2)*(1-.4*y^2)", .5, None),
            ("unresolved", ".999999999999*(1-4*(x-.5)^2)", .5, "did not converge within 16 subdivisions"),
            ("singular", "1", .5, "outside the mature degradation domain"),
            ("ambiguous-contact", ".3+.1*y", .5, "ambiguous stationary phase maximum")]:
        magnitude=4 if case=="scaled" else 1
        dest=root/case
        dest.mkdir()
        output=dest/"output"
        prm=dest/"input.prm"
        prm.write_text(f"""include $ASPECT_SOURCE_DIR/benchmarks/fixed_fault/stationary.prm
set Additional shared libraries = {args.plugin.resolve()}
set Output directory = {output}
set End time = 0
subsection Mesh refinement
  set Initial global refinement = 0
end
subsection Material model
  set Model name = prescribed ridge diagnostic
  subsection Phase field fault
    set Reference speed = .005
    set Degradation magnitude = {magnitude}
    set Fault vertices = .5,0; .5,.3; .5,1
    set Fit trace to phase = {str(case=="ambiguous-contact").lower()}
    set Profile half width = {width}
  end
end
subsection Initial composition model
  subsection Function
    set Function expression = 0;0;0;{phase}
  end
end
subsection Postprocess
  set List of postprocessors = ridge probe
end
""")
        env=dict(os.environ,DEAL_II_NUM_THREADS="1",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
        with (dest/"run.log").open("w") as stream:
            process=subprocess.run([args.aspect.resolve(),prm],cwd=dest,env=env,stdout=stream,stderr=subprocess.STDOUT)
        if failure:
            log=(dest/"run.log").read_text()
            require(process.returncode!=0 and failure in log,f"Missing rejection in {dest}")
            if case=="discarded":
                require("segment 0 at station 0 (omitted fraction 0.800000)" in " ".join(log.split()),
                        "Did not identify the station and its 80% loss")
            print(case, failure,flush=True)
            continue
        process.check_returncode()
        values=np.loadtxt(output/"labels-0-0.txt",skiprows=1)[:,3]
        nodes=np.array([0.,.3,1.])
        mass=np.zeros((3,3))
        rhs=np.zeros(3)
        x,w=gauss(256)
        for j,(lo,hi) in enumerate(zip(nodes[:-1],nodes[1:])):
            for q,v in zip(*gauss(3)):
                y=lo+(hi-lo)*q
                if case=="pole": d=.99*(1-4*(x-.5)**2)
                elif case=="unequal": d=.55*(1-4*(x-.5)**2)*(1-.4*y*y)
                else: d=np.full_like(x,.35)
                shape=np.array([1-q,q])
                rhs[j:j+2]+=(hi-lo)*v*shape*(w@(magnitude*d*(1+d)/(1-d)**2))
                mass[j:j+2,j:j+2]+=(hi-lo)*v*np.outer(shape,shape)
        reference=np.linalg.solve(mass,rhs)
        near(values,reference,1.e-10,1.e-12)
        if case=="scaled":
            raw=np.loadtxt(root/"constant/output/labels-0-0.txt",skiprows=1)[:,3]
            near(values,4*raw,1.e-12)
            for directory in (output,root/"constant/output"):
                chi=np.loadtxt(directory/"bulk-0-0.txt",skiprows=1,usecols=6)
                near(chi,np.ones_like(chi),1.e-12)
            print("Unit-width normal, m=1 and m=4:",raw[0],values[0],"m; normalized chi=1 m^-1 throughout")
        if case=="pole":
            q12,w12=gauss(12)
            d=.99*(1-4*(q12-.5)**2)
            fixed=w12@(d*(1+d)/(1-d)**2)
            near(fixed,755.6377468,1.e-6)
            print("unsplit Q12",fixed,"m; checked integral",values[0],"m")
        print(case,values,"max error vs independent Q256 plus dense mass",max(abs(values-reference)),flush=True)


if __name__=="__main__":
    main()
