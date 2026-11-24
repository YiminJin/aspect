"""Observe strip 15 at its first virgin-plane decision; replay its reduced score.

Capture is specific to the pinned x86-64 Linux release and Open MPI ABI. GDB
reads the result of the original Allreduce. The offline evaluation uses the
complete-strip force and centered localized second moment of each capture.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess

import mpmath as mp

SYMBOLS = {
    'parent': '_ZN6aspect12GrowingFault14MidpointSystem25select_prospective_planesERKSt6vectorIdSaIdEE',
    'donor-parent': '_ZN6aspect8internal15BulkTipCouplingILi2EE20update_virgin_planesERKN6dealii16TrilinosWrappers3MPI11BlockVectorE',
}


def capture(runs, dest):
    dest.mkdir(parents=True,exist_ok=False)
    # Adapt the existing breakpoint for 18 strips, with 9+3 entries per strip.
    original = Path(__file__).resolve().parents[2]/'strip8/reductions.py'
    script = original.read_text()
    for old,new in [('self.count//9','self.count//18'),('values[8*width:]','values[15*width:16*width]'),
                    ('{81,27}','{162,54}')]:
        if script.count(old)!=1: raise ValueError(f'Unexpected debugger template at {old}')
        script = script.replace(old,new)
    observer = dest/'reductions.py'
    observer.write_text(script)
    for case,symbol in SYMBOLS.items():
        out=dest/case
        out.mkdir()
        command=json.loads((runs/case/'run.json').read_text())['command']
        prm=out/'input.prm'
        prm.write_text((runs/case/'input.prm').read_text()+f'\nset Output directory = {out}/output\n')
        (out/'observe.gdb').write_text(f"""set pagination off
set debuginfod enabled off
set confirm off
set disable-randomization off
set print thread-events off
source {observer}
break *{symbol}
commands
silent
disable 1
python Reductions() if os.environ.get('OMPI_COMM_WORLD_RANK')=='0' else None
continue
end
run
""")
        cmd=[*command[:-2],'gdb','-nx','-batch','-return-child-result','-x','observe.gdb','--args',command[-2],str(prm)]
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',DEAL_II_NUM_THREADS='1',OMP_PROC_BIND='spread')
        with (out/'log').open('w') as log:
            subprocess.run(cmd,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180,check=True)
    write_moments(runs,dest)


def write_moments(runs,dest):
    with (dest/'moments.csv').open('w') as stream:
        writer=csv.writer(stream,lineterminator="\n")
        writer.writerow(['source','width_m','angle_rad',*[f'm{i}' for i in range(9)],'uu','uv','vv'])
        for case in SYMBOLS:
            folder=runs/case/'output'
            planes='tip-planes-0001.csv' if case=='donor-parent' else 'sections-0001.csv'
            with (folder/planes).open() as file:
                plane,=[r for r in csv.DictReader(file) if r['section']=='15']
            if case=='donor-parent':
                width=plane['reference_dxi']
            else:
                with (folder/'sections-0000.csv').open() as file:
                    width,=[r['ds'] for r in csv.DictReader(file) if r['section']=='15']
            values=[float(s) for n in (162,54) for s in (dest/case/f'live-{n}').read_text().split()]
            if len(values)!=12: raise ValueError('Missing live reduced moments')
            writer.writerow([case,width,plane['beta'],*values])


def score(row, beta):
    # Each recorded binary64 value is an exact input to the high-precision
    # control. The denominator uses the candidate, rather than seed, length.
    m=[mp.mpf(float(row[f'm{i}'])) for i in range(9)]
    uu,uv,vv=[mp.mpf(float(row[k])) for k in ('uu','uv','vv')]
    h=mp.mpf(.005)/2
    G=mp.mpf(1e6)
    scale=h*G/(1+h*G/1e8)*mp.mpf(float(row['width_m']))/mp.cos(beta)
    s,c=mp.sin(2*beta),mp.cos(2*beta)
    free=(m[0]*s+m[1]*c)/scale
    mobility=m[2]/scale
    def resistance(v):
        k=abs(v)/mp.mpf(.01)
        age=10*mp.exp(-k*h)-mp.expm1(-k*h)/k if k else 10+h
        return 200*mp.mpf(.015)*mp.asinh(v/(2*mp.mpf(.001))*mp.exp(
            (mp.mpf(.6)+mp.mpf(.02)*mp.log(age*mp.mpf(.001)/mp.mpf(.01)))/mp.mpf(.015)))+1000*v
    v=mp.findroot(lambda v:v+mobility*resistance(v)-free,(mp.mpf('5e-14'),mp.mpf('2e-13')))
    f=resistance(v)
    return f,uu*s*s+2*uv*s*c+vv*c*c+m[8]*((m[6]*s+m[7]*c)/m[8]-f)**2


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('moments',type=Path,help='CSV to read, or output directory with --capture')
    parser.add_argument('--capture',type=Path,help='Directory containing observed parent and donor-parent runs')
    args=parser.parse_args()
    if args.capture:
        capture(args.capture.resolve(),args.moments.resolve())
    else:
        mp.mp.dps=65
        with args.moments.open() as stream:
            rows=list(csv.DictReader(stream))
        angles=[mp.mpf(0),*[mp.mpf(float(r['angle_rad'])) for r in rows]]
        for row in rows:
            values=[score(row,a) for a in angles]
            for a,(f,S) in zip(angles,values):
                print(row['source'],mp.nstr(a,18),mp.nstr(f,18),mp.nstr(S,18))
            print('target-minus-donor score',mp.nstr(values[1][1]-values[2][1],12))
