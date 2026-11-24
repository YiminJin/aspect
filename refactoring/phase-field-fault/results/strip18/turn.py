"""Observe strip 15 at its first virgin-plane decision; replay its reduced score.

Capture is specific to the pinned x86-64 Linux release and Open MPI ABI. GDB
reads the result of the original Allreduce. The offline evaluation uses the
complete-strip force and centered localized second moment of each capture.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess

import mpmath as mp

SYMBOLS = {
    'parent': '_ZN6aspect12GrowingFault14MidpointSystem25select_prospective_planesERKSt6vectorIdSaIdEE',
    'donor-parent': '_ZN6aspect8internal15BulkTipCouplingILi2EE20update_virgin_planesERKN6dealii16TrilinosWrappers3MPI11BlockVectorE',
}

# Instructions immediately following angles[k] = selected in the pinned
# executables. Read the stored value, even on calls after the first reduction.
STORES = {
    'parent': ('8769c98b64923906964379ef78917567eb93edc40da93b9d38765b187184022d',
               '0x19ee', '$esi', '*(double*)$rcx'),
    'donor-parent': ('e640fa627141ea98343fee14526c532f005281d336da32aa2d2f3467c5589b10',
                     '0x1fb5', '$r9', '((double*)$r10)[$r9]'),
}


def capture(runs, dest, cross_inputs=None):
    dest.mkdir(parents=True,exist_ok=False)
    observer=Path(__file__).resolve().with_name('selector_buffers.py')
    choices=[]
    for case,symbol in SYMBOLS.items():
        out=dest/case
        out.mkdir()
        command=json.loads((runs/case/'run.json').read_text())['command']
        digest,offset,index,stored=STORES[case]
        with open(command[-2],'rb') as binary:
            if hashlib.file_digest(binary,'sha256').hexdigest()!=digest:
                raise ValueError(f'Angle-store observer requires the indexed {case} executable')
        prm=out/'input.prm'
        prm.write_text((runs/case/'input.prm').read_text()+f'\nset Output directory = {out}/output\n')
        opposite='donor-parent' if case=='parent' else 'parent'
        inputs=str(cross_inputs/opposite) if cross_inputs else None
        monitor='1' if cross_inputs else '$observer'
        # Both debuggers exit together immediately after the cross-input
        # decision. Only ordinary unmodified captures run on to publication.
        record=(f'printf "replay-write %d %d %.17g\\n", $rank, $calls, {stored}\nquit 0'
                if cross_inputs else f'printf "strip15-write %d %.17g\\n", $calls, {stored}\ncontinue')
        (out/'observe.gdb').write_text(f"""set pagination off
set debuginfod enabled off
set confirm off
set disable-randomization off
set print thread-events off
source {observer}
set $calls = 0
python gdb.set_convenience_variable('rank', int(os.environ['OMPI_COMM_WORLD_RANK']))
set $observer = ($rank == 0)
break *{symbol}
commands
silent
set $calls = $calls + 1
if $calls == 1 && {monitor}
python Reductions({inputs!r})
end
continue
end
break *({symbol}+{offset}) if {monitor} && {index} == 15
commands
silent
{record}
end
run
""")
        cmd=[*command[:-2],'gdb','-nx','-batch','-return-child-result','-x','observe.gdb','--args',command[-2],str(prm)]
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',DEAL_II_NUM_THREADS='1',OMP_PROC_BIND='spread')
        with (out/'log').open('w') as log:
            run=subprocess.run(cmd,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
        transcript=(out/'log').read_text()
        # A deliberate debugger exit inside the selector precedes MPI_Finalize.
        if not (cross_inputs and run.returncode==1 and 'exited without calling "finalize"' in transcript):
            run.check_returncode()
        if cross_inputs:
            for n in (162,54):
                values=(cross_inputs/opposite/f'live-{n}').read_text().split()
                expected=struct.pack(f'={n//18}d',*map(float,values))
                for rank in (0,1):
                    if (out/f'live-{n}-used-r{rank}').read_bytes()!=expected:
                        raise ValueError(f'Rank {rank} did not use the supplied {n}-entry reduction')
            hits=sorted(re.findall(r'^replay-write ([01]) (\d+) (\S+)$',transcript,re.M))
            if [(rank,call) for rank,call,_ in hits]!=[('0','1'),('1','1')] or hits[0][2]!=hits[1][2]:
                raise ValueError(f'Expected a matching first-call decision on both ranks, saw {hits}')
            if list((out/'output').glob('*-0001*.csv')):
                raise ValueError('Common-input probe advanced past selection')
            choices.append((opposite,case,hits[0][2]))
    if cross_inputs:
        with (dest/'choices.csv').open('w') as stream:
            writer=csv.writer(stream,lineterminator='\n')
            writer.writerow(['input_moments','production_selector','beta_rad'])
            writer.writerows(choices)
    else:
        write_moments(runs,dest)


def write_moments(runs,dest,output=None):
    records=[]
    for case in SYMBOLS:
        captured=dest/case
        table='tip-planes-0001.csv' if case=='donor-parent' else 'sections-0001.csv'
        def plane(path):
            with path.open() as file:
                row,=[r for r in csv.DictReader(file) if r['section']=='15']
            return row
        accepted=plane(captured/'output'/table)
        ordinary=plane(runs/case/'output'/table)
        if float(accepted['beta'])!=float(ordinary['beta']):
            raise ValueError(f'{case}: captured angle differs from ordinary acceptance')
        writes=re.findall(r'^strip15-write (\d+) (\S+)$',(captured/'log').read_text(),re.M)
        if [(int(i),float(a)) for i,a in writes]!=[(1,float(accepted['beta']))]:
            raise ValueError(f'{case}: strip 15 stores {writes} disagree with first-call acceptance')
        width=(accepted['reference_dxi'] if case=='donor-parent'
               else plane(captured/'output/sections-0000.csv')['ds'])
        values=[float(s) for n in (162,54) for s in (captured/f'live-{n}').read_text().split()]
        if len(values)!=12: raise ValueError('Missing live reduced moments')
        records.append([case,width,accepted['beta'],*values])
    with (output or dest/'moments.csv').open('w') as stream:
        writer=csv.writer(stream,lineterminator="\n")
        writer.writerow(['source','width_m','angle_rad',*[f'm{i}' for i in range(9)],'uu','uv','vv'])
        writer.writerows(records)


def candidate(row, beta):
    """Signed full-strip root and centered area-weighted score at fixed midpoint inputs."""
    m=[mp.mpf(float(row[f'm{i}'])) for i in range(9)]
    uu,uv,vv=[mp.mpf(float(row[k])) for k in ('uu','uv','vv')]
    h=mp.mpf(.005)/2
    ds=mp.mpf(float(row['width_m']))/mp.cos(beta)
    scale=h*10**6/(1+h*10**6/10**8)*ds
    s,c=mp.sin(2*beta),mp.cos(2*beta)
    free=(m[0]*s+m[1]*c)/scale
    mobility=m[2]/scale
    # Invert the signed rate in asinh coordinates. The physical bracket is
    # [0,Vfree] (reversed for negative Vfree), even on the cold-creep branch.
    direct,evolution,dc,vref=map(mp.mpf,(.015,.02,.01,.001))
    locked_age=10+h
    creep=2*vref*mp.exp(-(.6+evolution*mp.log(locked_age*vref/dc))/direct)
    def terms(z):
        v=creep*mp.sinh(z)
        x=h*abs(v)/dc
        theta=10*mp.exp(-x)+h*(-mp.expm1(-x)/x if x else 1)
        f=200*direct*mp.asinh(mp.sinh(z)*(theta/locked_age)**(evolution/direct))+1000*v
        return v,f
    def equation(z):
        v,f=terms(z)
        return v+mobility*f-free
    lo,hi=sorted((mp.mpf(0),mp.asinh(free/creep)))
    if equation(lo)>0 or equation(hi)<0:
        raise ValueError('Signed section root is outside its physical bracket')
    z=mp.findroot(equation,(lo,hi),solver='anderson') if free else mp.mpf(0)
    v,f=terms(z)
    # Check the bracketed solution in the original signed rate equation,
    # including constant-rate aging through the half-duration.
    k=abs(v)/dc
    age=10*mp.exp(-k*h)-mp.expm1(-k*h)/k if k else locked_age
    direct_force=200*direct*mp.asinh(v/(2*vref)*mp.exp(
        (mp.mpf(.6)+evolution*mp.log(age*vref/dc))/direct))+1000*v
    residual=v+mobility*direct_force-free
    if not lo<=z<=hi or abs(residual)>mp.mpf('1e-50'):
        raise ValueError(f'Unresolved section root at {beta} rad: {residual} m/s')
    centered=uu*s*s+2*uv*s*c+vv*c*c+m[8]*((m[6]*s+m[7]*c)/m[8]-f)**2
    return dict(ds=ds,free=free,rate=v,force=f,residual=residual,score=centered)


def score(row,beta):
    evaluated=candidate(row,beta)
    return evaluated['force'],evaluated['score']


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('moments',type=Path,help='CSV to read, or output directory with --capture')
    parser.add_argument('--capture',type=Path,help='Directory containing observed parent and donor-parent runs')
    parser.add_argument('--cross-inputs',type=Path,help='Replay the other captured input and stop at selection')
    args=parser.parse_args()
    if args.capture:
        capture(args.capture.resolve(),args.moments.resolve(),args.cross_inputs.resolve() if args.cross_inputs else None)
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
