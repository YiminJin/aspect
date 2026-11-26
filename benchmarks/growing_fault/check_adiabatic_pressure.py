"""Run the installed depth profile and compare the compression used by the two strip operations."""
import argparse
import math
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from check_physical_inputs import growing
from check_repairs import subsection
from compare import rows, continuation, compare_values


def adiabatic_constants(surface_pressure, gradient):
    """Set the parsed pressure intercept (Pa) and depth gradient (Pa/m)."""
    constants = f'set Function constants = p0={surface_pressure}, dp_dz={gradient}'
    return subsection('Adiabatic conditions model', subsection('Function', constants))


def resistance(v, pressure, h=.0025):
    x = h * abs(v) / .01
    age = 10 * math.exp(-x) + h * (-math.expm1(-x)/x if x else 1.)
    return pressure*.015*math.asinh(v/.002*math.exp((.6+.02*math.log(.1*age))/.015)) + 1000*v


def check_profile(out, p0, slope):
    pressure = rows([out/'friction-pressure-0001.csv'], ('section',))
    strip = rows([out/'sections-0001.csv'], ('section',))
    point = rows(out.glob('points-0001-r*.csv'), ('cell','q'))
    if set(pressure) != set(strip) or len(pressure)!=9:
        raise ValueError('Missing strip pressure')
    extent = math.sqrt(3)+.5
    for (k,),r in sorted(pressure.items()):
        # Along-seed coordinate relative to the box center. The bottom/top faces
        # cut it at -1/+1 m. The material strips span the projection of the whole box.
        u = .5*(max(-1.,extent*(k/9-.5)) + min(1.,extent*((k+1)/9-.5)))
        x,y = 1+u*math.sqrt(3)/2, .5+u*.5
        n = p0+slope*(1-y)
        v = float(strip[k,]['V'])
        np.testing.assert_allclose([float(r[col]) for col in ('x','y','depth','compression')],
                                   [x,y,1-y,n],atol=5e-13,rtol=0.)
        np.testing.assert_equal(float(r['V']),v)
        np.testing.assert_allclose(float(strip[k,]['force']),resistance(v,n),atol=1e-10,rtol=0.)
        # Independently invert the signed stage-aged law at the published force.
        # This relative check also applies when two layouts select different angles.
        force = float(strip[k,]['force'])
        lo, hi = (-1., 1.) if force else (0., 0.)
        for _ in range(200):
            mid = (lo+hi)/2.
            if resistance(mid,n) < force:
                lo = mid
            else:
                hi = mid
        np.testing.assert_allclose(v,(lo+hi)/2.,rtol=1e-10,atol=0.)
        if slope and k in (0, 8):
            pts=[q for q in point.values() if int(q['section'])==k]
            centroid=sum(float(q['weight'])*float(q['y']) for q in pts)/sum(float(q['weight']) for q in pts)
            if abs(n-(p0+slope*(1-centroid)))<.1 or abs(n-(p0+slope*.25))<1:
                raise ValueError('Seed-tip or area-centroid substitution is indistinguishable')
        print(f"strip {k} depth {1-y:.9f} expected {n:.12g} consumed {r['compression']} Pa V {v:.12g} m/s")
    return pressure


def check_score(out):
    samples = ((.03,.8,.4,-70.,210.),(.02,.6,.7,155.,140.))
    pressures=rows([out/'friction-pressure-0001.csv'],('section',))
    observed = []
    for path in sorted(out.glob('operator-r*.txt')):
        angles=[]
        for line in path.read_text().splitlines():
            if not line.startswith('pressure-score '):
                continue
            _, strip, angle, f, score=line.split()
            k=int(strip)
            beta=float(angle)
            # Pointwise reference, independent bisection on the signed rate.
            def oracle(n):
                tau=[u*math.sin(2*beta)+v*math.cos(2*beta) for a,w,F,u,v in samples]
                ds=(math.sqrt(3)+.5)/9/math.cos(beta)
                gbh_ds=1e6/(1+.0025*1e6/1e8)*.0025*ds
                free=sum(a*F*t for (a,w,F,u,v),t in zip(samples,tau))/gbh_ds
                mu=sum(a*F for a,w,F,u,v in samples)/gbh_ds
                lo,hi=sorted([0.,free])
                for _ in range(210):
                    mid=.5*(lo+hi)
                    if mid+mu*resistance(mid,n)>free:
                        hi=mid
                    else:
                        lo=mid
                force=resistance(.5*(lo+hi),n)
                return force,sum(a*w*(1-F)**2*(t-force)**2 for (a,w,F,u,v),t in zip(samples,tau))
            correct=oracle(float(pressures[k,]['compression']))
            np.testing.assert_allclose([float(f),float(score)],correct,atol=1e-10,rtol=1e-11)
            if abs(correct[1]-oracle(200.)[1])<.01:
                raise ValueError('Candidate score is insensitive to its section compression')
            angles.append((k,beta))
            print('eligible candidate',k,beta,'rad, force/score',correct)
        if not angles or not {0.,.23}.issubset(set(a for k,a in angles)):
            raise ValueError('Missing candidate score controls')
        observed.append(angles)
    if not observed or any(candidate!=observed[0] for candidate in observed):
        raise ValueError('Missing rank score or eligibility')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('aspect','plugin','destination'):
        parser.add_argument(arg,type=lambda p:Path(p).resolve())
    parser.add_argument('group',choices=['profile','constant','lifecycle'])
    parser.add_argument('--ranks',type=int,choices=(1,2),default=1)
    parser.add_argument('--mpiexec',default='mpirun')
    parser.add_argument('--serial',type=Path)
    parser.add_argument('--stage-b',type=Path)
    parser.add_argument('--pressure-plugin',type=lambda p:Path(p).resolve())
    args=parser.parse_args()
    args.destination.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parent
    def case(name, p=200, slope=50, extras='', end=.005, probe=False, plugin=None, resume=None, diagnostic=None):
        run=args.destination/name
        run.mkdir()
        shutil.copy(root/'onecut.prm', run)
        source='include onecut.prm\n'
        if p is not None:
            source=(root/'adiabatic_pressure.prm').read_text() + adiabatic_constants(p, slope)
        if resume:
            shutil.copytree(resume/'restart/02',run/'output/restart/02')
            extras+='\nset Resume computation = true\n'+subsection('Checkpointing','set Resume checkpoint = 2')
        source+=f'\nset Additional shared libraries = {args.plugin}'+(f', {plugin}' if plugin else '')+'\n'
        source+=subsection('Postprocess','set List of postprocessors = growing fault '+('samples' if probe else 'history'))
        (run/'input.prm').write_text(source+f'\nset End time = {end}\nset Output directory = output\n'+extras)
        launch=[] if args.ranks==1 else [args.mpiexec,'-np',str(args.ranks)]
        with (run/'log').open('w') as log:
            proc=subprocess.run([*launch,args.aspect,'input.prm'],cwd=run,stdout=log,stderr=subprocess.STDOUT,
                                env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',DEAL_II_NUM_THREADS='1'),timeout=180)
        if diagnostic:
            message=' '.join((run/'log').read_text().split())
            if not proc.returncode or diagnostic not in message or list((run/'output').glob('balance-*.csv')):
                raise ValueError(f'{name}: expected {diagnostic}, see {run}/log')
            print(name,'rejected:',diagnostic)
        else:
            proc.check_returncode()
        return run/'output'

    if args.group=='profile':
        affine=case('affine',probe=True)
        measured=check_profile(affine,200,50)
        check_score(affine)
        if args.serial:
            serial=args.serial/'affine/output'
            old=rows([serial/'friction-pressure-0001.csv'],('section',))
            for column in ('x','y','depth','compression','V'):
                print('MPI',column,compare_values(old,measured,column,1e-11 if column=='V' else 1e-13,1e-7))
            a,b=(rows([d/'sections-0001.csv'],('section',)) for d in [serial,affine])
            for rank in range(args.ranks):
                records=rows([affine/f'points-0001-r{rank}.csv'],('cell','q'))
                print('MPI rank',rank,'owned native points',len(records))
            for k in a:
                if float(a[k]['beta'])==float(b[k]['beta']):
                    np.testing.assert_allclose(float(a[k]['V']),float(b[k]['V']),atol=0.,rtol=1e-10)
            va,vb = (float(table[8,]['V']) for table in (a,b))
            print('strip 8 signed rates, absolute and relative difference:', va,vb,abs(vb-va),abs(vb/va-1.))
            failures=[]
            def compare(left,right,field,atol,rtol):
                try:
                    print('MPI',field,compare_values(left,right,field,atol,rtol))
                except ValueError as exc:
                    failures.append(str(exc))
            for field,atol in [('formed',0.),('force',1e-6),('beta',1e-7),('ds',1e-11),('age',1e-11),('slip',1e-11)]:
                compare(a,b,field,atol,1e-7)
            pts=[rows(d.glob('points-0001-r*.csv'),('cell','q')) for d in (serial,affine)]
            for col in ('section','x','y','weight','phi','tauxx','tauyy','tauzz','tauxy','Axx','Ayy','Azz','Axy',
                        'signed_shear','absolute_shear'):
                tol=1e-6 if col.startswith('tau') else 1e-9 if col=='phi' else 1e-10 if col[0] in 'Asa' and col!='section' else 1e-14
                compare(*pts,col,tol,1e-7 if tol>=1e-10 else 0.)
            balances=[rows([d/'balance-0001.csv'],('step',)) for d in (serial,affine)]
            for col in ('bulk','cohesive','fracture','Wext','Db','Dc','Df','phase_remainder'):
                compare(*balances,col,1e-11,1e-6)
            if failures:
                raise ValueError('Cross-rank strict comparisons: '+'; '.join(failures))
        return
    if args.group=='constant':
        if not args.stage_b or not args.pressure_plugin:
            parser.error('constant requires --stage-b and --pressure-plugin')
        base=None
        for n in [200,250]:
            out=case(str(n),n,0)
            check_profile(out,n,0)
            continuation(args.stage_b/str(n)/'output',out,[1])
            if n==200:
                base=out
        shifted=case('datum',200,0,plugin=args.pressure_plugin)
        check_profile(shifted,200,0)
        a,b=(rows(d.glob('fields-0001-r*.csv'),('cell','component','local')) for d in (base,shifted))
        shifts=[float(b[k]['value'])-float(a[k]['value']) for k in a if k[1]==2]
        np.testing.assert_allclose(shifts,100.,rtol=0.,atol=1e-6)
        v=[float(rows([d/'sections-0001.csv'],('section',))[0,]['V']) for d in (base,shifted)]
        np.testing.assert_allclose(v[0],v[1],rtol=1e-10,atol=0.)
        print('Stokes datum min/max shift (Pa)',min(shifts),max(shifts),'strip 0 rates',v)
        return
    save=subsection('Checkpointing','set Steps between checkpoint = 1\nset Number of checkpoints to keep = 4')
    clean=case('clean',extras=save,end=.015)
    resumed=case('resume',resume=clean,end=.015)
    continuation(clean,resumed,[2,3])
    for step in (2,3):
        a,b=(rows([d/f'friction-pressure-{step:04}.csv'],('section',)) for d in (clean,resumed))
        for col in ('x','y','depth','compression','V'):
            compare_values(a,b,col,1e-12,1e-12)
    # At 15 ms the previous frame has a turned, formed strip 8. Its incoming
    # graph leaves the axis at station 7 and exits the unit-high box on top.
    parent=rows([clean/'sections-0002.csv'],('section',))
    station=rows([clean/'friction-pressure-0003.csv'],('section',))[8,]
    beta=float(parent[8,]['beta'])
    if parent[8,]['formed']!='1' or abs(beta)<.1:
        raise ValueError('Missing bent accepted graph')
    tangent=np.array([math.sqrt(3)/2,.5])
    normal=np.array([-.5,math.sqrt(3)/2])
    xi7=float(parent[7,]['xi'])
    joint=np.array([1.,.5])+(xi7-(math.sqrt(3)+.5)/2)*tangent
    direction=tangent+math.tan(beta)*normal
    start=joint+(.5*(float(parent[8,]['xi'])-xi7))*direction
    end=start+min((np.array([2.,1.])-start)/direction)*direction
    center=.5*(start+end)
    np.testing.assert_allclose([float(station[s]) for s in ('x','y','compression')],
                               [*center,200+50*(1-center[1])],atol=5e-13,rtol=0.)
    print('Turned strip 8 at 15 ms:',*center,'m',station['compression'],'Pa')
    case('slope',slope=49,resume=clean,end=.015,diagnostic='profile differs from the checkpoint')
    case('closure',p=None,resume=clean,end=.015,diagnostic='source differs from the checkpoint')
    old=case('legacy',p=None,extras=save,end=.01)
    case('old-to-profile',resume=old,end=.015,diagnostic='source differs from the checkpoint')
    case('conflict',extras=growing('set Prescribed fault compression = 0'),diagnostic='omit Prescribed fault compression')
    case('empty-strip',extras=growing('set Forward seed tip = 1.4330127018922194,.95'),
         diagnostic='has no in-box fault segment for adiabatic pressure')
    case('negative',p=-1,slope=0,diagnostic='Invalid adiabatic fault compression -1 Pa at strip')
    at_surface=case('zero-surface',extras=subsection('Adiabatic conditions model',subsection('Function',
                    'set Function expression = 293;if(depth>0,200,0);1')))
    check_profile(at_surface,200,0)
    # Same filename, changed pressure column. Check the loaded interpolation
    # table, including knots at which no strip happened to sample the profile.
    table=args.destination/'profile.txt'
    ascii_profile=subsection('Adiabatic conditions model','set Model name = ascii data\n'+subsection(
        'Ascii data model', f'set Data directory = {args.destination}/\nset Data file name = profile.txt'))
    def write_table(middle):
        table.write_text(f'# POINTS: 3\ndepth pressure temperature density\n0 200 293 1\n.5 {middle} 293 1\n1 250 293 1\n')
    write_table(225)
    ascii_clean=case('ascii',extras=ascii_profile+save,end=.01)
    check_profile(ascii_clean,200,50)
    ascii_resume=case('ascii-resume',extras=ascii_profile,resume=ascii_clean,end=.01)
    continuation(ascii_clean,ascii_resume,[2])
    write_table(224)
    case('ascii-changed',extras=ascii_profile,resume=ascii_clean,end=.01,diagnostic='profile differs from the checkpoint')
    cut=growing('set Radiation damping = 1.15')+subsection('Termination criteria','set Termination criteria = end step\nset End step = 1')
    retry=case('retry',extras=cut+'\nset Nonlinear solver failure strategy = cut timestep size\n')
    half=case('half',extras=cut+'\nset Maximum first time step = .0025\nset Maximum time step = .0025\n')
    if (retry.parent/'log').read_text().count('Trying to cut the current time step')!=1:
        raise ValueError('Expected one pressure-dependent cutback')
    continuation(half,retry,[1])
    ns=[float(r['compression']) for r in rows([retry/'friction-pressure-0001.csv'],('section',)).values()]
    if not min(ns)*.005<1.15<max(ns)*.005 or max(ns)*.0025>=1.15:
        raise ValueError('Duration test does not separate the shallow and deep stations')
    # Reverse the gradient so checking only the first strip would admit 5 ms.
    reverse=case('reverse-retry',p=250,slope=-50,extras=cut+'\nset Nonlinear solver failure strategy = cut timestep size\n')
    reversed_clean=case('reverse-half',p=250,slope=-50,extras=cut+'\nset Maximum first time step = .0025\n')
    if (reverse.parent/'log').read_text().count('Trying to cut the current time step')!=1:
        raise ValueError('Deep strip escaped the all-section duration guard')
    continuation(reversed_clean,reverse,[1])
    print('5 ms shallow/deep damping bounds',min(ns)*.005,max(ns)*.005,'accepted 2.5 ms')


if __name__=='__main__':
    main()
