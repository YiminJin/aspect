"""Exercise the 5A repair cases through ordinary ASPECT parameter files."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

from check_section_publication import check as check_cold
from compare import continuation, rows


def subsection(name, settings):
    return f'\nsubsection {name}\n{settings}\nend\n'


def function(expression):
    return subsection('Function', f'set Function expression = {expression}')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for arg in ('aspect','plugin','destination'):
        ap.add_argument(arg,type=lambda p:Path(p).resolve())
    ap.add_argument('group',choices=['friction','constraints','mpi','lifecycle','companion','geometry'])
    ap.add_argument('--legacy-snapshot',type=Path,help='generation 03 of a physical v1 checkpoint without a balance')
    ap.add_argument('--pressure-plugin',type=lambda p:Path(p).resolve())
    ap.add_argument('--mpiexec',default='mpirun')
    args=ap.parse_args()
    root=args.destination
    root.mkdir(parents=True,exist_ok=False)
    base=Path(__file__).with_name('onecut.prm').read_text()
    observe=subsection('Postprocess','set List of postprocessors = midpoint publication, velocity statistics')
    def run(name, extra='', *, source=base, expected='', ranks=1, end='.005', snapshot=None, extra_plugin=None):
        path=root/name
        path.mkdir()
        output=path/'output'
        input_file=path/'case.prm'
        if snapshot:
            shutil.copytree(snapshot,output/'restart'/snapshot.name)
            extra+='\nset Resume computation = true\n'+subsection('Checkpointing',f'set Resume checkpoint = {int(snapshot.name)}\nset Number of checkpoints to keep = 4')
        libs=str(args.plugin)+(f', {extra_plugin}' if extra_plugin else '')
        input_file.write_text(source+f'\nset Output directory = {output}\nset End time = {end}\nset Additional shared libraries = {libs}\n'+observe+extra)
        cmd=[str(args.aspect),str(input_file)]
        if ranks>1:
            cmd=[args.mpiexec,'--map-by','core','--bind-to','core','-np',str(ranks),*cmd]
        env=dict(os.environ,OMP_NUM_THREADS='1',DEAL_II_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        with (path/'log').open('w') as log:
            code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=150).returncode
        text=(path/'log').read_text()
        if expected:
            if not code or expected not in text or list(output.glob('balance-*.csv')):
                raise ValueError(f'{name}: expected {expected!r} before physical publication, exit {code}; see {path}/log')
        elif code:
            raise RuntimeError(f'{name}: exit {code}; see {path}/log')
        print(name,'ranks',ranks,'exit',code,expected,flush=True)
        return output

    wall=lambda expr:subsection('Boundary velocity model',function(expr))
    phase=lambda expr:subsection('Initial composition model',function('0;0;0;'+expr))
    gravity=lambda expr:subsection('Gravity model','set Model name = function\n'+function(expr))
    if args.group=='friction':
        cold=run('cold',subsection('Material model',subsection('Growing fault','set Reference friction coefficient = .9')))
        check_cold(cold)
        one=run('fully-damaged',phase('1'))
        fields=[rows(one.glob(f'fields-{step:04}-r*.csv'),('cell','component','local')) for step in (0,1)]
        if any(fields[0][k]['value']!=fields[1][k]['value'] for k in fields[0] if k[1]==7):
            raise ValueError('Changed incoming coefficients at the fully damaged obstacle')
        for step in (0,1):
            pts=rows(one.glob(f'points-{step:04}-r*.csv'),('cell','q'))
            if max(abs(float(r['phi'])-1) for r in pts.values())>3.e-14:
                raise ValueError('Changed fully damaged polynomial')
        run('invalid-phase',phase('1.001'),expected='Invalid initial phase mean')
    if args.group=='constraints':
        run('flux',wall('1e-5*x;0'),expected='Incompatible prescribed net velocity flux')
        run('affine',wall('1e-5*x;-1e-5*y'))
        run('dilation',subsection('Formulation','set Enable prescribed dilation = true')+
            subsection('Prescribed dilation','set List of model names = function\n'+function('1e-3*(x-1)')),
            expected='incompressible continuity without prescribed dilation')
        run('mass',subsection('Formulation','set Mass conservation = reference density profile'),
            expected='incompressible continuity without prescribed dilation')
        run('gravity-layer',gravity('0;if(y<0.1,1000,0)'),expected='zero gravity at every native integration point')
        run('direct',subsection('Solver parameters',subsection('Stokes solver parameters','set Stokes solver type = direct solver')),
            expected='separate velocity and pressure blocks')
        run('iteration-output',subsection('Postprocess','set Run postprocessors on nonlinear iterations = true'),expected='iteration postprocessing is unsupported')
        if not args.pressure_plugin:
            raise ValueError('Constraint group needs the pressure_constraint test plugin')
        datum=run('datum',extra_plugin=args.pressure_plugin)
        measured=float((datum/'published-constraints').read_text().splitlines()[-1].split()[1])
        if measured!=100.: raise ValueError(f'Published pressure datum {measured}')
        print('Published pressure DoF 306:',measured,'Pa')
        drive=subsection('Prescribed solution','set List of model names = velocity function\n'+
                         subsection('Velocity function',subsection('Indicator function',
                         'set Function expression = if(abs(x-1)+abs(y-.5)<1e-8,1,0);0')+function('1e-5;0')))
        run('unresolved-interior-drive',wall('0;0')+drive,expected='Unresolved native slip rate on strip')
        # With mu0=.6, opposing point increments lose the sign of the extremely
        # small mean. Keep the same actuator and walls, and resolve the creep
        # rate separately to test which prescribed velocities supplied work.
        actuator=run('interior-drive',wall('0;0')+drive+
                     subsection('Material model',subsection('Growing fault','set Reference friction coefficient = .3')))
        wall_max=float((actuator/'published-constraints').read_text().splitlines()[-1].split()[2])
        r=rows([actuator/'balance-0001.csv'],('step',))[1,]
        defect=float(r['Wext'])-sum(float(r[k]) for k in ('energy_change','Db','Dc','Df','phase_remainder','phase_work'))
        if wall_max!=0. or float(r['Wext'])<=0 or abs(defect)>1.e-11: raise ValueError(f'Interior work {r}, wall speed {wall_max}')
        print('Wall speed',wall_max,'m/s, actuator work',r['Wext'],'balance defect',defect,'J/m')
    if args.group=='mpi':
        # The 8 x 4 two-owner fixture puts the lower-left cell on one rank.
        run('owner-gravity',gravity('0;if(x<0.2 && y<0.1,1000,0)'),ranks=2,expected='nonzero on 1 of 2 ranks')
    if args.group=='lifecycle':
        snapshots=subsection('Checkpointing','set Steps between checkpoint = 1\nset Number of checkpoints to keep = 4')
        clean=run('clean',snapshots,end='.015')
        for generation,steps in [(1,[1,2,3]),(3,[3])]:
            other=run(f'restart-{generation}',snapshot=clean/'restart'/f'{generation:02}',end='.015')
            continuation(clean,other,steps)
            for item in ['Native growing fault v2','Growing fault accepted balance v1']:
                if (clean/('published-'+item)).read_bytes()!=(other/('pre-solve-'+item)).read_bytes():
                    raise ValueError(f'Generation {generation}: restored 10 ms {item} differs')
        if args.legacy_snapshot:
            legacy=run('legacy',snapshot=args.legacy_snapshot,end='.015')
            if (not (legacy/'pre-solve-Native growing fault v2').is_file() or
                (legacy/'pre-solve-Growing fault accepted balance v1').exists() or not (legacy/'balance-0003.csv').is_file()):
                raise ValueError('Legacy restore failed to distinguish missing work from physical state')
            print('Legacy generation 03: native history restored, prior balance unavailable, new 15 ms balance published.')
        completions=[list(map(int,line.split())) for line in (clean/'completions').read_text().splitlines()]
        for step in (1,2,3):
            r=rows([clean/f'balance-{step:04}.csv'],('step',))[step,]
            if [line for line in completions if line[0]==step]!=[[step,step-1,int(r['iterations']),1]]:
                raise ValueError('Incorrect completion or premature publication')
        if (clean.parent/'log').read_text().count('RMS, max velocity:')!=4:
            raise ValueError('Missing accepted velocity statistics')
        print('Completions (clock, accepted parent, iterations, success)',completions)
        for label,setting in [('rsf','set Radiation damping = 0.75'),('maxwell','set Viscosity = 1800')]:
            common=subsection('Material model',subsection('Growing fault',setting))+'\nset Nonlinear solver failure strategy = cut timestep size\n'
            stop=subsection('Termination criteria','set Termination criteria = end step\nset End step = 1')
            half=run(label+'-half',common+stop+'\nset Maximum first time step = .0025\n')
            retry=run(label+'-cut',common+stop)
            if (retry.parent/'log').read_text().count('Trying to cut the current time step')!=1:
                raise ValueError(f'{label}: expected one nonlinear failure')
            continuation(half,retry,[1])
            for f in half.glob('*0001*.csv'):
                if f.read_bytes()!=(retry/f.name).read_bytes():
                    raise ValueError(f'{label}: clean and cut differ in {f.name}')
            for folder in (half,retry):
                if sorted(f.name for f in folder.glob('balance-*.csv'))!=['balance-0001.csv']:
                    raise ValueError('Extra physical publication after cutback')
                book=rows([folder/'balance-0001.csv'],('step',))[1,]
                if float(book['time'])!=.0025: raise ValueError('Unexpected cutback clock')
            print(label,'automatic cut recovered the clean 2.5 ms endpoint')
    if args.group=='geometry':
        clean=run('origin',end='.015')
        shifted=(base.replace('set X extent = 2','set Box origin X coordinate = 10000\n  set X extent = 2')
                 .replace('1.4330127018922194,0.75','10001.433012701892,0.75').replace('x-(1)','x-(10001)'))
        translated=run('translated',source=shifted,end='.015')
        p,q=(rows(folder.glob('points-0000-r*.csv'),('cell','q')) for folder in (clean,translated))
        if p.keys()!=q.keys() or len(p)!=4968 or any(p[k]['section']!=q[k]['section'] for k in p):
            raise ValueError('Translated geometry changed quadrature identities or strip membership')
        position=max(max(abs(float(q[k]['x'])-10000-float(p[k]['x'])),abs(float(q[k]['y'])-float(p[k]['y']))) for k in p)
        if position>4.e-12: raise ValueError(f'Translated physical quadrature positions differ by {position} m')
        for step in range(4):
            a,b=(rows([folder/f'sections-{step:04}.csv'],('section',)) for folder in (clean,translated))
            if any(a[k]['xi']!=b[k]['xi'] or a[k]['formed']!=b[k]['formed'] or
                   (step==0 and a[k]['ds']!=b[k]['ds']) for k in a):
                raise ValueError(f'Translated stations, widths or formation differ at clock {step}')
        print('Translated native points',len(p),'maximum physical position defect',position,'m')
    if args.group=='companion':
        clean=run('standard',end='.01')
        reordered=(base.replace('tau_xx, tau_yy, tau_xy, phase_field','phase_field, tau_xx, tau_yy, tracer')
                   .replace('generic, generic, generic, phase field','phase field, generic, generic, generic')
                   .replace('static, static, static, midpoint fault phase','midpoint fault phase, static, static, field'))
        seed=next(l for l in base.splitlines() if 'Function expression = 0;0;0;' in l)
        reordered=reordered.replace(seed,seed.replace('0;0;0;','')+';0;0;x')
        mixed=run('phase-first',source=reordered,end='.01')
        def values(path,step,component):
            fields=rows(path.glob(f'fields-{step:04}-r*.csv'),('cell','component','local'))
            return {(cell,j):float(row['value']) for (cell,c,j),row in fields.items() if c==component}
        if any(values(clean,n,7)!=values(mixed,n,4) for n in (0,1,2)): raise ValueError('Phase changed under field permutation')
        a,b=values(mixed,1,7),values(mixed,2,7)
        change=max(abs(a[k]-b[k]) for k in a)
        if change<1.e-8: raise ValueError('Implicit companion failed to advance')
        print('Maximum companion coefficient change',change)


if __name__=='__main__':
    main()
