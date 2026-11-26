"""Exercise the 5A repair cases through ordinary ASPECT parameter files."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

from check_section_publication import check as check_cold
from compare import continuation, rows
from parameter_file import composition_names, function, resolved_function, subsection


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for arg in ('aspect','plugin','destination'):
        ap.add_argument(arg,type=lambda p:Path(p).resolve())
    ap.add_argument('group',choices=['friction','constraints','mpi','lifecycle','companion','geometry'])
    ap.add_argument('--repaired-checkpoints',type=Path,help='checkpoint directory written by repaired 5A before ownership cleanup')
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
    gravity=lambda expr:subsection('Gravity model','set Model name = function\n'+function(expr))
    if args.group=='friction':
        cold=run('cold',subsection('Material model',subsection('Growing fault','set Reference friction coefficient = .9')))
        check_cold(cold)
        names, phase_name = composition_names(cold)
        phase_index = names.index(phase_name)
        def phase(expr):
            components = resolved_function(cold,'Initial composition model')['Function expression']['value'].split(';')
            components[phase_index] = expr
            return subsection('Initial composition model',function(';'.join(components)))
        one=run('fully-damaged',phase('1'))
        fields=[rows(one.glob(f'fields-{step:04}-r*.csv'),('cell','component','local')) for step in (0,1)]
        if any(fields[0][k]['value']!=fields[1][k]['value'] for k in fields[0] if k[1]==4+phase_index):
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
        parents = [('restart',clean/'restart')]
        if args.repaired_checkpoints:
            parents.append(('repaired',args.repaired_checkpoints))
        for label,checkpoints in parents:
            for generation,steps in [(1,[1,2,3]),(3,[3])]:
                other=run(f'{label}-{generation}',snapshot=checkpoints/f'{generation:02}',end='.015')
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
        from check_seed_frame import check_geometry
        check_geometry(run, args.plugin.parent/'timestep/libmidpoint_fields.release.so')
    if args.group=='companion':
        clean=run('standard',end='.01')
        names, phase_name = composition_names(clean)
        expressions = resolved_function(clean, 'Initial composition model')['Function expression']['value'].split(';')
        initial = dict(zip(names, expressions, strict=True))
        order = [phase_name, 'tau_xx', 'tau_yy', 'tracer']
        extra = subsection('Compositional fields',
                           'set Names of fields = ' + ', '.join(order) +
                           '\nset Types of fields = phase field, generic, generic, generic' +
                           '\nset Compositional field methods = midpoint fault phase, static, static, field')
        initial['tracer'] = 'x'
        extra += subsection('Initial composition model', function(';'.join(initial[name] for name in order)))
        mixed=run('phase-first',extra,end='.01')
        def values(path,step,name):
            # Two velocity components, pressure and temperature precede compositions.
            component = 4 + composition_names(path)[0].index(name)
            fields=rows(path.glob(f'fields-{step:04}-r*.csv'),('cell','component','local'))
            result = {(cell,j):float(row['value']) for (cell,c,j),row in fields.items() if c==component}
            if not result:
                raise ValueError(f'{path}: missing {name} coefficients')
            return result
        if any(values(clean,n,phase_name)!=values(mixed,n,phase_name) for n in (0,1,2)):
            raise ValueError(f'{phase_name} changed under field permutation')
        a,b=values(mixed,1,'tracer'),values(mixed,2,'tracer')
        change=max(abs(a[k]-b[k]) for k in a)
        if change<1.e-8: raise ValueError('Implicit companion failed to advance')
        print('Maximum companion coefficient change',change)


if __name__=='__main__':
    main()
