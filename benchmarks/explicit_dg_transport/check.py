"""Exercise the normal scalar DG composition dispatch; create a new run directory per invocation."""
import argparse
import math
import os
from pathlib import Path
import re
import subprocess
import shutil

import numpy as np
from reference import Reference, basis, mean_weights, nodes_x, nodes_y, tensor, to_coeff, to_values

BASE=Path(__file__).with_name('mixed.prm').read_text() + """
subsection Postprocess
  set List of postprocessors = DG coefficient samples
end
"""


def require(ok,message):
    if not ok:
        raise ValueError(message)


def run(args,name,settings='',ranks=1,fail=None,retry=False,checkpoint=None,template=BASE,dim=2):
    dest=args.directory/name
    dest.mkdir(parents=True)
    output=dest/'output'
    if checkpoint is not None:
        # Copy the entire parent including the accepted output prefix and
        # checkpoint slot. Each resume is independent of the other variants.
        shutil.copytree(checkpoint,output)
    so=args.plugins/f'libexplicit_dg_compositions.{args.mode}.so'
    libraries=[so]
    if retry:
        libraries.append(args.plugins/f'libfixed_fault.{args.mode}.so')
    prm=dest/'input.prm'
    prm.write_text(template+'\n'+settings+'\nset Output directory = '+str(output)+'\nset Additional shared libraries = '+', '.join(map(str,libraries))+'\n')
    cmd=([args.mpiexec,'-n',str(ranks)] if ranks>1 else [])+[args.aspect,prm]
    with (dest/'run.log').open('w') as log:
        status=subprocess.run(cmd,cwd=dest,stdout=log,stderr=subprocess.STDOUT,
                              env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',DEAL_II_NUM_THREADS='1')).returncode
    msg=' '.join((dest/'run.log').read_text().split())
    if fail:
        require(status!=0 and fail in msg,f'{name} missed diagnostic: {fail}; {dest}/run.log')
        return
    require(status==0,f'{name} failed: {dest}/run.log')
    if retry:
        require(msg.count('Repeating the current time step')==1,'expected one rejected trial')
    return read_output(output,ranks,dim)


def read_output(output,ranks,dim=2):
    data={}
    shards=set()
    for path in output.glob('coefficients-*-*'):
        _,step,rank=path.name.split('-')
        shards.add((int(step),int(rank)))
        lines=path.read_text().splitlines()
        clock=tuple(map(float,lines[0].split()))
        frame=data.setdefault(int(step),(clock,{}))
        require(frame[0]==clock,'partition clocks differ')
        for line in lines[1:]:
            cell,field,fe,*numbers=line.split()
            box=tuple(map(float,numbers[:2*dim]))
            vals=np.array(numbers[2*dim:],float)
            key=(cell,field)
            require(key not in frame[1] and len(vals)==3**dim and np.isfinite(vals).all(),'bad field record')
            frame[1][key]=(box,fe,vals,int(rank))
    require(data and sorted(data)==list(range(len(data))),'missing publication')
    require(shards=={(k,r) for k in data for r in range(ranks)},'unexpected rank shards')
    clocks=[data[k][0] for k in sorted(data)]
    for rank in range(ranks):
        published=(output/f'publication-{rank}').read_text().splitlines()
        require([tuple(map(float,line.split())) for line in published]==clocks,'duplicate or missing publication')
    return data


def field_data(data,step,field):
    """Return physical boxes, native basis coefficients and owners in geometric order."""
    records=sorted((row for (cell,f),row in data[step][1].items() if f==field),key=lambda r:r[0])
    return [r[0] for r in records],np.array([r[2] for r in records]),[r[3] for r in records]


def compare(a,b,tolerance):
    """Compare accepted coefficients by clock, CellId and field, across owner layouts."""
    require(a.keys()==b.keys(),'different publication counts')
    worst=0.
    for k,(clock,rows) in a.items():
        require(clock==b[k][0] and rows.keys()==b[k][1].keys(),'different clock or identities')
        for key,(box,fe,x,r) in rows.items():
            box2,fe2,y,_=b[k][1][key]
            require(box==box2 and fe==fe2,'field or geometry changed')
            err=np.max(np.abs(x-y))
            require(err<=tolerance,f'{key} differs by {err}')
            worst=max(worst,err)
    return worst


def section(name,text):
    return f'subsection {name}\n{text}\nend\n'


def setting(name,value):
    return f'set {name} = {value}\n'


def fields(**kwargs):
    return section('Compositional fields',''.join(setting(k.replace('_',' '),v) for k,v in kwargs.items()))


def fn(parent,expr,sub='Function'):
    return section(parent,section(sub,setting('Function expression',expr)))


def constant_velocity(u,v):
    return lambda x,y:(np.full_like(x,u),np.full_like(y,v))


def oracle(data,field,initial,velocity,divergence,exterior,bounded,*,expected_ranks):
    boxes, _, owners = field_data(data, 0, field)
    require(set(owners) == set(range(expected_ranks)), f'{field}: missing expected cell owner')
    reference = Reference(boxes, velocity, divergence, exterior, bounded)
    expected = np.array([initial(x0 + (x1-x0)*nodes_x, y0 + (y1-y0)*nodes_y, 0.)
                         for x0,y0,x1,y1 in boxes])
    for (_, name), (_, fe, _, _) in data[0][1].items():
        if name == field:
            require(('Bernstein' in fe) == bounded, 'wrong native basis')

    original_mass = np.dot(reference.areas, expected @ mean_weights)
    reference.filter(expected)
    # Count incoming neighbor contributions with nonzero action on the initial
    # field. An empty output from a second rank cannot qualify a partition face.
    partition_faces = sum(owners[i] != owners[j] and np.any(matrix @ expected[j])
                          for i,j,matrix in reference.upstream)
    if expected_ranks > 1:
        require(partition_faces > 0, 'missing active upwind partition face')
    entry_scales = reference.limited
    worst = 0.
    for step, (clock, _) in sorted(data.items()):
        time, dt = clock
        if step:
            expected, _ = reference.advance(expected, time-dt, dt)
        else:
            require(clock == (0.,0.), 'initialization has a positive physical interval')
        current_boxes, coefficients, current_owners = field_data(data, step, field)
        require(current_boxes == boxes and current_owners == owners, 'changed mesh or ownership')
        # The strong-form reference advances values at the nine Lobatto nodes.
        # Convert Bernstein coefficients before comparing pointwise or integrating.
        observed = coefficients @ to_values.T if bounded else coefficients
        error = np.max(abs(observed - expected))
        require(error < 3.e-11, f'{field} step {step}: production/strong-form defect {error}')
        worst = max(worst, error)
        mass_delta = np.dot(reference.areas, observed @ mean_weights) - original_mass
        require(abs(mass_delta - reference.balance) < 1.e-12, 'stage-weighted material balance failed')
    print(f'{field}: {len(boxes)} cells, {len(data)-1} intervals, max nodal defect {worst:.3g}, '
          f'entry/stage scales {entry_scales}/{reference.limited-entry_scales}, partition faces {partition_faces}, '
          f'source {reference.integrated_source:.9g}, outward flux {reference.integrated_flux:.9g}, '
          f'balance defect {mass_delta-reference.balance:.3g}',flush=True)
    return reference.limited-entry_scales


def composition_block_counts(log, frames):
    """Read the two nodal composition blocks from each accepted timestep's matrix statistics."""
    counts = {}
    for step_log in log.split('*** Timestep ')[1:]:
        header, body = step_log.split('\n', 1)
        match = re.fullmatch(r'(\d+):\s+t=(\S+) seconds, dt=(\S+) seconds', header)
        require(match is not None, f'unrecognized timestep header {header}')
        step, time, dt = match.groups()
        clock = (float(time), float(dt))
        require(int(step) in frames and frames[int(step)][0] == clock and clock not in counts,
                f'unexpected matrix clock {header}')
        matrices = body.split('system matrix nnz by block:')
        require(len(matrices) == 2, f'expected one system matrix at {clock}')
        rows = [line.split() for line in matrices[1].strip().splitlines()[:5]]
        counts[clock] = tuple(int(rows[i][i].replace(',', '')) for i in (3, 4))
    require(len(counts) == len(frames), 'missing matrix statistics')
    return counts


def check_method_setup(log, name, expected, frames, expected_blocks):
    setup = re.findall(r'^DG setup (\w+) (\d) (\d+) (\d+) (\d+)$', log, re.M)
    completions = re.findall(r'DG completion (\d+) (\w+)$', log, re.M)
    require(completions == [(str(step), field) for step in sorted(frames) for field in expected],
            f'{name}: missing or repeated field completion')
    require(len(setup) == len(expected), f'{name}: missing or repeated setup record')
    for (field, explicit), row in zip(expected.items(), setup):
        count = int(not explicit)
        require(row == (field, str(int(explicit)), str(count), str(count), str(count)),
                f'{name}: wrong assembler setup {row}')
        require(log.count(f'Solving {field} system') == count*len(frames), f'{name}: wrong implicit solve count')
    counts = composition_block_counts(log, frames)
    require(counts.keys() == expected_blocks.keys(), f'{name}: different matrix clocks')
    for clock, diagonal in counts.items():
        require(diagonal == expected_blocks[clock],
                f'{name} at {clock}: composition blocks {diagonal}, expected {expected_blocks[clock]}')
    print(f'{name}: {setup}, composition diagonal nnz by clock {counts}, '
          f'{len(completions)} field completions', flush=True)


def mixed(args):
    profile=lambda x,y,t:(x-t-.4)**2-.2
    u=constant_velocity(1,0)
    zero=lambda x,y:np.zeros_like(x)
    a=run(args,'mixed')
    swap=run(args,'swapped',fields(Names_of_fields='tracer, companion'))
    statistics = section('Postprocess', setting('List of postprocessors',
                         'DG coefficient samples, DG method setup, matrix statistics'))
    b = run(args, 'default', fields(SSPRK3_fields='') + statistics)
    implicit_blocks = composition_block_counts((args.directory/'default/run.log').read_text(), b)
    require(all(first > 0 and second == 0 for first, second in implicit_blocks.values()),
            f'all-implicit control lacks shared sparsity: {implicit_blocks}')
    print('all-implicit composition diagonal nnz by clock', implicit_blocks, flush=True)
    for name, names, methods, legacy in [
            ('method-last', 'companion, tracer', 'field, explicit DG SSPRK3', a),
            ('method-first', 'tracer, companion', 'explicit DG SSPRK3, field', swap)]:
        case = run(args, name, fields(Names_of_fields=names, Compositional_field_methods=methods,
                                     SSPRK3_fields='') + statistics)
        print(name, 'old/new coefficient defect', compare(case, legacy, 0.), flush=True)
        log = (args.directory/name/'run.log').read_text()
        check_method_setup(log, name, {field: field=='tracer' for field in names.split(', ')}, case, implicit_blocks)
    both = run(args, 'both-by-name', fields(SSPRK3_fields='companion, tracer'))
    broadcast = run(args, 'broadcast', fields(Compositional_field_methods='explicit DG SSPRK3') + statistics)
    # The old input still names tracer, redundantly, while broadcast also selects companion.
    print('broadcast/redundant coefficient defect', compare(broadcast, both, 0.), flush=True)
    check_method_setup((args.directory/'broadcast/run.log').read_text(), 'broadcast',
                       {'companion': True, 'tracer': True}, broadcast, {clock: (0, 0) for clock in implicit_blocks})
    for invalid in ('static', 'midpoint fault phase'):
        run(args, 'conflict-'+invalid.replace(' ', '-'), fields(Compositional_field_methods='field, '+invalid),
            fail='conflicts with compositional field method '+invalid+' for tracer')
    run(args, 'duplicate', fields(SSPRK3_fields='tracer, tracer'), fail='Duplicate names in SSPRK3 fields')
    run(args, 'unknown', fields(SSPRK3_fields='absent'), fail='Unknown SSPRK3 field absent')
    oracle(a,'tracer',profile,u,zero,profile,False,expected_ranks=1)
    oracle(swap,'tracer',profile,u,zero,profile,False,expected_ranks=1)
    separation=[]
    for step in a:
        base=field_data(b,step,'companion')[1]
        for case in (a,swap):
            comp=field_data(case,step,'companion')[1]
            np.testing.assert_allclose(comp,base,rtol=0,atol=1e-12)
        if step:
            separation.append(float(np.max(abs(base-field_data(a,step,'tracer')[1]))))
    require(min(separation)>5.e-5,'methods failed to separate')
    print('implicit/explicit endpoint separations',separation,flush=True)
    parallel=run(args,'mixed-mpi',ranks=2)
    oracle(parallel,'tracer',profile,u,zero,profile,False,expected_ranks=2)
    print('mixed serial/MPI max defect',compare(a,parallel,1.e-12),flush=True)
    # Two complete sets of clocks and valid coefficients, but one empty owner.
    false_partition=args.directory/'empty-rank'
    false_partition.mkdir()
    for path in (args.directory/'mixed/output').glob('coefficients-*-0'):
        shutil.copy(path,false_partition/path.name)
        (false_partition/(path.name[:-1]+'1')).write_text(path.read_text().splitlines()[0]+'\n')
    for rank in (0,1):
        shutil.copy(args.directory/'mixed/output/publication-0',false_partition/f'publication-{rank}')
    try:
        oracle(read_output(false_partition,2),'tracer',profile,u,zero,profile,False,expected_ranks=2)
    except ValueError as error:
        require('missing expected cell owner' in str(error),str(error))
        print('false two-rank fixture rejected:',error,flush=True)
    else:
        raise ValueError('qualified an empty second rank')
    static_boundary=section('Boundary composition model',section('Function',
                    setting('Variable names','x,y')+setting('Function expression','.3;.3')))
    spatial=run(args,'static-inflow',static_boundary)
    oracle(spatial,'tracer',profile,u,zero,lambda x,y,t:np.full_like(x,.3),False,expected_ranks=1)
    reaction=section('Material model',setting('Model name','tracer reaction'))
    run(args,'explicit-reaction',reaction+fields(SSPRK3_fields='', Compositional_field_methods='field, explicit DG SSPRK3'),
        fail='cannot omit a reaction increment for tracer')
    require((args.directory/'explicit-reaction/output/publication-0').read_text().split()==['0','0'],
            'reacting trial reached publication')
    unrelated=run(args,'implicit-reaction',reaction+fields(SSPRK3_fields='companion'))
    oracle(unrelated,'companion',profile,u,zero,profile,False,expected_ranks=1)
    require(np.max(abs(field_data(unrelated,1,'tracer')[1]-field_data(b,1,'tracer')[1]))>.009,
            'implicit reaction was discarded')


def phase_settings(expr,exterior,flow,level=0,end=.025,dt=.0125):
    return (fields(Number_of_fields='1',Names_of_fields='damage',Types_of_fields='phase field',SSPRK3_fields='damage')
            +fn('Initial composition model',expr)+fn('Boundary composition model',exterior)
            +fn('Prescribed Stokes solution',flow,'Velocity function')+setting('End time',end)
            +section('Mesh refinement',setting('Initial global refinement',level))
            +fn('Time stepping',str(dt)))


def ridge_expr(sign):
    return f'.15+.58*exp(-((x-({sign})*t-.5-.45*(y-({sign})*.4*t-.5))/.12)^2)'


def ridge(x,y,t,s):
    return .15+.58*np.exp(-((x-s*t-.5-.45*(y-s*.4*t-.5))/.12)**2)


def ridge_metrics(data,s):
    step=max(data)
    time=data[step][0][0]
    boxes,raw,_=field_data(data,step,'damage')
    nodal=raw@to_values.T
    def sample(x,y):
        z=np.full_like(x,np.nan)
        for (a,b,c,d),v in zip(boxes,nodal):
            use=(a<=x)&(x<=c)&(b<=y)&(y<=d)
            if np.any(use):
                z[use]=tensor(basis((x[use]-a)/(c-a)),basis((y[use]-b)/(d-b)))@v
        require(np.isfinite(z).all(),'transect left the domain')
        return z
    # Mid-height horizontal transect resolves a displacement of (1-.45*.4)*u*t.
    x=np.linspace(.1,.9,4001)
    values=sample(x,np.full_like(x,.5))
    crest=x[np.argmax(values)]
    disp=crest-.5
    expect_disp=.82*s*time
    center=np.array([.5+s*time,.5+.4*s*time])
    normal=np.array([1.,-.45])/math.sqrt(1+.45**2)
    ends=sorted((side-center[k])/normal[k] for k in (0,1) for side in (0.,1.))
    ray=np.linspace(ends[1]+1.e-12,ends[2]-1.e-12,6001)
    x,y=center[:,None]+normal[:,None]*ray
    phi=sample(x,y)
    analytic=ridge(x,y,time,s)
    weight=lambda v:v*(1+v)/(1-v)**2
    # Normalize by an independent analytic translated full-ray integral.
    integrate=lambda f:np.dot(np.diff(ray),(f[:-1]+f[1:])/2)
    integral=integrate(weight(phi))/integrate(weight(analytic))
    half=ray[phi>=.15+.58/2]
    width=half[-1]-half[0]
    width_ref=2*.12*math.sqrt(math.log(2)/(1+.45**2))
    error=max(abs(phi-analytic))
    print(f'ridge {s:+}, {len(boxes)} cells: displacement {disp:.6g}/{expect_disp:.6g}, peak {max(phi):.6g}, '
          f'normal FWHM {width:.6g}/{width_ref:.6g}, analytic-normalized integral {integral:.6g}, '
          f'transect max defect {error:.6g}, certified hull [{raw.min():.6g}, {raw.max():.6g}]',flush=True)
    return abs(disp-expect_disp),error,abs(width-width_ref),abs(integral-1)


def phases(args):
    zero=lambda x,y:np.zeros_like(x)
    for sign in (1,-1):
        defects=[]
        for level in (2,3):
            # 2x1 coarse box: 8x4 and 16x8 after refinement.
            settings=phase_settings(ridge_expr(sign),ridge_expr(sign),f'{sign};{.4*sign}',level,.15,.025)
            a=run(args,f'ridge-{sign}-{level}',settings)
            f=lambda x,y,t:ridge(x,y,t,sign)
            oracle(a,'damage',f,constant_velocity(sign,.4*sign),zero,f,True,expected_ranks=1)
            defects.append(ridge_metrics(a,sign))
            if level==2:
                b=run(args,f'ridge-{sign}-mpi',settings,ranks=2)
                oracle(b,'damage',f,constant_velocity(sign,.4*sign),zero,f,True,expected_ranks=2)
                print('partition defect',compare(a,b,3e-12),flush=True)
        require(defects[1][0]<.006 and defects[1][1]<.012 and defects[1][1]<defects[0][1] and
                defects[1][2]<.006 and defects[1][3]<.025,f'unresolved translated ridge: {defects}')

    # A true quadratic has admissible physical range and a coefficient > 1.
    profile=lambda x,y,t:.9-.65*(2*y-1)**2
    settings=phase_settings('.9-.65*(2*y-1)^2','.9-.65*(2*y-1)^2','1;0')
    a=run(args,'enclosure',settings)
    require(field_data(a,0,'damage')[1].max()>1.5,'expected out-of-range coefficient')
    oracle(a,'damage',profile,constant_velocity(1,0),zero,profile,True,expected_ranks=1)

    # At entry the in-cell maximum is 1.1 and mean is 14/15. An incoming
    # jump at later stages exercises scaling independently of initialization.
    profile=lambda x,y,t:.6+2*y*(1-y)
    flow=lambda x,y:(1+.35*x,np.zeros_like(y))
    inflow=lambda x,y,t:np.full_like(x,.15+.1*t)
    settings=phase_settings('.6+2*y*(1-y)','.15+.1*t','1+.35*x;0')
    a=run(args,'divergent-limited',settings,ranks=2)
    changed=oracle(a,'damage',profile,flow,lambda x,y:np.full_like(x,.35),inflow,True,expected_ranks=2)
    require(changed>0,'stage limiter never activated')
    boxes,stored,_=field_data(a,0,'damage')
    wrong=Reference(boxes,flow,lambda x,y:np.full_like(x,.35),inflow,True)
    clock=a[1][0]
    endpoint_only,_=wrong.advance(stored@to_values.T,0.,clock[1],limit_internal=False)
    defect=np.max(abs(endpoint_only-field_data(a,1,'damage')[1]@to_values.T))
    require(defect>1.e-6,'endpoint-only limiting passed the stage check')
    print('endpoint-only limiter first-step defect',defect,flush=True)
    initial=field_data(a,0,'damage')[1]@to_values.T
    np.testing.assert_allclose(initial@mean_weights,14/15,rtol=0,atol=1e-14)
    require(abs(initial.max()-1)<1e-13,'real overshoot survived')
    bad=phase_settings('if(x<.49,2,.5)','.5','1;0')
    run(args,'bad-mean',bad,ranks=2,fail='Inadmissible Bernstein phase mean at transport entry on 1 of 2 ranks')
    f=ridge_expr(1)
    settings=phase_settings(f,f,'1;.4',2)
    # Adjacent nodal DGQ2 / Bernstein DGQ2 / nodal DGQ2 in the same FESystem.
    tracer='(x-.4)^2-.2'
    boundary='(x-t-.4)^2-.2'
    settings+=fields(Number_of_fields=3,Names_of_fields='companion, damage, tracer',
                     Types_of_fields='generic, phase field, generic',SSPRK3_fields='tracer, damage')
    settings+=fn('Initial composition model',';'.join([tracer,f,tracer]))
    settings+=fn('Boundary composition model',';'.join([boundary,f,boundary]))
    clean=run(args,'clean',settings)
    oracle(clean,'tracer',lambda x,y,t:(x-t-.4)**2-.2,constant_velocity(1,.4),zero,
           lambda x,y,t:(x-t-.4)**2-.2,False,expected_ranks=1)
    oracle(clean,'damage',lambda x,y,t:ridge(x,y,t,1),constant_velocity(1,.4),zero,
           lambda x,y,t:ridge(x,y,t,1),True,expected_ranks=1)
    retry=run(args,'retry',settings+section('Time stepping',setting('List of model names','function, reject long first fault step')+
               section('Function',setting('Function expression','if(time==0,.025,.0125)'))),retry=True)
    print('retry max defect',compare(clean,retry,0.),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('aspect','plugins','directory'):
        parser.add_argument(name,type=lambda s:Path(s).resolve())
    parser.add_argument('--mpiexec',default='mpiexec')
    parser.add_argument('--only',choices=('mixed','phase'))
    args=parser.parse_args()
    args.mode=args.aspect.name.split('-')[-1]
    if args.only!='phase':
        mixed(args)
    if args.only!='mixed':
        phases(args)
