"""Stage A (four ordinary steps), then B (eight safe half-steps). Never launch.

Use the same original checkpoint/input for both invocations. B requires A's
completed normal_summary.csv, not a guessed timestep schedule.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import struct
import zlib

from stage_normal_stress_diagnostic import stage, sha, parameters


def half_clock(rows, start, checkpoint_step):
    if len(rows)!=4:
        raise ValueError('A must complete exactly four accepted steps')
    result=[];adjustments=[];previous=start
    for i,row in enumerate(rows):
        if int(row['step'])!=checkpoint_step+i+1:
            raise ValueError('A accepted-step sequence is not contiguous')
        end,dt=float(row['time_s']),float(row['dt'])
        if previous+dt!=end or not dt>0:
            raise ValueError('A clock does not reproduce the actual accepted time')
        first=dt/2;middle=previous+first;second=end-middle
        if not previous<middle<end or middle+second!=end:
            raise ValueError('Half-step endpoints cannot be represented')
        correction=second-first
        if abs(correction)>math.ulp(end):
            raise ValueError('Unexpected clock adjustment exceeding one absolute-time ULP')
        result.extend([(checkpoint_step+2*i+1,middle,first),(checkpoint_step+2*i+2,end,second)])
        adjustments.append(dict(A_step=int(row['step']),A_dt=dt,first_dt=first,second_dt=second,
                                second_minus_exact_half=correction,relative_adjustment=correction/first,
                                endpoint=end,absolute_time_ULP=math.ulp(end)))
        previous=end
    return result,adjustments


def retime_pending(data, expected_time, expected_dt, expected_step, new_time, new_dt):
    # Reuse the narrowly qualified binary-clock operation from
    # bp3/run_coupled_substeps.py. Fail closed on any archive-layout change.
    header=struct.unpack('<4I',data[:16]);raw=zlib.decompress(data[16:])
    if header!=(1,len(raw),len(raw),len(data)-16):raise ValueError('Unknown archive framing')
    offset=237
    old_clock=struct.unpack_from('<dddI',raw,offset)
    if old_clock[0]!=expected_time or old_clock[1]!=expected_dt or old_clock[3]!=expected_step:
        raise ValueError(f'Pending checkpoint clock does not match A: {old_clock}')
    if raw.count(struct.pack('<dddI',*old_clock))!=1:raise ValueError('Ambiguous clock layout')
    if not math.isfinite(old_clock[2]) or old_clock[2]<=0:raise ValueError('Invalid retained previous dt')
    changed=raw[:offset]+struct.pack('<dd',new_time,new_dt)+raw[offset+16:]
    assert changed[:offset]==raw[:offset] and changed[offset+16:]==raw[offset+16:]
    compressed=zlib.compress(changed,9)
    return struct.pack('<4I',1,len(changed),len(changed),len(compressed))+compressed,old_clock


def prepare(args):
    args.steps=4 if args.branch=='A' else 8
    args.control=False;args.local_verification=False
    config=parameters(args.input.read_text())
    if config.get(('Use years instead of seconds',),'true')!='false':
        raise ValueError('The matched clock requires the existing seconds convention')
    rows=[];schedule=[];adjustments=[]
    if args.branch=='B':
        if args.a_output is None:raise ValueError('B requires --a-output')
        with (args.a_output/'normal_summary.csv').open() as stream:
            rows=list(csv.DictReader(stream))
        schedule,adjustments=half_clock(rows,5310111071.5634108,5612)
        a_manifest=json.loads((args.a_output.parent/'staging.json').read_text())
        if a_manifest['original_input_sha256']!=sha(args.input):raise ValueError('A/B inputs differ')
        for name,digest in a_manifest['checkpoint_sha256'].items():
            if sha(args.checkpoint/name)!=digest:raise ValueError('A/B source checkpoint differs')
        if 'function' in [s.strip() for s in config['Time stepping','List of model names'].split(',')]:
            raise ValueError('Do not replace a production function controller')
    stage(args)
    dest=args.destination.resolve()
    if args.branch=='B':
        b_manifest=json.loads((dest/'staging.json').read_text())
        if a_manifest['libraries']!=b_manifest['libraries']:
            raise ValueError('A and B must load byte-identical plugins')
        if a_manifest['staged_input_sha256']!=b_manifest['staged_input_sha256']:
            raise ValueError('A and B scientific fixture files differ')
    addition='''
subsection Postprocess
  subsection BP5 normal diagnostic
    set Small windows only = true
    set Raw every step = true
    set Native centerline = true
  end
end
'''
    manifest=dict(branch=args.branch,solves=args.steps,source_checkpoint=str(args.checkpoint.resolve()),
                  physics_unchanged=True,clock_adjustments=adjustments)
    if args.branch=='B':
        checkpoint=dest/'output-normal-diagnostic/restart/01/resume.z'
        original_hash=sha(checkpoint)
        changed,old_clock=retime_pending(checkpoint.read_bytes(),float(rows[0]['time_s']),float(rows[0]['dt']),5613,
                                        schedule[0][1],schedule[0][2])
        checkpoint.write_bytes(changed)
        manifest.update(resume_before_sha256=original_hash,resume_after_sha256=sha(checkpoint),
                        old_clock=old_clock,only_pending_time_dt_modified=True,
                        A_directory=str(args.a_output.parent.resolve()))
        (dest/'expected_clock.txt').write_text(''.join(f'{s} {t:.17g} {dt:.17g}\n' for s,t,dt in schedule))
        manifest['expected_clock_sha256']=sha(dest/'expected_clock.txt')
        # The Function plugin is only an additional MIN cap. Existing convection,
        # fault, state, global/growth and termination controls remain active.
        expression='1e100'
        for _,target,dt in reversed(schedule):
            expression=f'if(time<{target:.17g},{dt:.17g},{expression})'
        addition+=f'''
subsection Time stepping
  set List of model names = {config['Time stepping','List of model names']}, function
  subsection Function
    set Variable names = time
    set Function expression = {expression}
  end
end
subsection Postprocess
  subsection BP5 normal diagnostic
    set Expected clock file = expected_clock.txt
  end
end
'''
        manifest['schedule']=[dict(step=s,time_s=t,dt=dt) for s,t,dt in schedule]
    path=dest/'normal_stress_diagnostic_restart.prm'
    path.write_text(path.read_text()+addition)
    manifest['run_prm_sha256']=sha(path)
    (dest/'experiment.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Prepared branch {args.branch}; no solves launched.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('branch',choices=['A','B'])
    for name in ('checkpoint','input','job','destination','bp5-library','diagnostic-library'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--a-output',type=Path)
    prepare(p.parse_args())
