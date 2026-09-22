"""Prepare only the requested matched-time fine/restart/half-step branches."""
import argparse
import csv
import json
import shutil
from pathlib import Path

from run_short import OUT, BIN, LIB, BP3, parameters, render, digest, prepare
from run_coupled_substeps import retime
from slip_history import restore_prefix


def main(kind):
    source=OUT/'candidate-six'
    accepted=list(csv.DictReader((source/'accepted_steps.csv').open()))
    assert len(accepted)>=3
    values=parameters((source/'run.prm').read_text())
    if kind=='reference':
        times=[float(r['time']) for r in accepted[1:3]]
        label='reference-two'
        prepare(label,'reference',False,times[-1])
        out=OUT/label
        values=parameters((out/'run.prm').read_text())
        record=json.loads((out/'launch.json').read_text())
    else:
        assert len(accepted)>=5
        label='restart-two' if kind=='restart' else 'half-step'
        out=OUT/label
        out.mkdir()
        paths=[]
        for path in (source/'restart').glob('[0-9][0-9]'):
            if int((path/'bp3_accepted_state.txt').read_text().split()[0])==2:
                paths.append(path)
        assert len(paths)==1, 'Need exactly one coherent accepted-step-2 checkpoint'
        checkpoint=paths[0]
        hashes={str(p.relative_to(checkpoint)):digest(p) for p in checkpoint.rglob('*') if p.is_file()}
        shutil.copytree(checkpoint,out/'restart/01')
        (out/'restart/last_good_checkpoint.txt').write_text('1\n')
        for path in (checkpoint/'bp3_output_metadata').iterdir():
            shutil.copy2(path,out/path.name)
        shutil.copy2(source/'cumulative_slip.csv',out/'cumulative_slip.csv')
        restore_prefix(out/'cumulative_slip.csv',2)
        times=[float(r['time']) for r in accepted[3:5]]
        if kind=='half':
            start=float(accepted[2]['time'])
            times=[(start+times[0])/2,times[0],(times[0]+times[1])/2,times[1]]
            old_clock=(float(accepted[3]['time']),float(accepted[3]['dt']),float(accepted[2]['dt']),3)
            data,offset=retime((checkpoint/'resume.z').read_bytes(),old_clock,times[0],times[0]-start)
            (out/'restart/01/resume.z').write_bytes(data)
            change=dict(old_clock=old_clock,new_time=times[0],new_dt=times[0]-start,offset=offset,
                        only_pending_time_and_dt_changed=True)
        else:
            change=None
        (out/'checkpoint_source.json').write_text(json.dumps(dict(source=str(checkpoint),hashes=hashes,retime=change),indent=2)+'\n')
        record=json.loads((source/'launch.json').read_text())
        record['cap_seconds']=900
        record['hashes']={p:s for p,s in record['hashes'].items() if p!=str(source/'run.prm')}
        values['Resume computation',]='true'
        values['End time',]=str(times[-1])
        values['Checkpointing','Steps between checkpoint']='0'
        values['Postprocess','BP3','Last accepted step']='2147483647'
        values['Postprocess','BP3','Graceful wall seconds']='850'
    # Standard ASPECT function cap lands on common times but is always combined
    # by MIN with the live physical controllers. It never forces a larger dt.
    expression='4e6'
    for target in reversed(times):
        expression=f'if(time<{target:.17g},{target:.17g}-time,{expression})'
    values['Time stepping','List of model names']+=', function'
    values['Time stepping','Function','Function expression']=expression
    values['Output directory',]=str(out)
    (out/'run.prm').write_text(render(values))
    record['command'][-1]=str(out/'run.prm')
    record['hashes'][str(out/'run.prm')]=digest(out/'run.prm')
    record['hashes'][str(Path(__file__).resolve())]=digest(Path(__file__).resolve())
    record['time_targets']=times
    record['comparison']=kind
    (out/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
    print(label,times)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind',choices=['reference','restart','half'])
    main(parser.parse_args().kind)
