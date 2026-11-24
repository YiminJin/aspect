"""Exercise the four 5B review findings against retained, keyed physical records."""
import argparse
import csv
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

import audit

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'refactoring/phase-field-fault/results/strip18'))
from turn import write_moments


def shadow(folder, dest, changed):
    """Materialize only the modified tables; all other entries link to the original."""
    dest.mkdir()
    for f in folder.iterdir():
        if f.name not in changed:
            (dest/f.name).symlink_to(f.resolve())
        elif f.is_dir():
            shadow(f,dest/f.name,changed[f.name])
        else:
            changed[f.name](f,dest/f.name)
    if not set(changed)<=set(f.name for f in folder.iterdir()):
        raise ValueError(f'Missing mutation input in {folder}')


def change_table(key,column,new_value):
    def change(src,dst):
        with src.open() as stream:
            reader=csv.DictReader(stream)
            rows=list(reader)
        row,=[r for r in rows if all(r[k]==v for k,v in key.items())]
        row[column]=new_value
        with dst.open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=reader.fieldnames,lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    return change


def records(folder):
    keyed={}
    for path in folder.glob('*.csv'):
        with path.open() as stream:
            for row in csv.DictReader(stream):
                keyed[path.stem,row['quantity']]=row
    return keyed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('runs','reference','returned_rows','capture','baseline','dest'):
        p.add_argument(name,type=Path)
    args=p.parse_args()
    dest=args.dest.resolve()
    dest.mkdir(exist_ok=False)
    original=records(args.baseline)
    evidence=[]
    strict={(f,'strip/beta') for f in ('parent-1','coarse-2','half-2','half-3')}
    if {k for k,r in original.items() if int(r['failed'])}!=strict:
        raise ValueError('Unexpected baseline failure set')
    with (args.capture/'moments.csv').open() as stream:
        for moment in csv.DictReader(stream):
            donor=moment['source']=='donor-parent'
            for case,step in [('parent',1),('coarse',2),('half',2),('half',3)]:
                run=args.runs/(('donor-' if donor else '')+case)/'output'
                stem='tip-planes' if donor else 'sections'
                with (run/f'{stem}-{step:04}.csv').open() as table:
                    strip,=[r for r in csv.DictReader(table) if r['section']=='15']
                if strip['formed']!='1' or float(strip['beta'])!=float(moment['angle_rad']):
                    raise ValueError(f'{run}, step {step}: strip 15 lost its first selected plane')
    for (frame,key),row in sorted(original.items()):
        if key.startswith(('endpoint/','archived_endpoint/','seed_strip/','solver/')) or (frame,key) in strict:
            evidence.append(dict(control='ordinary',frame=frame,**row))
    def rerun(name,runs,returned,expected):
        output=dest/name
        audit.main([str(runs),str(args.reference),str(output),'--returned-rows',str(returned)])
        observed=records(output)
        changed={k for k in original.keys()|observed.keys() if original.get(k)!=observed.get(k)}
        new_failures={k for k,r in observed.items() if int(r['failed'])}-strict
        if not new_failures or changed!=expected or new_failures!=expected:
            raise AssertionError(f'{name}: changed {changed}, added failures {new_failures}')
        for key in sorted(changed):
            evidence.append(dict(control=name,frame=key[0],**observed[key]))
        print(f'{name}: {len(expected)} new failing comparisons; all other keyed rows identical')

    # Apply each fault only to the accepted fine 7.5 ms endpoint calculation.
    real_interval=audit.interval
    real_rows=audit.endpoint_rows
    def reversed_residual(geom,coeff,gradient,drive):
        load=geom.load(drive)
        r=load-gradient
        return load,r,(coeff<1)&(r<0),np.clip(coeff-r/(audit.KF*geom.mass),coeff,1)-coeff>1e-7
    def omitted_drive(geom,coeff,gradient,drive):
        return real_rows(geom,coeff,gradient,np.zeros_like(drive))
    for name,wrong,cols in [
            ('reversed',reversed_residual,['endpoint/residual','endpoint/strict_downhill',
                 'endpoint/projected_above_tolerance','archived_endpoint/downhill','archived_endpoint/above_stage_tolerance']),
            ('omitted',omitted_drive,['endpoint/load','endpoint/residual','archived_endpoint/max_load'])]:
        def interval(old,new,*a,**kw):
            if abs(new.time-.0075)<1e-15:
                with patch.object(audit,'endpoint_rows',wrong):
                    return real_interval(old,new,*a,**kw)
            return real_interval(old,new,*a,**kw)
        with patch.object(audit,'interval',interval):
            rerun(name,args.runs,args.returned_rows,{('half-2',col) for col in cols})

    # Preserve parent equality across the two schedules and the ordinary vs
    # returned-row byte comparison, so each mutation reaches its intended gate.
    seed={'sections-0000.csv':change_table({'section':'15'},'beta','0.1')}
    balance={'balance-0002.csv':change_table({},'iterations','77')}
    # Shadow only the indexed case trees, avoiding recursion into scratch runs.
    runs=dest/'runs'
    runs.mkdir()
    for case in ('parent','half','coarse','donor-parent','donor-coarse','donor-half'):
        changes=({'output':seed} if case=='parent' else {'output':seed|balance} if case=='half' else {})
        shadow(args.runs/case,runs/case,changes)
    shadow(args.returned_rows,dest/'rows',{'half':{'output':balance}})
    rerun('seed-and-count',runs,dest/'rows',{('seed','seed_strip/beta'),('half-2','solver/target_balance_vs_completion')})

    modified={'sections-0001.csv':change_table({'section':'15'},'beta','0.1')}
    shadow(args.capture,dest/'capture',{'parent':{'output':modified}})
    try:
        write_moments(args.runs,dest/'capture',dest/'invalid-moments.csv')
    except ValueError as e:
        if 'captured angle differs from ordinary acceptance' not in str(e):
            raise
        print(f'Captured angle mutation rejected: {e}')
    else:
        raise AssertionError('Accepted the altered capture')
    with (dest/'review-checks.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(evidence[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(evidence)


if __name__=='__main__':
    main()
