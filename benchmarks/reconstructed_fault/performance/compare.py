"""Summarize measured preparations and compare accepted evolving trajectories."""
import argparse
import json
from pathlib import Path
import re
import numpy as np

def timing(path):
    path=Path(path)
    log=path.with_suffix('.log').read_text()
    resources=json.loads(path.with_suffix('.resources.json').read_text())
    numbers=lambda pattern: [float(x) for x in re.findall(pattern,log)]
    cold=numbers(r'End fault I_h preparation: integrated, ([\d.eE+\-]+) s')
    warm=numbers(r'value cache hit, integration requests=0, ([\d.eE+\-]+) s')
    properties=numbers(r'End fault surface-property preparation: ([\d.eE+\-]+) s')
    return dict(status=resources['status'],wall_seconds=resources['wall_seconds'],
                peak_rss_KiB=resources['peak_rss_KiB'],integrated_Ih_seconds=cold,
                completed_value_hits_seconds=warm,property_preparation_seconds=properties,
                integrated_Ih_sum=sum(cold),property_preparation_sum=sum(properties),
                preparation_fraction=sum(properties)/resources['wall_seconds'],
                projected_relative_differences=numbers(r'maximum projected nodal relative difference=([\d.eE+\-]+)'),
                cold_detail=re.findall(r'Fault I_h cold detail[^\n]*',log),
                cell_work=re.findall(r'Cell I_h: [^\n]*',log))

def compare(first,second):
    result={}
    for kind in ('surface','phase','time','particles','fault','history','constitutive_normal'):
        result[kind]={}
        for original in sorted(first.glob(kind+'_[0-9]*.csv')):
            other=second/original.name
            if not other.exists(): continue
            a=np.atleast_1d(np.genfromtxt(original,names=True,delimiter=','))
            b=np.atleast_1d(np.genfromtxt(other,names=True,delimiter=','))
            assert a.shape==b.shape and a.dtype.names==b.dtype.names
            columns={}
            for name in a.dtype.names:
                error=float(np.max(np.abs(a[name]-b[name])))
                scale=float(np.max(np.abs(a[name])))
                columns[name]=dict(absolute=error,relative=error/scale if scale else None,
                                   bitwise=bool(np.array_equal(a[name],b[name])))
            result[kind][original.stem]=columns
    guards={}
    for path in (first,second):
        guards[str(path)]={file.stem:json.loads(file.read_text())['passed']
                           for file in sorted(path.glob('guard_*.json'))}
    return dict(fields=result,guards=guards)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories',nargs='+',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report={'timings':{str(p):timing(p) for p in args.directories}}
    if len(args.directories)==2:
        report['comparison']=compare(*args.directories)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='comparison'},indent=2))
