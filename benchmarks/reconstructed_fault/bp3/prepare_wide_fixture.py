"""Preserve the old physical leaf mesh; add graded side strips to a centered box."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE/'fixtures/modified_bp3'
OUT=HERE/'fixtures/modified_bp3_wide'


def cell(root,path,wide):
    # The distributed Box mesh numbers coarse cells in Morton order.
    if wide:
        ix=(root%2)+2*(root//4);iy=(root//2)%2
        x,y,h=-50000+ix*50000,iy*50000,50000.
    else:
        assert root==0
        x,y,h=0.,0.,100000.
    for digit in path:
        c=int(digit);h/=2;x+=(c%2)*h;y+=(c//2)*h
    return x,y,h


def target():
    leaves={}
    original=set()
    for name in (BASE/'target_cells.txt').read_text().splitlines():
        root_level,path=name.split(':');root,level=map(int,root_level.split('_'))
        assert root==0 and len(path)==level and level>=1
        quadrant=int(path[0]);new_root=(1,4,3,6)[quadrant]
        key=f'{new_root}_{level-1}:{path[1:]}'
        position=cell(root,path,False)
        assert position==cell(new_root,path[1:],True)
        leaves[key]=position;original.add(position)
    def exterior(root,path):
        x,y,h=cell(root,path,True)
        distance=max(-x-h,x-100000,0.)
        wanted=6250. if distance<12500 else 12500. if distance<25000 else 25000.
        if h>wanted:
            for c in range(4): exterior(root,path+str(c))
        else: leaves[f'{root}_{len(path)}:{path}']=(x,y,h)
    for root in (0,2,5,7): exterior(root,'')
    assert sum(h*h for x,y,h in leaves.values())==200000.*100000.
    assert {p for p in leaves.values() if 0<=p[0]<100000}==original
    return leaves,original


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-export',type=Path)
    args=parser.parse_args()
    leaves,original=target()
    if args.check_export:
        actual={}
        for path in sorted(args.check_export.glob('mesh_cells_rank*.csv')):
            for row in csv.DictReader(path.open()):
                name=row['cell'];x,y,h=[float(row[k]) for k in ('x','y','h')]
                assert name not in actual
                actual[name]=(x-h/2,y-h/2,h)
        assert set(actual)==set(leaves),(len(actual),len(leaves))
        error=max(abs(a-b) for name in leaves for a,b in zip(actual[name],leaves[name]))
        assert error<1e-8,error
        report=dict(passed=True,cells=len(actual),original_cells=len(original),
                    added_cells=len(actual)-len(original),maximum_coordinate_error_m=error,
                    minimum_cell_width_m=min(p[2] for p in actual.values()),
                    new_strip_cell_widths_m=sorted({p[2] for p in actual.values() if p[0]<0 or p[0]>=100000}))
        (args.check_export/'mesh_verification.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2));return
    OUT.mkdir(exist_ok=True)
    data='\n'.join(sorted(leaves))+'\n'
    path=OUT/'target_cells.txt'
    if path.exists(): assert path.read_text()==data,'Refuse to replace a different target mesh'
    else: path.write_text(data)
    record={'target_cells.txt':{'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                                'source':'prepare_wide_fixture.py; unchanged central leaf cells plus graded lateral extension'}}
    (OUT/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(f'{len(leaves)} cells: {len(original)} unchanged central cells, {len(leaves)-len(original)} added cells')


if __name__=='__main__': main()
