"""Extend the ordinary fault mesh to 300 km; cap square side length at 12.5 km.

The physical fault and its endpoint neighbourhoods do not move. Cell IDs are
renumbered for the Box coarse Morton tree; no histories are transferred.
"""
import hashlib
import json
from pathlib import Path
from prepare_wide_fixture import cell

HERE=Path(__file__).resolve().parent
BASE=HERE/'fixtures/modified_bp3_long_run'
OUT=HERE/'fixtures/modified_bp3_long_run_300km'


def coordinate(root,path):
    ix=root%2+2*(root//4);iy=(root//2)%2
    x,y,h=-100000+50000*ix,50000*iy,50000.
    for digit in path:
        c=int(digit);h/=2;x+=(c%2)*h;y+=(c//2)*h
    return x,y,h


def main():
    leaves={};old={}
    def admit(root,path):
        x,y,h=coordinate(root,path)
        if h>12500:
            for c in range(4): admit(root,path+str(c))
        else: leaves[f'{root}_{len(path)}:{path}']=(x,y,h)
    for name in (BASE/'target_cells.txt').read_text().split():
        root=int(name.split('_')[0]);path=name.split(':')[1]
        old[name]=cell(root,path,True)
        ix=root%2+2*(root//4)+1;iy=(root//2)%2
        shifted=(ix//2)*4+2*iy+ix%2
        assert coordinate(shifted,path)==old[name]
        admit(shifted,path)
    for root in (0,2,9,11): admit(root,'')
    assert sum(h*h for x,y,h in leaves.values())==300000.*100000.
    assert max(h for x,y,h in leaves.values())<=12500.
    # All original near-fault leaves survive exactly; only coarse far field
    # is subdivided and the new exterior strips are added.
    new_positions=set(leaves.values())
    retained={p for p in old.values() if p[2]<=12500.}
    assert retained<=new_positions
    OUT.mkdir(exist_ok=True)
    contents={'target_cells.txt':'\n'.join(sorted(leaves))+'\n'}
    for name in ('fault.txt','prestress.txt','completion.txt'):
        contents[name]=(BASE/name).read_text()
    for name,text in contents.items():
        path=OUT/name
        if path.exists(): assert path.read_text()==text,f'Refuse to overwrite {path}'
        else: path.write_text(text)
    manifest={name:{'sha256':hashlib.sha256((OUT/name).read_bytes()).hexdigest()} for name in contents}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report=dict(cells=len(leaves),old_cells=len(old),minimum_side_m=min(p[2] for p in leaves.values()),
                maximum_side_m=max(p[2] for p in leaves.values()),box=[-100000,200000,0,100000],
                retained_physical_cells=len(retained),fault_and_prestress_and_completion_unchanged=True,
                dip_degrees=60,ell_m=400)
    (OUT/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
