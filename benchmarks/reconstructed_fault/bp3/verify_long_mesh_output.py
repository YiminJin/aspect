"""Check the actual 300-km mesh and genuine Q2 VTU cells (not linear patches)."""
import argparse
import base64
import csv
import json
import struct
import zlib
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from prepare_300km_fixture import coordinate,OUT


def vtk_array(node):
    text=''.join(node.text.split())
    n=struct.unpack('<I',base64.b64decode(text[:8])[:4])[0]
    header_chars=4*((4*(3+n)+2)//3)
    header=struct.unpack('<'+'I'*(3+n),base64.b64decode(text[:header_chars]))
    packed=base64.b64decode(text[header_chars:]);start=0;pieces=[]
    for size in header[3:]:
        pieces.append(zlib.decompress(packed[start:start+size]));start+=size
    return np.frombuffer(b''.join(pieces),dtype={'Int32':'<i4','UInt8':'u1'}[node.get('type')])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);args=p.parse_args()
    target={}
    for name in (OUT/'target_cells.txt').read_text().split():
        x,y,h=coordinate(int(name.split('_')[0]),name.split(':')[1]);target[name]=(x+h/2,y+h/2,h)
    actual={}
    for part in args.run.glob('initial_mesh_*.csv'):
        with part.open() as f:
            for r in csv.DictReader(f):
                assert r['cell'] not in actual
                actual[r['cell']]=tuple(float(r[k]) for k in ('x','y','h'))
    assert actual.keys()==target.keys(),(len(actual),len(target))
    error=max(abs(a-b) for cell in actual for a,b in zip(actual[cell],target[cell]))
    assert error<1e-8 and max(p[2] for p in actual.values())<=12500.
    cells=[]
    for path in sorted((args.run/'solution').glob('*.vtu')):
        tree=ET.parse(path)
        values={n.get('Name'):vtk_array(n) for n in tree.findall('.//Cells/DataArray') if n.get('Name') in ('types','offsets')}
        assert np.all(values['types']==70),path  # VTK_LAGRANGE_QUADRILATERAL
        assert np.all(np.diff(np.r_[0,values['offsets']])==9),path
        cells.append(dict(file=str(path.relative_to(args.run)),Q2_cells=len(values['types'])))
    assert cells
    fault_file=args.run/'reconstructed_faults/reconstructed_faults-00000.vtu'
    f=ET.parse(fault_file)
    points=np.fromstring(f.find('.//Points/DataArray').text,sep=' ').reshape(-1,3)
    expected=np.loadtxt(OUT/'fault.txt')[:,:2]
    np.testing.assert_array_equal(points[:,:2],expected)
    np.testing.assert_allclose((points[0]+points[-1])/2,[50000,50000,0],rtol=0,atol=1e-9)
    sn=np.sqrt(3)/2;xt=50000*(1+.5/sn)
    windows={}
    for lo,hi in [(0,2000),(15000,18000),(35000,45000),(113000,116000)]:
        sizes=[h for x,y,h in actual.values() if abs((xt-x)*sn-(100000-y)*.5)<=h*(sn+.5)/2
               and lo<=(xt-x)*.5+(100000-y)*sn<=hi]
        assert sizes and set(sizes)=={97.65625}
        windows[f'{lo}-{hi}']=dict(cells=len(sizes),side_m=97.65625)
    result=dict(passed=True,cells=len(actual),max_side_m=max(p[2] for p in actual.values()),
                coordinate_error_m=error,fault_vertices=len(points),fault_midpoint_m=[50000,50000],
                crossed_windows=windows,VTK_type=70,nodes_per_cell=9,outputs=cells)
    (args.run/'mesh_output_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS: exact 300-km tree, max side 12.5 km, unchanged fault; all bulk VTU cells are genuine 9-node Q2.')


if __name__=='__main__':main()
