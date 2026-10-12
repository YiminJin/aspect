#!/usr/bin/env python3
"""Decode real ASPECT compressed VTU payloads and compare to accepted fields."""
import base64,hashlib,zlib,xml.etree.ElementTree as XML
import numpy as np
import matched_initialization as m


def decode(array):
    s=''.join(array.text.split());head=np.frombuffer(base64.b64decode(s[:16]),dtype='<u4');blocks=int(head[0])
    count=4*(3+blocks);nchars=4*((count+2)//3);header=np.frombuffer(base64.b64decode(s[:nchars]),dtype='<u4')
    packed=base64.b64decode(s[nchars:]);data=[];offset=0
    for size in header[3:]:
        size=int(size);data.append(zlib.decompress(packed[offset:offset+size]));offset+=size
    assert offset==len(packed)
    dtype={'Float32':'<f4','Float64':'<f8','Int32':'<i4','UInt32':'<u4'}[array.attrib['type']]
    return np.frombuffer(b''.join(data),dtype=dtype)


def main():
    E=m.E;out=E/'aspect-visualization';fields=np.load(E/'n512/fields.npz');hist=m.old.read(E/'n512/history.json');rows=[]
    times=[0.,.53125,.5625,.59375,.625]
    for i,t in enumerate(times):
        path=out/f'accepted-{i}.vtu';root=XML.parse(path).getroot();points=decode(root.find('.//Points/DataArray')).reshape(-1,3)
        data={a.attrib['Name']:decode(a) for a in root.findall('.//PointData/DataArray')}
        j=round(t*1024);phi=np.interp(points[:,1],fields['y'],fields['phase'][j]);error=float(max(abs(data['phase']-phi)))
        assert error<3e-8 and min(data['phase'])>=0 and max(data['phase'])<1
        source=E/'aspect-visualization'/f'snapshot-{i}.txt';samples=np.loadtxt(source,skiprows=1)
        errors={name:float(max(abs(data[name]-np.interp(points[:,1],samples[:,0],samples[:,col])))) for col,name in enumerate(
            ['phase','localization','shear_stress','cohesive_traction','cumulative_slip','velocity_x','velocity_y'],1)}
        assert errors['localization']<1e-6 and errors['shear_stress']<1e-4 and errors['cohesive_traction']<3e-5 and errors['cumulative_slip']<3e-9
        assert errors['velocity_x']<4e-9 and errors['velocity_y']==0
        rows.append(dict(time=t,errors=errors,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    record=m.old.read(E/'snapshot-native.json');record.update(decoded_payload_checks=rows,
       accepted_input_sha256=hashlib.sha256((E/'n512/fields.npz').read_bytes()).hexdigest(),
       scalar_storage='Compressed Float32 display output; accepted solver fields remain Float64',
       scope='Output-only ASPECT postprocessing on native mesh, not native coupled-solver qualification')
    m.old.write(E/'snapshot-native.json',record);print(m.old.json.dumps(rows),flush=True)

if __name__=='__main__':main()
