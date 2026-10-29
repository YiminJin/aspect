#!/usr/bin/env python3
"""Join exported restart segments, retaining their actual accepted sequence."""
import argparse,json
import numpy as np
from continuation_analysis import BASE,OUT,read,save_csv

def merge(name,parts):
    data=[];histories=[];contours=[];last=-np.inf;provenance=[]
    for token in parts:
        part,sep,cut=token.partition(':');upper=float(cut) if sep else np.inf
        d=np.load(BASE/(part+'-plot-data.npz'));keep=(d['times']>last+1e-12)&(d['times']<=upper+1e-12)
        if not np.any(keep):continue
        data.append((d,keep));lo=float(d['times'][keep][0]);last=float(d['times'][keep][-1]);provenance.append(dict(part=part,first=lo,last=last))
        for suffix,dest in [('history',histories),('contours',contours)]:
            a=read(OUT/(part+'-'+suffix+'.csv'));dest.append(a[(a['time_s']>=lo-1e-12)&(a['time_s']<=last+1e-12)])
    assert len(data)
    first=data[0][0];keys=['phase','stress','tauxx','tauyy','Y','rate','maxrate','forces','steps','times','reaction','increment','growth','delta']
    result={k:np.concatenate([d[k][mask] for d,mask in data]) for k in keys}
    for key in ['x','y','initial','parent']:result[key]=first[key]
    assert np.min(np.diff(result['phase'],axis=0))>=0 and np.all(np.diff(result['times'])>1e-12)
    np.savez_compressed(BASE/(name+'-plot-data.npz'),**result)
    for suffix,arrays in [('history',histories),('contours',contours)]:
        a=np.concatenate(arrays);save_csv(OUT/(name+'-'+suffix+'.csv'),a.dtype.names,a.tolist())
    (OUT/(name+'-segments.json')).write_text(json.dumps(provenance,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('parts',nargs='+');a=p.parse_args();merge(a.name,a.parts)
