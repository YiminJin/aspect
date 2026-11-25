"""Replay the captured first turn and final signed strip root on each C1 layout."""
import csv
import sys
from pathlib import Path
import mpmath as mp

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'strip8'))
from replay import Direct, moments


def keyed(path, prefix):
    result={}
    for file in path.glob(prefix+'-*'):
        for line in file.read_text().splitlines():
            cell,local,*v=line.split()
            key=(cell,int(local))
            if key in result:
                raise ValueError(key)
            result[key]=tuple(map(float,v))
    return dict(sorted(result.items()))


def records(files,keys):
    return {tuple(row[k] for k in keys):row for file in files for row in csv.DictReader(file.open())}


def main(paths):
    mp.mp.dps=65
    angles=[float(list(csv.DictReader((p/'sections-0001.csv').open()))[8]['beta']) for p in paths]
    for table,keys in [('fields',['cell','component','local']),('points',['cell','q']),('sections',['section'])]:
        left,right=(records(p.glob(table+'-0000*.csv'),keys) for p in paths)
        if not left or left!=right:
            raise ValueError('Different incoming '+table)
        print('Identical incoming',table,len(left))
    stage=[keyed(p,'turn-fields') for p in paths]
    mappings=records(paths[0].glob('fields-0000*.csv'),['cell','component','local'])
    # System-local FE indices follow the order written by system_to_component_index.
    components={}
    for cell,c,j in mappings:
        if c in ("0","1","2","7"):
            components.setdefault(cell,[]).append(c)
    for c in sorted({k[1] for k in mappings}):
        diffs=[stage[1][k][0]-v[0] for k,v in stage[0].items() if components[k[0]][k[1]]==c]
        if diffs:
            print('stage component',c,'min/max difference',min(diffs),max(diffs))
    data=[keyed(p,'turn-points') for p in paths]
    if len(data[0])!=180 or data[0].keys()!=data[1].keys():
        raise ValueError('Missing strip points')
    print('Point deltas area,kernel,F,u,v,phi',*[max(abs(v[j]-data[1][k][j]) for k,v in data[0].items()) for j in range(6)])
    for p,pts in zip(paths,data):
        width=float(list(csv.DictReader((p/'sections-0000.csv').open()))[8]['ds'])
        free,mob,beta,pressure=map(mp.mpf,(p/'root-input').read_text().split())
        direct=Direct(pts,width,pressure)
        live=[list(map(float,l.split())) for l in (p/'turn-sums-0').read_text().splitlines()]
        print('global minus ordered moments',[a-b for a,b in zip(live[1],moments(pts))])
        for a,f,score in live[2:]:
            force,reference=direct.value(mp.mpf(a))
            print('angle',a,'resistance difference',float(force)-f,'score',score,'direct',float(reference),'error',float(score-reference))
        center=mp.mpf(angles[0])
        step=mp.mpf('1e-5')
        value=lambda a: direct.value(a)[1]
        curvature=(value(center-step)-2*value(center)+value(center+step))/step**2
        print('direct difference between maxima',float(value(mp.mpf(angles[1]))-value(center)),'curvature',float(curvature))
        row=list(csv.DictReader((p/'sections-0001.csv').open()))[8]
        def reaction(v):
            x=abs(v)*mp.mpf(.0025)/mp.mpf(.01)
            age=10*mp.exp(-x)-mp.mpf(.0025)*mp.expm1(-x)/x
            return pressure*mp.mpf(.015)*mp.asinh(v/mp.mpf(.002)*mp.exp((mp.mpf(.6)+mp.mpf(.02)*mp.log(mp.mpf(.001)*age/mp.mpf(.01)))/mp.mpf(.015)))+1000*v
        v=mp.findroot(lambda v:v+mob*reaction(v)-free,(mp.mpf('1e-13'),mp.mpf('2e-13')))
        if abs(mp.mpf(row['beta'])-beta)>mp.mpf('1e-17') or abs(mp.mpf(row['V'])/v-1)>mp.mpf('1e-10') or abs(reaction(v)-mp.mpf(row['force']))>mp.mpf('1e-10'):
            raise ValueError('Published pair differs from the reconstructed signed strip root')
        print('final angle, signed rate, relative root error, resistance error',float(beta),row['V'],float(mp.mpf(row['V'])/v-1),float(mp.mpf(row['force'])-reaction(v)))


if __name__=='__main__':
    main(list(map(Path,sys.argv[1:])))
