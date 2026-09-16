"""Read the preserved per-iteration lower-bound probe; no constitutive reevaluation."""
import argparse
import json
from pathlib import Path
import re
import numpy as np

def main(directory):
    directory=Path(directory)
    log=directory.with_suffix('.log').read_text()
    blocks=re.split(r'\*\*\* Timestep (\d+):',log)
    result={'steps':{},'Ih':{}}
    for k in range(1,len(blocks),2):
        step=int(blocks[k]); text=blocks[k+1]
        path=directory/f'nonlinear_bounds_{step}.csv'
        if not path.exists(): continue
        table=np.genfromtxt(path,names=True,delimiter=',')
        rows=[]
        accepted=re.findall(r'line search accepted after (\d+) rejected candidates; alpha=([^\s]+)\.',text)
        for it in np.unique(table['iteration']).astype(int):
            r=table[table['iteration']==it]; free=r['prescribed']==0
            rows.append(dict(iteration=int(it),free=int(np.sum(free&(r['lower_active']==0))),
                lower_active=int(np.sum(free&(r['lower_active']!=0))),
                minimum_V=float(r['V'][free].min()),
                negative_Fmin=int(np.sum(free&(r['Fmin_weak_density']<0))),
                Fmin_range=[float(r['Fmin_weak_density'][free].min()),float(r['Fmin_weak_density'][free].max())],
                alpha_max=float(r['alpha_max'][0]),bulk=float(r['bulk'][0]),surface=float(r['surface'][0]),
                accepted_alpha=float(accepted[it][1]) if it<len(accepted) else None))
        linear=re.findall(r'Fault linear solve: iterations=(\d+), estimated=([^,]+), fresh=([^,]+), target=([^,]+)',text)
        result['steps'][step]=dict(iterations=rows,linear_directions=len(linear),
            fresh_checks_pass=all(float(x[2])<=float(x[3]) for x in linear))
    result['Ih']['cold_seconds']=[float(v) for v in re.findall(r'End fault I_h preparation: integrated, ([\deE+.\-]+)',log)]
    result['Ih']['warm_seconds']=[float(v) for v in re.findall(r'value cache hit, integration requests=0, ([\deE+.\-]+)',log)]
    result['Ih']['all_later_preparations_hit']=len(result['Ih']['cold_seconds'])==1
    result['Ih']['cold_breakdown']=re.findall(r'Fault I_h phases[^\n]+',log)
    output=directory/'cache_audit.json'
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    main(parser.parse_args().directory)
