#!/usr/bin/env python3
"""Actual ASPECT postprocessing of accepted snapshots, without history import."""
import hashlib,json,os,re,subprocess,time,xml.etree.ElementTree as XML
from pathlib import Path
import numpy as np
from scipy.integrate import cumulative_trapezoid
import matched_initialization as m
from run_native_fe import flatten

P=m.HERE;ROOT=m.old.ROOT;BUILD=ROOT/'build-propagation-integration';E=m.E

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    data=np.load(E/'n512/fields.npz');hist=m.old.read(E/'n512/history.json');out=E/'aspect-visualization';out.mkdir(exist_ok=True)
    assert not list(out.glob('*.vtu')),'Preserve successful visualization output'
    if (E/'snapshot-native.json').exists():
        assert m.old.read(E/'snapshot-native.json')['returncode']!=0
        attempt=1
        while (E/f'snapshot-native.json.failed-{attempt}').exists():attempt+=1
        for name in ['snapshot-native.json','snapshot-native.log','snapshot-visualization.prm']:
            (E/name).rename(E/(name+f'.failed-{attempt}'))
    times=[0.,.53125,.5625,.59375,.625];velocity_errors=[]
    # Dense integration is display-only. Governing quantities use native six-point rule.
    for frame,t in enumerate(times):
        j=round(t*1024);phi=data['phase'][j];I=hist[j-1]['I'] if j else m.old.ExactCoupled(512).r.direct(phi)[0]
        rate=.256*(t-1/2048) if j else 0.;q=hist[j-1]['q'] if j else 0.;slip=hist[j-1]['slip'] if j else 0.
        T=hist[j-1]['stress_min'] if j else 0.;y=data['y'];velocity=np.zeros_like(y)
        if j:
            oldphi=data['phase'][j-1];oldI=hist[j-2]['I'] if j>1 else I
            Ta=hist[j-2]['stress_min'] if j>1 else 0.;Za=float(data['Z'][j-1]);V=hist[j-1]['V']
            beta=np.exp(-m.G/(1024*m.ETA));kappa=-m.ETA*np.expm1(-m.G/(1024*m.ETA))
            fine=np.linspace(0,1,512*64+1)
            chi=m.h(np.interp(fine,y,phi))/I;chia=m.h(np.interp(fine,y,oldphi))/oldI
            du=(T-beta*(Ta+Za*(chia-chi)))/kappa+chi*V
            cumulative=cumulative_trapezoid(du,fine,initial=0.)
            velocity=np.interp(y,fine,-rate/2+cumulative)
            velocity_errors.append(dict(time=t,upper_boundary_error=float(abs(velocity[-1]-rate/2))))
        values=np.c_[y,phi,m.h(phi)/I,np.full_like(y,T),np.full_like(y,q),np.full_like(y,slip),velocity,np.zeros_like(y)]
        with (out/f'snapshot-{frame}.txt').open('w') as f:
            f.write(f'{t:.17g}\n');np.savetxt(f,values,fmt='%.17g')
    plugin=BUILD/'snapshot-visualization/libsnapshot_visualization.release.so'
    prm=E/'snapshot-visualization.prm'
    text='\n'.join(re.sub(r'set List of postprocessors\s*=.*','set List of postprocessors = particles, P2 accepted snapshot visualization',line)
                   for line in flatten(P/'quadrature_evidence/q2-refined.prm'))
    text+=f'''
set Additional shared libraries = {plugin}
set Output directory = {out}/
set End time = 0
set Nonlinear solver scheme = no Advection, no Stokes
subsection Geometry model
 subsection Box
  set Y repetitions = 32
 end
end
subsection Material model
 subsection Phase field fault
  set Bulk history representation = particle stress
  set Evolve phase field = false
 end
end
subsection Boundary velocity model
 subsection Function
  set Function expression = 0; 0
 end
end
subsection Postprocess
 set List of postprocessors = particles, P2 accepted snapshot visualization
 subsection Particles
  set Data output format = none
 end
 subsection P2 accepted snapshot visualization
  set Input directory = {out}
 end
end
'''
    prm.write_text(text+'\n');env=dict(os.environ,ASPECT_SOURCE_DIR=str(ROOT),ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    start=time.perf_counter()
    with (E/'snapshot-native.log').open('w') as log:
        child=subprocess.Popen([str(BUILD/'aspect-release'),str(prm)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        _,status,usage=os.wait4(child.pid,0);child.returncode=os.waitstatus_to_exitcode(status)
    record=dict(returncode=child.returncode,runtime_seconds=time.perf_counter()-start,peak_RSS_KiB=usage.ru_maxrss,
        compiler='/opt/gcc/12.4.0/bin/g++',deal_II='9.6-local',executable_sha=sha(BUILD/'aspect-release'),plugin_sha=sha(plugin),input_sha=sha(prm),
        meaning='Actual ASPECT postprocessor: accepted exact-bulk n512 reference-initialization snapshots on native 16x512 mesh; output-only, no history import or physical advancement.',
        display_only_velocity_quadrature=velocity_errors)
    m.old.write(E/'snapshot-native.json',record)
    assert child.returncode==0,'See snapshot-native.log'
    assert max(r['upper_boundary_error'] for r in velocity_errors)<1e-7
    records=XML.parse(out/'accepted.pvd').findall('.//DataSet');assert [float(r.attrib['timestep']) for r in records]==times
    sizes=[]
    for r in records:
        path=out/r.attrib['file'];xml=XML.parse(path);piece=xml.find('.//Piece');fields=[v.attrib.get('Name') for v in piece.findall('./PointData/DataArray')]
        assert set(['phase','localization','shear_stress','cohesive_traction','cumulative_slip','velocity_x','velocity_y'])<=set(fields)
        assert int(piece.attrib['NumberOfCells'])==16*512
        assert path.stat().st_size<3*1024**2,'Keep each compressed frame small'
        sizes.append(dict(file=path.name,bytes=path.stat().st_size,cells=int(piece.attrib['NumberOfCells']),points=int(piece.attrib['NumberOfPoints']),fields=fields))
    record['frames']=sizes;record['total_VTU_bytes']=sum(v['bytes'] for v in sizes);m.old.write(E/'snapshot-native.json',record)
    print(json.dumps(record,indent=2),flush=True)

if __name__=='__main__':main()
