#!/usr/bin/env python3
"""Check captured production polygon/ownership data against exported stencils."""
import json
from pathlib import Path
import re

import numpy as np


def point(text):
    match=re.search(r"_M_elems = \{([^{}]+)\}",text)
    return np.fromstring(match.group(1),sep=",")


def main():
    root=Path(__file__).parent/"results"
    saved=np.genfromtxt(root/"particle_stencil_checks.csv",delimiter=",",names=True)
    reports=[]
    for particle,filename in ((3836,"cause_probe.log"),(3983,"cause_probe_3983.log"),(3986,"cause_probe_3986.log")):
        text=(root/filename).read_text()
        assert "CPDI_CAUSE_PROBE_COMPLETE" in text
        assert "Solving phase field system" not in text
        polygon_block=text.split("TARGET_POLYGON",1)[1].split("faces =",1)[0]
        polygon=np.array([np.fromstring(s,sep=",") for s in re.findall(r"_M_elems = \{([^{}]+)\}",polygon_block)])
        counts=np.zeros(len(polygon),dtype=int)
        vertex_units=[[] for _ in polygon]
        centroid_count=0
        for match in re.finditer(r"(VERTEX_SAMPLE (\d+)|CENTROID_SAMPLE)\n(.*?)(?=VERTEX_SAMPLE|CENTROID_SAMPLE|PRODUCTION_STENCIL)",text,re.S):
            unit=point(match.group(3))
            # All examined samples are away from the upper domain boundary;
            # x=0 is exactly represented. Thus these are the actual strict
            # half-open acceptance inequalities in the production helper.
            accepted=bool(np.all(unit>=0)&np.all(unit<1))
            if match.group(2) is None:
                centroid_count+=accepted
            else:
                v=int(match.group(2)); counts[v]+=accepted
                vertex_units[v].append(unit.tolist())
        assert np.all(counts<=1) and centroid_count==1
        # Reconstruct the normalized linear-on-triangle constant integral
        # solely from the captured polygon and surviving sample ownership.
        x,y=polygon.T
        cross=x*np.roll(y,-1)-np.roll(x,-1)*y
        area=abs(cross.sum()/2)
        center=np.array([np.sum((x+np.roll(x,-1))*cross),np.sum((y+np.roll(y,-1))*cross)])/(3*cross.sum())
        total=0.
        integrated_gradient=np.zeros(2)
        for v in range(len(polygon)):
            a,b=polygon[v]-center,polygon[(v+1)%len(polygon)]-center
            triangle_area=abs(a[0]*b[1]-a[1]*b[0])/2
            total+=triangle_area*(counts[v]+counts[(v+1)%len(polygon)]+centroid_count)/3
            triangle=np.array([polygon[v],polygon[(v+1)%len(polygon)],center])
            retained=[counts[v],counts[(v+1)%len(polygon)],centroid_count]
            for f in range(3):
                edge=triangle[(2+f)%3]-triangle[(1+f)%3]
                integrated_gradient+=.5*retained[f]*np.array([edge[1],-edge[0]])
        predicted=total/area
        stencil=text.split("PRODUCTION_STENCIL",1)[1]
        observed=sum(float(v) for v in re.findall(r"first = ([\deE+.-]+)",stencil))
        prior=saved[saved["id"]==particle][0]
        assert abs(predicted-observed)<1e-13
        assert abs(observed-prior["sum_w"])<1e-13
        predicted_gradient=integrated_gradient/area
        assert np.max(abs(predicted_gradient-[prior["sum_gx"],prior["sum_gy"]]))<1e-10
        reports.append(dict(particle=particle,polygon=polygon.tolist(),owner_counts=counts.tolist(),
                            centroid_owner_count=centroid_count,predicted_sum_w=predicted,
                            actual_production_sum_w=observed,
                            predicted_sum_gradient=predicted_gradient.tolist(),
                            unowned_vertices=[dict(vertex=v,position=polygon[v].tolist(),
                                                   candidate_unit_coordinates=vertex_units[v])
                                              for v in range(len(polygon)) if counts[v]==0]))
    assert reports[0]["owner_counts"].count(0)==2
    assert reports[1]["owner_counts"].count(0)==0
    assert reports[2]["owner_counts"].count(0)==1
    (root/"cpdi_cause.json").write_text(json.dumps(reports,indent=2)+"\n")
    for report in reports:
        print(f"Particle {report['particle']}: owners={report['owner_counts']}, "
              f"predicted/production sum_w={report['predicted_sum_w']:.16g}/{report['actual_production_sum_w']:.16g}")
    print("3 captured production cases passed; no phase solve or correction performed.")


if __name__ == "__main__":
    main()
