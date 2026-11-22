"""Independent vertical-slice integration of DGQ2 stress moments on strip-cut cells."""
import json

import numpy as np

from check import bulk, gauss, labels, lagrange, require


def strips(origin, h, vertices, length, tangent, normal, width):
    """Affine (station, normal distance) on the unit cell for each candidate strip."""
    corners = np.array([[0,0],[1,0],[0,1],[1,1]])
    delta = origin-vertices[:-1]
    station = np.column_stack((h*tangent, np.sum(delta*tangent,axis=1)))
    distance = np.column_stack((h*normal, np.sum(delta*normal,axis=1)))
    s = station[:,:2] @ corners.T + station[:,2,None]
    d = distance[:,:2] @ corners.T + distance[:,2,None]
    ids = np.flatnonzero((s.max(1)>0)&(s.min(1)<length)&(d.max(1)>-width)&(d.min(1)<width))
    edges = []
    for j in ids:
        for line, bounds in ((station[j],(0,length[j])), (distance[j],(-width,width))):
            vals = corners@line[:2]+line[2]
            for bound in bounds:
                if vals.min()<bound<vals.max():
                    edges.append(line-[0,0,bound])
    for i,j in enumerate(ids):
        for k in ids[:i]:
            for line in (distance[j]-distance[k],distance[j]+distance[k]):
                vals=corners@line[:2]+line[2]
                if vals.min() < -1.e-14*h and vals.max() > 1.e-14*h:
                    edges.append(line)
    return ids, station[ids], distance[ids], edges


def vertical_panels(lines):
    # Split the x-axis wherever two lines intersect or a line enters the
    # cell. Between these events the ordering of all y cuts is constant.
    lines = [line/np.linalg.norm(line[:2]) for line in lines if np.linalg.norm(line[:2])>1.e-15]
    lines.extend([np.array([0.,1.,0.]),np.array([0.,1.,-1.])])
    events=[0.,1.]
    for i,(a,b,c) in enumerate(lines):
        if abs(b)<1.e-14:
            events.append(-c/a)
        for d,e,f in lines[:i]:
            det=a*e-b*d
            if abs(det)>1.e-13:
                x=(b*f-c*e)/det
                y=(c*d-a*f)/det
                if -1.e-13<=y<=1.+1.e-13:
                    events.append(x)
    xs=np.unique(np.clip(events,0,1))
    graphs=np.array([[-a/b,-c/b] for a,b,c in lines if abs(b)>1.e-14])
    panels=[]
    for a,b in zip(xs[:-1],xs[1:]):
        if b-a<1.e-14: continue
        mid=graphs[:,0]*(.5*(a+b))+graphs[:,1]
        order=mid.argsort()
        for i,j in zip(order[:-1],order[1:]):
            if mid[j]-mid[i]>1.e-14 and 0<.5*(mid[i]+mid[j])<1:
                panels.append([a,b,*graphs[i],*graphs[j]])
    return np.array(panels)


def integrate(order, panels, ids, station, distance, length, width, frame, label, nodal, m, kappa, check_cut=True):
    q,w=gauss(order)
    a,b,low,offset,high,top=panels.T
    x=a[:,None]+(b-a)[:,None]*q
    lower=low[:,None]*x+offset[:,None]
    dy=(high[:,None]*x+top[:,None])-lower
    y=(lower[:,:,None]+dy[:,:,None]*q).ravel()
    weights=((b-a)[:,None,None]*w[None,:,None]*dy[:,:,None]*w).ravel()
    x=np.repeat(x.ravel(),order)
    xy=np.column_stack((x,y))
    s=xy@station[:,:2].T+station[:,2]
    d=abs(xy@distance[:,:2].T+distance[:,2])
    active=(s>=0)&(s<=length[ids])&(d<=width)
    dist=np.where(active,d,np.inf)
    nearest=dist.argmin(axis=1)
    inside=active.any(axis=1)
    seg=ids[nearest]
    if check_cut and len(np.unique(np.where(inside,seg,-1)))<2:
        return None
    xi=np.clip(s[np.arange(len(x)),nearest]/length[seg],0.,1.)
    shape=(lagrange(y)[:,:,None]*lagrange(x)[:,None,:]).reshape(-1,9)
    phi,dxx,dyy,dxy,xx,yy,xy_stress=(shape@nodal).T
    d0=np.maximum(phi,0.)
    weight=m*d0*(1+d0)/(1-d0)**2
    Ih=(1-xi)*label[seg,3]+xi*label[seg+1,3]
    V=(1-xi)*label[seg,6]+xi*label[seg+1,6]
    slip=(2*kappa*inside*weight*V/Ih)[:,None]*frame[seg]
    raw=np.column_stack((xx,yy,xy_stress))-slip
    delta=np.column_stack((dxx,dyy,dxy))-raw
    monomials=np.stack([x**(j%3)*y**(j//3) for j in range(9)])
    return (monomials*weights)@delta, abs(raw).max()


def check_moments(directory,grid,step,dt):
    output=directory/"output"
    data=np.sort(bulk(output,step),order=["cell","q"]).reshape(-1,9)
    label=labels(output,step)
    vertices=label[:,1:3]
    edge=np.diff(vertices,axis=0)
    length=np.linalg.norm(edge,axis=1)
    t=edge/length[:,None]
    n=np.column_stack((-t[:,1],t[:,0]))
    frame=np.column_stack((t[:,0]*n[:,0],t[:,1]*n[:,1],.5*(t[:,0]*n[:,1]+t[:,1]*n[:,0])))
    prm=(directory/"input.prm").read_text().splitlines()
    def parameter(name):
        return float(next(line.split("=")[1] for line in prm if "set "+name+" =" in line))
    width,m=parameter("Profile half width"),parameter("Degradation magnitude")
    # At every accepted rate, the independently reconstructed
    # Q3 smooth remainder (viscous gradient + projected relaxed DG parent)
    # is a tensor Q2 polynomial on each Cartesian cell.
    material=json.loads((output/"parameters.json").read_text())["Material model"]
    eta=float(material["Simple model"]["Viscosity"]["value"])
    modulus=float(material["Phase field fault"]["Shear modulus"]["value"])
    kappa=-eta*np.expm1(-modulus*dt/eta)
    peak=scale=uncertainty=unsplit_defect=0.
    worst=largest=uncertain=None
    count=0
    for cell in data:
        h=1/grid
        origin=np.floor((np.array([cell["x"][0],cell["y"][0]])+.5)*grid)*h-.5
        ids,station,distance,edges=strips(origin,h,vertices,length,t,n,width)
        if not edges: continue
        panels=vertical_panels(edges)
        smooth=np.column_stack([cell[key] for key in ("exx","eyy","exy")])
        smooth+=2*kappa*(cell["chi"]*cell["rate"])[:,None]*frame[cell["seg"]]
        nodal=np.column_stack([cell["phi"],cell["xx"],cell["yy"],cell["xy"],smooth])
        args=(panels,ids,station,distance,length,width,frame,label,nodal,m,kappa)
        fine=integrate(10,*args)
        if fine is None: continue
        coarse=integrate(6,*args)
        require(coarse is not None,"Vertical panels did not resolve the same cut")
        # The former tensor Q8 projection matches Q8 moments of these same
        # raw fields. Compare its counterfactual defect on the identical cells.
        unit_square=np.array([[0.,1.,0.,0.,0.,1.]])
        tensor_moments=integrate(8,unit_square,*args[1:],check_cut=False)[0]
        unsplit_defect=max(unsplit_defect,abs(fine[0]-tensor_moments).max()*h*h)
        count+=1
        defect=abs(fine[0]).max()*h*h
        delta=abs(fine[0]-coarse[0]).max()*h*h
        where=(cell["cell"][0],tuple(origin))
        if defect>peak: peak,worst=defect,where
        if fine[1]*h*h>scale: scale,largest=fine[1]*h*h,where
        if delta>uncertainty: uncertainty,uncertain=delta,where
    require(count>0 and scale>0, "No resolved active cut cells")
    print(f"{directory.name} step {step}: {count} strip-cut cells, signed-component/Q2 moment max {peak:.10g} Pa m² at {worst}, "
          f"reference scale {scale:.10g} Pa m² at {largest}, E={peak/scale:.5e}, "
          f"Q6/Q10 uncertainty {uncertainty:.5g} Pa m² (scaled {uncertainty/scale:.5e}) at {uncertain}; "
          f"unsplit Q8 on identical inputs {unsplit_defect:.8g} Pa m², scaled {unsplit_defect/scale:.5e}",flush=True)
    require(uncertainty/scale <= .001,"Cut reference exceeds 0.1% uncertainty budget")
    if grid==64: require(peak/scale <= .01,"Accepted cut stress exceeds 1% budget")
    if directory.name=="inclined-fit-64":
        require(unsplit_defect/scale > .01, "Fitted projection control lost its discriminating cut")
    return peak/scale
