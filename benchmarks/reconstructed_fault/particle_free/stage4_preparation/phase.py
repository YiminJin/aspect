"""Native Bernstein field/connected-contour reduction, from stage4_release.

Cell-local quadratic roots retain both DG traces; sampling identifies only the
connected component. Compare two sampling densities to measure that choice.
"""
from pathlib import Path
import numpy as np
from scipy.ndimage import label,binary_dilation
from scipy.optimize import brentq
H=1/32;NX=256;NY=128
def read(p):return np.atleast_1d(np.genfromtxt(p,delimiter=',',names=True))
def coefficients(out,step):
    files=sorted(Path(out).glob(f'release-phase-{step:04d}-r*.csv*'));assert len(files)==4
    a=np.concatenate([np.loadtxt(p,delimiter=',',skiprows=1,usecols=range(1,12)) for p in files])
    ix=np.rint(a[:,0]/H-.5).astype(int);iy=np.rint(a[:,1]/H-.5).astype(int)
    assert len(a)==NX*NY and len(np.unique(NX*iy+ix))==NX*NY
    b=np.empty((NY,NX,3,3));b[iy,ix]=a[:,2:].reshape(-1,3,3);return b
def basis(t):return np.stack([(1-t)**2,2*t*(1-t),t*t],axis=-1)
def sample(b,n=8):
    q=basis((np.arange(n)+.5)/n)
    return np.einsum('ai,yxij,bj->yaxb',q,b,q).reshape(NY*n,NX*n)
def value(b,x,y):
    x=np.asarray(x);y=np.broadcast_to(y,x.shape)
    ix=np.minimum((x/H).astype(int),NX-1);iy=np.minimum((y/H).astype(int),NY-1)
    return np.einsum('...i,...ij,...j->...',basis(y/H-iy),b[iy,ix],basis(x/H-ix))
def normal_maximum(b,u):
    d=b@basis(np.asarray(u));a=d[0]-2*d[1]+d[2];v=(d[0]-d[1])/a if a<0 else -1.
    choices=[(d[0],0.),(d[2],1.)]
    if 0<v<1:choices.append((float(basis(np.asarray(v))@d),float(v)))
    return max(choices)
def connected_front(b,n=8,level=.05):
    f=sample(b,n);labels,count=label(f>=level);seed=labels[64*n,128*n];assert seed>0
    connected=labels==seed;cells=connected.reshape(NY,n,NX,n).any(axis=(1,3));nearby=binary_dilation(cells)
    candidates=np.argwhere(nearby & (np.max(b,axis=(2,3))>=level));right=-np.inf;extrema=[]
    for iy,ix in candidates[np.argsort(candidates[:,1])[::-1]]:
        if (ix+1)*H<right:break
        cell=b[iy,ix];u=np.linspace(0,1,65);g=np.array([normal_maximum(cell,v)[0]-level for v in u]);positive=np.flatnonzero(g>=0)
        if not len(positive):continue
        j=positive[-1];r=1. if j==64 else brentq(lambda v:normal_maximum(cell,v)[0]-level,u[j],u[j+1],xtol=1e-13)
        xx=(ix+r)*H;yy=(iy+normal_maximum(cell,r)[1])*H
        extrema.append((xx,yy));right=max(right,xx)
    assert np.isfinite(right)
    areas=np.bincount(labels.ravel())*(H/n)**2;areas[0]=0
    # Preserve normal locations when contour extrema are numerically tied.
    # Also retain both half-domain extents to distinguish an inherited offset
    # from newly developed geometry.
    ys=[yy for xx,yy in extrema if abs(xx-right)<=1e-9]
    upper=max((xx for xx,yy in extrema if yy>=2.),default=right)
    lower=max((xx for xx,yy in extrema if yy<=2.),default=right)
    return dict(right_x=float(right),right_upper_x=float(upper),right_lower_x=float(lower),right_y=float(np.mean(ys)),right_y_range=[float(min(ys)),float(max(ys))],sampled_components=int(count),connected_sample_area=float(connected.sum()*(H/n)**2),detached_sample_area=float(sum(areas)-areas[seed]))
def fronts(b,n=8):
    right=connected_front(b,n);left=connected_front(b[:,::-1,:,::-1],n)
    return dict(right_x=right['right_x'],right_upper_x=right['right_upper_x'],right_lower_x=right['right_lower_x'],right_y=right['right_y'],right_y_range=right['right_y_range'],left_x=8-left['right_x'],left_upper_x=8-left['right_upper_x'],left_lower_x=8-left['right_lower_x'],left_y=left['right_y'],left_y_range=left['right_y_range'],sampled_components=right['sampled_components'],detached_sample_area=right['detached_sample_area'])
