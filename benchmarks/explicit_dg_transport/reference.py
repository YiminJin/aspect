"""Independent strong-form DGQ2 action on affine rectangles, in physical nodal values."""
import math
import numpy as np
from numpy.polynomial.legendre import Legendre, leggauss


def basis(x):
    return np.array([(1-x)*(1-2*x), 4*x*(1-x), x*(2*x-1)]).T


def deriv(x):
    return np.array([4*x-3, 4-8*x, 4*x-1]).T


def tensor(x, y):
    return np.einsum('qi,qj->qij', y, x).reshape(len(x),9)


nodes = np.array([0.,.5,1.])
xx, yy = np.meshgrid(nodes,nodes)
nodes_x, nodes_y = xx.ravel(), yy.ravel()
bern = np.array([[1.,0.,0.],[.25,.5,.25],[0.,0.,1.]])
to_values = np.kron(bern,bern)
to_coeff = np.linalg.inv(to_values)
mean_weights = np.kron([1.,4.,1.],[1.,4.,1.])/36


def limit(values):
    """Bound the physical polynomial in nodal coordinates; return its limited values."""
    coef = to_coeff@values
    if min(coef)>=0 and max(coef)<=1:
        return values, False
    # Evaluate the polynomial on child Lobatto points, then recover each child's
    # coefficients by interpolation. This is independent of de Casteljau.
    def enclose(c, depth):
        lo,hi = min(c),max(c)
        if depth == 0 or (lo >= 0 and hi <= 1):
            return lo,hi
        ranges = [enclose(mat@c,depth-1) for mat in children]
        return min(a for a,b in ranges), max(b for a,b in ranges)
    lo,hi = enclose(coef,4)
    mean = mean_weights@values
    if not (-3.e-14 <= mean <= 1+3.e-14):
        raise ValueError(f'inadmissible oracle mean {mean}')
    scale=1.
    if lo<0 and lo<mean:
        scale=min(scale,max(0.,mean/(mean-lo)))
    if hi>1 and hi>mean:
        scale=min(scale,max(0.,(1-mean)/(hi-mean)))
    result=mean+scale*(values-mean) if scale<1 else values
    np.testing.assert_allclose(mean_weights@result,mean,atol=4.e-15,rtol=0)
    return result, scale < 1-1.e-12


children = [to_coeff @ tensor(basis((nodes_x+i)/2),basis((nodes_y+j)/2)) @ to_values
            for i in range(2) for j in range(2)]


class Reference:
    """The material form is -u.grad(phi) plus the lift of each incoming jump.
    Separately integrate physical source and outward upwind flux for the balance."""
    def __init__(self, boxes, velocity, divergence, exterior, bounded):
        self.bounded, self.exterior = bounded, exterior
        self.boxes = boxes
        nc=len(boxes)
        if bounded:
            poly=Legendre.basis(4)
            s=np.array([-1.,-math.sqrt(3/7),0.,math.sqrt(3/7),1.])
            w=1/(20*poly(s)**2)  # Lobatto five scaled to [0,1]
        else:
            s,w=leggauss(8)
            w/=2
        z=(1+s)/2
        # Exact polynomial mass, independently formed in nodal coordinates.
        pts,wt=leggauss(4)
        m=basis((pts+1)/2).T @ (wt[:,None]*basis((pts+1)/2))/2
        mass_inv=np.linalg.inv(np.kron(m,m))
        x,y=np.meshgrid(z,z)
        x,y=x.ravel(),y.ravel()
        B=tensor(basis(x),basis(y))
        Gx=tensor(deriv(x),basis(y))
        Gy=tensor(basis(x),deriv(y))
        vol_w=np.outer(w,w).ravel()
        self.areas=[]
        self.self_mat=np.zeros((nc,9,9))
        self.upstream=[]
        self.boundary=[]
        self.source=[]
        self.outflow=[]
        self.rate=0.
        by_origin={(a,b):i for i,(a,b,c,d) in enumerate(boxes)}
        for i,(a,b,c,d) in enumerate(boxes):
            hx,hy=c-a,d-b
            area=hx*hy
            self.areas.append(area)
            X,Y=a+x*hx,b+y*hy
            u,v=velocity(X,Y)
            self.self_mat[i]= -mass_inv @ B.T @ (vol_w[:,None]*(u[:,None]*Gx/hx+v[:,None]*Gy/hy))
            div=divergence(X,Y)
            self.source.append(area*(vol_w*div)@B)
            speed=np.array([max(abs(u)), max(abs(v))])
            loss=max(0.,-min(div))
            for axis,sign in ((0,-1),(0,1),(1,-1),(1,1)):
                face=np.full_like(z,(sign+1)/2)
                fx,fy=(face,z) if axis==0 else (z,face)
                fx0,fy0=a+fx*hx,b+fy*hy
                uv=velocity(fx0,fy0)
                speed=np.maximum(speed,[max(abs(k)) for k in uv])
                vn=sign*uv[axis]
                face_w=w*(hy if axis==0 else hx)
                loss+=6*max(0.,max(vn))*(hy if axis==0 else hx)/area
                trace=tensor(basis(fx),basis(fy))
                nb=by_origin.get((a+sign*hx,b) if axis==0 else (a,b+sign*hy))
                pos=np.maximum(vn,0.)*face_w
                neg=np.minimum(vn,0.)*face_w
                if nb is None:
                    self.outflow.append((i,pos@trace))
                if np.any(neg):
                    lift=mass_inv @ (trace.T*(-neg/area))
                    self.self_mat[i]-=lift@trace
                    if nb is None:
                        self.boundary.append((i,lift,fx0,fy0,neg))
                    else:
                        upstream=tensor(basis(1-fx if axis==0 else fx),basis(1-fy if axis==1 else fy))
                        self.upstream.append((i,nb,lift@upstream))
            self.rate=max(self.rate,(speed[0]/hx+speed[1]/hy)/.1,loss/.8 if bounded else 0.)
        self.source=np.array(self.source)
        self.limited=0
        self.balance=0.
        self.integrated_source=self.integrated_flux=0.

    def action(self,state,time):
        rhs=np.einsum('cij,cj->ci',self.self_mat,state)
        for i,j,m in self.upstream:
            rhs[i]+=m@state[j]
        flux=sum(w@state[i] for i,w in self.outflow)
        for i,lift,x,y,w in self.boundary:
            exterior=self.exterior(x,y,time)
            rhs[i]+=lift@exterior
            flux+=w@exterior
        return rhs,np.sum(self.source*state),flux

    def filter(self,state):
        if self.bounded:
            for i,row in enumerate(state):
                state[i],changed=limit(row)
                self.limited+=int(changed)
        return state

    def advance(self, parent, time, duration, limit_internal=True):
        """Advance physical nodal values and return their integrated mass change.

        The source is the integral of the scalar times velocity divergence. Outward
        upwind flux is positive. The three SSPRK stages contribute weights
        1/6, 1/6 and 2/3 to the balance over each substep.
        """
        initial_mass = np.dot(self.areas, parent @ mean_weights)
        current = self.filter(parent.copy())
        steps = max(1, math.ceil(duration*self.rate))
        h = duration/steps
        for k in range(steps):
            t = time + k*h
            rate0, source0, flux0 = self.action(current, t)
            first = current + h*rate0
            if limit_internal:
                self.filter(first)
            rate1, source1, flux1 = self.action(first, t+h)
            second = (3*current + first + h*rate1)/4
            if limit_internal:
                self.filter(second)
            rate2, source2, flux2 = self.action(second, t+h/2)
            current = self.filter((current + 2*second + 2*h*rate2)/3)
            self.integrated_source += h*(source0 + source1 + 4*source2)/6
            self.integrated_flux += h*(flux0 + flux1 + 4*flux2)/6
            self.balance += h*((source0-flux0) + (source1-flux1) + 4*(source2-flux2))/6
        return current, np.dot(self.areas, current @ mean_weights) - initial_mass
