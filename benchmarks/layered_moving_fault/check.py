"""Independent Cartesian Q2 reconstruction of the frozen donor and moving target.

Usage: check.py TARGET_MATRIX DONOR_OUTPUT FIGURE [--zero OUTPUT --reverse OUTPUT]
"""
import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def same(a, b, tol, name):
    error = float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
    require(np.isfinite(error) and error <= tol, f"{name}: {error:g} exceeds {tol:g}")
    return error


def read(path):
    rows = np.atleast_1d(np.genfromtxt(path, names=True, dtype=None, encoding="utf8", delimiter=","))
    for k in rows.dtype.names:
        if rows[k].dtype.kind in "fiu":
            require(np.isfinite(rows[k]).all(), f"nonfinite {path}: {k}")
    return rows


@lru_cache(None)
def gauss(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return (1+x)/2, w/2


def basis(s):
    g = gauss(3)[0]
    return np.stack([np.prod([(s-b)/(a-b) for b in g if b != a], axis=0) for a in g], axis=-1)


keys = "phi vx vy tauxx tauyy tauxy Lxx Lxy Lyx Lyy eta pressure".split()
h = .5
origin = np.array([-6., -2.])


class Frame:
    def __init__(self, directory, step, donor=False):
        prefix = f"material-{step}-"
        self.surface = read(directory/(prefix+("surface.csv" if donor else "surface-0.csv")))
        same(self.surface["i"], np.arange(20), 0, "label order")
        self.X = np.column_stack([self.surface[k] for k in ("x", "y")])
        shards = sorted(directory.glob(prefix+"rank*.csv"))
        self.owners = {}
        for rank, path in enumerate(shards):
            for cell in set(read(path)["cell"]):
                require(cell not in self.owners, "duplicate cell owner")
                self.owners[cell] = rank
        self.rows = np.sort(np.concatenate([read(path) for path in shards]), order=("cell", "q"))
        require(len(self.owners) == 192 and len(self.rows) == 192*9, "incomplete grid")
        same(self.rows["q"], np.tile(np.arange(9),192), 0, "quadrature identity")
        z = self.rows.reshape(-1,9)
        self.centers = np.stack([z[k].mean(axis=1) for k in ("x","y")], axis=-1)
        ij = np.floor((self.centers-origin)/h).astype(int)
        self.grid = np.empty((24,8,3,3,len(keys)))
        self.grid[ij[:,0],ij[:,1]] = np.stack([z[k] for k in keys], axis=-1).reshape(-1,3,3,len(keys))
        self.owner_grid = np.empty((24,8), dtype=int)
        self.owner_grid[ij[:,0],ij[:,1]] = [self.owners[c] for c in z["cell"][:,0]]

    def at(self, x):
        local = (x-origin)/h
        ij = np.clip(np.floor(local).astype(int), 0, [23,7])
        bx, by = (basis(local[:,d]-ij[:,d]) for d in (0,1))
        return np.einsum("ni,nj,njik->nk", bx, by, self.grid[ij[:,0],ij[:,1]])

    def quadrature(self, order):
        g, w = gauss(order)
        pts = np.stack(np.meshgrid(g,g), axis=-1).reshape(-1,2)
        return ((self.centers[:,None,:]+h*(pts-.5)).reshape(-1,2),
                np.tile(h*h*np.outer(w,w).ravel(),192))

    def transfer(self, points, phi):
        """Nearest admitted segment, increasing material ID at equal distance."""
        n = len(points)
        dist = np.full(n, np.inf)
        ids = np.zeros(n, dtype=int)
        xi = np.zeros(n)
        normal = np.zeros((n,2))
        for j, edge in enumerate(np.diff(self.X, axis=0)):
            t = edge/np.linalg.norm(edge)
            perp = np.array([-t[1],t[0]])
            offset = points-self.X[j]
            u = (offset@edge)/(edge@edge)
            d = abs(offset@perp)
            radius = (1-u)*self.surface["width_m"][j]+u*self.surface["width_m"][j+1]
            take = (u >= -1e-14) & (u <= 1+1e-14) & (d <= radius) & (d < dist)
            ids[take], xi[take], normal[take], dist[take] = j, u[take], perp, d[take]
        def interpolate(values):
            return (1-xi)*values[ids]+xi*values[ids+1]
        chi = np.where(np.isfinite(dist), 50*phi*(1+phi)/(1-phi)**2/interpolate(self.surface["I_m"]), 0.)
        return ids, xi, chi, normal

    def normal_mass(self, order):
        """Integrate both rays in physical metres, split at every Cartesian face."""
        g, w = gauss(order)
        g3, w3 = gauss(3)
        stations = ((np.arange(8)[:,None]+g3)/8).ravel()
        weights = np.tile(w3/8,8)
        M = np.zeros((20,20))
        rhs = np.zeros(20)
        for j, edge in enumerate(np.diff(self.X, axis=0)):
            length = np.linalg.norm(edge)
            n = np.array([-edge[1],edge[0]])/length
            centers = self.X[j]+stations[:,None]*edge
            xs, ws = [], []
            sizes = []
            for c in centers:
                cuts = np.sort(np.concatenate([(origin[d]+h*np.arange([25,9][d])-c[d])/n[d] for d in (0,1)]))
                starts = cuts[:-1]
                widths = np.diff(cuts)
                mid = c+(starts+widths/2)[:,None]*n
                inside = np.all((mid>origin)&(mid<-origin),axis=1)
                u = starts[inside,None]+widths[inside,None]*g
                xs.append(c+u.ravel()[:,None]*n)
                ws.append((widths[inside,None]*w).ravel())
                sizes.append(u.size)
            phi = self.at(np.concatenate(xs))[:,0]
            samples = 50*phi*(1+phi)/(1-phi)**2*np.concatenate(ws)
            integral = np.add.reduceat(samples,np.r_[0,np.cumsum(sizes)[:-1]])
            N = np.stack([1-stations,stations],axis=1)
            rhs[j:j+2] += (length*weights*integral)@N
            M[j:j+2,j:j+2] += N.T@((length*weights)[:,None]*N)
        return np.linalg.solve(M,rhs)

    def observables(self, parent, seed):
        x, w = self.quadrature(8)
        a = self.at(x)
        phi, xx, yy, xy, pressure = a[:,0],a[:,3],a[:,4],a[:,5],a[:,11]
        ids,u,chi,n = self.transfer(x,phi)
        t = np.stack([n[:,1],-n[:,0]],axis=1)
        shear = xx*t[:,0]*n[:,0] + yy*t[:,1]*n[:,1] + xy*(t[:,0]*n[:,1]+t[:,1]*n[:,0])
        compression = 5.+pressure-xx*n[:,0]**2-yy*n[:,1]**2-2*xy*n[:,0]*n[:,1]
        N = np.zeros((len(x),20))
        N[np.arange(len(x)),ids] = 1-u
        N[np.arange(len(x)),ids+1] = u
        mass = N.T@((w*chi)[:,None]*N)
        load = N.T@(w*chi*shear)
        v = self.surface["V_m_s"]
        speed = N@v
        theta = N@parent.surface["Theta_s"]
        background = N@(3.7+.15*np.cos(3*seed.X[:,0]))
        mu = .025*np.arcsinh(speed/.002*np.exp((.6+.015*np.log(theta*.001/.008))/.025))
        residual = N.T@(w*chi*(background+shear-compression*mu-speed))
        row_scale = N.T@(w*chi*abs(background))
        section_defect = np.max(abs(residual)/row_scale)
        require(section_defect<1e-8,"pressure-inclusive fixed-age section residual")
        work = load@v
        adjoint = same(work, w@(chi*(N@v)*shear), 2e-14, "signed work adjoint")
        eta = 5.5+4.5*np.tanh(x[:,1]/.35)
        energy = w@(xx*xx+yy*yy+(xx+yy)**2+2*xy*xy)/40
        viscous = w@((xx*xx+yy*yy+(xx+yy)**2+2*xy*xy)/(2*eta))
        X = self.X
        ds = np.linalg.norm(np.diff(X,axis=0),axis=1)
        mass_phi = w@phi
        # DGQ2 extrema sampled at equally spaced points on each one-sided cell.
        nodes = np.linspace(0,1,9)
        bn = basis(nodes)
        samples = np.einsum("ai,bj,xyji->xyab",bn,bn,self.grid[...,0])
        # Sample the physical ridge maximum at x=.075, away from a DG face.
        yline=np.linspace(-.6,.6,2401)
        transect=self.at(np.column_stack([np.full_like(yline,.075),yline]))[:,0]
        return dict(section_defect=section_defect,ridge_y=yline[np.argmax(transect)],mass=mass_phi, mean=mass_phi/48, extrema=np.array([samples.min(),samples.max()]),
                    centroid=(w*phi)@x/mass_phi, length=ds.sum(), tangent=np.diff(X,axis=0)/ds[:,None],
                    support=int(np.count_nonzero(chi>1e-10)), integrated_chi=w@chi,
                    work_shear=np.linalg.solve(mass,load), normal=np.linalg.solve(mass,N.T@(w*chi*compression)),
                    work=work, adjoint=adjoint, stored_energy=energy, maxwell_dissipation=viscous)

    def label_step(self, old, dt):
        a = old.at(old.X)
        same(self.X, old.X+dt*a[:,1:3], 3e-13, "material mapping")
        eta = 5.5+4.5*np.tanh(self.X[:,1]/.35)
        same(self.surface["eta"], eta, 1e-14, "layer at arrival")
        # The midpoint derivative is the secant to O(dy²). A bound on the
        # third derivative of the specified tanh law gives the absolute budget.
        dy = self.X[:,1]-old.X[:,1]
        mid = (self.X[:,1]+old.X[:,1])/2
        change = self.surface["eta"]-old.surface["eta"]
        remainder = 9./(.35**3)*abs(dy)**3/24+5e-15
        require(np.all(abs(change-(4.5/.35)/np.cosh(mid/.35)**2*dy)<=remainder),"Eulerian layer slope")
        v = self.surface["V_m_s"]
        same(self.surface["Theta_s"], old.surface["Theta_s"]*np.exp(-dt*v/.008)-.008/v*np.expm1(-dt*v/.008),
             3e-14,"material age")
        same(self.surface["Z_Pa_m"], np.exp(-10*dt/eta)*old.surface["Z_Pa_m"]-eta*np.expm1(-10*dt/eta)*v,
             3e-15,"material memory")
        same(self.surface["slip"],old.surface["slip"]+dt*v,2e-15,"material accumulated slip")

    def weak(self, old, donor, directory, step):
        """Q3 RHS from installed assembler; Q8 independent complete-stress virtual work."""
        cell, actual = (directory/f"load-{step}.txt").read_text().split()
        a = self.rows[self.rows["cell"] == cell]
        w = h*h*np.outer(gauss(3)[1],gauss(3)[1]).ravel()
        ex, shear = a["y"],1.4+a["x"]
        readback = -w@(a["offset_xx"]*ex+a["offset_xy"]*shear)
        same(float(actual),readback,1e-8,"actual last Picard RHS")
        k = -a["eta"]*np.expm1(-.125/a["eta"])
        implicit = w@(2*k*((a["Lxx"]-(a["Lxx"]+a["Lyy"])/3)*ex+.5*(a["Lxy"]+a["Lyx"])*shear))
        g, wg = gauss(8)
        x = h*np.stack(np.meshgrid(g,g),axis=-1).reshape(-1,2)
        w8 = h*h*np.outer(wg,wg).ravel()
        tau = donor.at(x)[:,3:6]
        expected = -w8@(tau[:,0]*x[:,1]+tau[:,2]*(1.4+x[:,0]))
        full = float(actual)-implicit
        same(full,expected,1e-8,"donor Q8 physical weak action")
        # Change only the receiving geometry and normal integrals in the rate return.
        phi = self.at(x)[:,0]
        k8 = -(5.5+4.5*np.tanh(x[:,1]/.35))*np.expm1(-.125/(5.5+4.5*np.tanh(x[:,1]/.35)))
        def slip_action(geometry):
            j,u,chi,n = geometry.transfer(x,phi)
            v = self.surface["V_m_s"]
            speed = (1-u)*v[j]+u*v[j+1]
            tn = n[:,0]*n[:,1]*x[:,1]+.5*(n[:,1]**2-n[:,0]**2)*(1.4+x[:,0])
            return w8@(2*k8*chi*speed*tn)
        slip = slip_action(self)
        stale = slip_action(old)
        require(abs(stale-slip)>1e-6,"insensitive stale support test")
        if step == 1:  # accepted DG and exterior stress are exactly zero
            v = self.at(x)
            dxx = v[:,6]-(v[:,6]+v[:,9])/3
            dxy = .5*(v[:,7]+v[:,8])
            raw = slip-w8@(2*k8*(dxx*x[:,1]+dxy*(1.4+x[:,0])))
            same(full, raw, 1e-8,"zero-parent Q8 constitutive load")
        print(f"step {step} actual RHS {float(actual):.12g}, full virtual work {full:.12g}, donor {expected:.12g}, "
              f"fresh/stale slip work {slip:.12g}/{stale:.12g}")


def endpoint(target, reference, weak=True):
    old = Frame(target,0)
    donor_old = Frame(reference,0,True)
    same(old.X,donor_old.X,3e-15,"seed coordinates")
    same(old.surface["Theta_s"],10+2*np.sin(2*old.X[:,0]),3e-14,"initial age")
    same(old.surface["slip"],0.,0.,"initial slip")
    # Absolute budgets in the units of each column. Iterative errors propagate
    # through dt, the spatial phase gradient and the fixed-age endpoint law.
    budget = dict(x=2e-10,y=2e-10,I_m=1e-8,Theta_s=2e-7,Z_Pa_m=2e-9,V_m_s=1e-8,width_m=2e-8,eta=1e-8)
    bulk = dict(phi=3e-10,vx=1e-8,vy=1e-8,tauxx=1e-8,tauyy=1e-8,tauxy=1e-8,pressure=1e-8)
    seed = old
    donor_seed = donor_old
    slip = np.zeros(20)
    for step in (1,2):
        a,b = Frame(target,step),Frame(reference,step,True)
        require(np.array_equal(a.rows["cell"],b.rows["cell"]), "different physical cell identities")
        a.label_step(old,.0125)
        slip += .0125*b.surface["V_m_s"]
        same(a.surface["slip"],slip,3e-10,"donor accumulated slip")
        x,w = a.quadrature(8)
        j,u,chi,_ = a.transfer(x,a.at(x)[:,0])
        jb,ub,chib,_ = b.transfer(x,b.at(x)[:,0])
        admitted = chi>1e-10
        require(np.array_equal(admitted,chib>1e-10) and np.array_equal(j[admitted],jb[admitted]),"Q8 support identity")
        same(u[admitted],ub[admitted],1e-7,"in-segment coordinate")
        points=np.column_stack([a.rows[k] for k in ("x","y")])
        _,_,chi3,_=a.transfer(points,a.rows["phi"])
        same(a.rows["chi"],chi3,2e-13,"production transfer at physical Q3 points")
        surface = {k:same(a.surface[k],b.surface[k],tol,k) for k,tol in budget.items()}
        fields = {k:same(a.rows[k],b.rows[k],tol,k) for k,tol in bulk.items()}
        same(a.rows["eta"],5.5+4.5*np.tanh(a.rows["y"]/.35),1e-14,"both sides of Eulerian layer")
        ray = a.normal_mass(24)
        same(ray,a.normal_mass(16),5e-12,"independent normal integration convergence")
        normal_error = same(ray,a.surface["I_m"],5e-12,"cell-split Q1 ray projection")
        aa,bb = a.observables(old,seed),b.observables(donor_old,donor_seed)
        obs = {k:same(v,bb[k],{"support":0,"tangent":1e-8,"work_shear":3e-8,"normal":3e-8}.get(k,1e-8),k)
               for k,v in aa.items()}
        print("step",step,"max label defects",surface,"max physical field defects",fields)
        print("phase mass/mean/extrema/centroid, length, active Q8, integrated chi, energy, dissipation, shear work, adjoint",
              *(aa[k] for k in ("mass","mean","extrema","centroid","length","support","integrated_chi","stored_energy","maxwell_dissipation","work","adjoint")))
        print("observable differences",obs,"independent Q1 ray defect",normal_error)
        if weak:
            a.weak(old,b,target,step)
        old,donor_old = a,b


def lifecycle(root):
    principal=root/"principal/output"
    phase_shards = [Frame(root/"mpi/output",i) for i in range(3)]
    for case in ("retry","restart","mpi"):
        out=root/case/"output"
        clock=read(out/"layered-background.csv")
        same(clock["step"],[0,1,2],0,"single publication")
        same(clock["time"],[0,.0125,.025],0,"accepted clock")
        same(clock["parent_time"],[0,0,.0125],0,"boundary parent")
        if case=="retry":
            same(clock["retries"],[0,1,1],0,"discarded boundary attempt")
            require((out.parent/"log").read_text().count("Repeating the current time step")==1,"wrong repeat count")
        defects=[]
        for step in range(3):
            a,b=Frame(out,step),Frame(principal,step)
            tol=1e-10 if case=="mpi" else 0.
            require(np.array_equal(a.rows["cell"],b.rows["cell"]),"cell ownership")
            defects.append(max(same(a.rows[k],b.rows[k],tol,k) for k in a.rows.dtype.names if k!="cell"))
            for k in a.surface.dtype.names:
                defects.append(same(a.surface[k],b.surface[k],tol,k))
                if case=="mpi":
                    same(a.surface[k],read(out/f"material-{step}-surface-1.csv")[k],0.,"replicated labels")
            require((out/f"background-{step}.csv").read_bytes()==(principal/f"background-{step}.csv").read_bytes(),"boundary checkpoint/rollback")
        print(case,"max endpoint difference",max(defects))
    a,b=phase_shards[0],phase_shards[2]
    owner=lambda q: q.owner_grid[tuple(np.floor((q.X-origin)/h).astype(int).T)]
    cells=lambda q: np.floor((q.X-origin)/h).astype(int)
    crossed=np.where((cells(a)!=cells(b)).any(axis=1))[0]
    migrated=np.where(owner(a)!=owner(b))[0]
    active=[]
    for i,j in np.ndindex(24,8):
        for axis,adj in [(0,(i+1,j)),(1,(i,j+1))]:
            if adj[0]>=24 or adj[1]>=8 or a.owner_grid[adj]==a.owner_grid[i,j]: continue
            x=origin+h*(np.array([i,j])+.5)
            x[axis]+=h/2
            face=a.at(x[None,:])[0]
            if abs(face[0]*face[1+axis])>1e-7: active.append((i,j,axis,face[0]*face[1+axis]))
    require(len(crossed)>0 and active,"no resolved crossing or active MPI partition face")
    print("cell crossings",crossed,"owner crossings",migrated,"active partition phi*u at face centers",active)
    cross=[Frame(root/"crossing-mpi/output",s) for s in range(3)]
    require(owner(cross[0])[9]!=owner(cross[1])[9],"material vertex failed to change supporting MPI rank")
    for s,q in enumerate(cross):
        serial=Frame(root/"crossing-serial/output",s)
        for k in q.surface.dtype.names:
            same(q.surface[k],serial.surface[k],1e-8,"partition crossing history "+k)
            same(q.surface[k],read(root/"crossing-mpi/output"/f"material-{s}-surface-1.csv")[k],0.,"replicated history")
        for k in keys:
            same(q.rows[k],serial.rows[k],1e-8,"partition crossing bulk "+k)
        if s: q.label_step(cross[s-1],.0125)
        print("migrating label 9",s,"owner",owner(q)[9],"record",q.surface[9])
    # The archive spans full in-box normals. This additional compact case has
    # genuinely interior geometric support, including at the outermost labels.
    for s in range(3):
        q=Frame(root/"interior/output",s)
        edges=np.diff(q.X,axis=0)
        normals=edges[:,::-1]*[-1,1]/np.linalg.norm(edges,axis=1)[:,None]
        tips=np.concatenate([q.X[j:j+19]+sign*normals*q.surface["width_m"][j:j+19,None]
                             for j in (0,1) for sign in (-1,1)])
        clearance=min(np.min(tips-origin),np.min(-origin-tips))
        require(clearance>0.,"finite-width strip met a box face")
        same(q.surface["V_m_s"],Frame(principal,s).surface["V_m_s"],2e-6,"finite-width reference rates")
        print("finite strip",s,"box clearance",clearance)
    for case in ("constant","zero","reverse"):
        q=Frame(root/case/"output",2)
        print(case,"translation of centroid of labels",np.mean(q.X-a.X,axis=0),"eta range",np.ptp(q.surface["eta"]))
        if case=="constant": same(q.surface["eta"],10.,0.,"constant viscosity")
    for step in (0,1):
        a,b=[Frame(root/name/"output",step) for name in ("constant","constant-base")]
        for k in a.rows.dtype.names:
            if k!="cell": same(a.rows[k],b.rows[k],2e-13,"moving base Maxwell "+k)
        for k in a.surface.dtype.names: same(a.surface[k],b.surface[k],2e-13,"moving base Maxwell "+k)
    for f in ("fault-checks","fault-0-0.txt","fault-1-0.txt","fault-2-0.txt"):
        p=(root/"homogeneous-base/output"/f)
        require(p.is_file(),f"missing prior homogeneous check {f}")
        require(p.read_bytes()==(root/"homogeneous-wrapper/output"/f).read_bytes(),f"homogeneous wrapper {f}")


def plot(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(1,3,figsize=(11,3.9),sharex=True,sharey=True,constrained_layout=True)
    x,y=np.meshgrid(np.linspace(-1.15,1.15,370),np.linspace(-.85,.85,275))
    for step,ax in enumerate(axes):
        f=Frame(out,step)
        phi=f.at(np.column_stack([x.ravel(),y.ravel()]))[:,0].reshape(x.shape)
        image=ax.contourf(x,y,phi,levels=np.linspace(0,.35,15),cmap="Blues",extend="max")
        ax.axhspan(-.35,.35,color="#ef9d42",alpha=.11)
        ax.axhline(0,color="#aa6622",lw=1,ls="--")
        for face in np.arange(-1,1.1,.5):
            ax.axvline(face,color=".65",lw=.45)
            ax.axhline(face,color=".65",lw=.45)
        ax.plot(*f.X.T,".-",color="#a91f42",lw=1.2,ms=4)
        for j in [3,9,15]:
            ax.annotate(str(j),f.X[j],xytext=(2,5),textcoords="offset points",fontsize=8)
        ax.set(title=f"{12.5*step:g} ms",xlabel="x (m)",aspect="equal")
    axes[0].set_ylabel("y (m)")
    fig.colorbar(image,ax=axes,label="FE phase")
    fig.suptitle("Material labels on the transported ridge · dashed line at the viscosity midpoint")
    fig.savefig(path,dpi=180)
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("target","donor","figure"): p.add_argument(arg,type=Path)
    for case in ("zero","reverse"): p.add_argument("--"+case,type=Path)
    args=p.parse_args()
    for case,donor in [("principal",args.donor),("zero",args.zero),("reverse",args.reverse)]:
        if donor:
            print(case,flush=True)
            endpoint(args.target/case/"output",donor)
    lifecycle(args.target)
    plot(args.target/"principal/output",args.figure)
