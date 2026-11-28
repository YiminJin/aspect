"""Forward x on the seed-connected, two-dimensional DG excursion.

Sample inside each Cartesian Bernstein cell, including its own four traces.
Join local components only over positive-length overlaps of the two face
polynomials (also on hanging faces). Refine the leading x crossing along
sampled y rows using the quadratic itself. The sampling diagonal describes
interior resolution; a separate Bernstein envelope can certify a much
tighter x bound when no detached excursion lies farther forward.
"""
from collections import defaultdict
import numpy as np
from scipy.ndimage import label

from trajectory import bernstein


def intervals(row, level):
    """Positive-length superlevel intervals of a quadratic in Bernstein form."""
    left, mid, right = row - level
    a, b, c = left - 2*mid + right, 2*(mid-left), left
    roots = []
    if a:
        disc = b*b - 4*a*c
        if disc > 0:
            q = -.5*(b + np.copysign(np.sqrt(disc), b))
            roots = (q/a, c/q)
        elif disc == 0:
            roots = (-b/(2*a),)
    elif b:
        roots = (-c/b,)
    cuts = [0., *sorted(z for z in roots if 0 < z < 1), 1.]
    return [(lo,hi) for lo,hi in zip(cuts,cuts[1:])
            if (a*((lo+hi)/2)+b)*((lo+hi)/2)+c >= 0]


def seed_front(cells, origin, anchor, level, samples):
    """Return a connected witness, a global upper x bound, and sample spacing in m."""
    if samples < 2:
        raise ValueError("At least two sampling intervals per finest cell edge are required")
    spacing = np.min([width for _,width,_ in cells], axis=0)/samples
    parents = [0]
    def root(n):
        while n != parents[n]:
            parents[n] = parents[parents[n]]
            n = parents[n]
        return n

    # Using integer lattice keys avoids false gaps from roundoff when a cell
    # origin was recovered by native quadrature. FE values remain one-sided.
    sides = defaultdict(lambda: ([], []))
    blocks, anchors = [], []
    upper = -np.inf
    for lower, width, patch in cells:
        if np.max(patch) < level:
            continue
        offset = np.rint((lower-origin)/spacing).astype(int)
        count = np.rint(width/spacing).astype(int)
        np.testing.assert_allclose((lower-origin)/spacing, offset, rtol=0, atol=1e-8)
        np.testing.assert_allclose(width/spacing, count, rtol=0, atol=1e-8)
        base = origin+offset*spacing
        width = count*spacing
        u,v = [np.linspace(0,1,n+1) for n in count]
        samples_y = bernstein(v) @ patch
        field = samples_y @ bernstein(u).T
        local,n = label(field >= level)
        start = len(parents)-1
        groups = np.where(local, local+start, 0)
        parents.extend(range(start+1,start+n+1))
        blocks.append((base,width,samples_y,u,v,groups))
        if base[0]+width[0] >= upper:
            for row in patch:
                for _,b in intervals(row,level):
                    upper = max(upper, base[0]+b*width[0])
        pos = (anchor-base)/width
        if (np.all((pos > 1e-12) | (anchor == origin)) and np.all(pos >= -1e-12)
                and np.all(pos <= 1+1e-12)):
            i,j = np.rint(np.clip(pos,0.,1.)*count).astype(int)
            actual = (bernstein(pos[1:2]) @ patch @ bernstein(pos[:1]).T).item()
            # Deterministic lower-side trace, independent of the partition.
            anchors.append((tuple(base),groups[j,i],actual))
        for axis in (0,1):
            along = 1-axis
            for side in (0,1):
                end = -side
                row = patch[:,end] if axis==0 else patch[end,:]
                ids = groups[:,end] if axis==0 else groups[end,:]
                z = v if axis==0 else u
                for a,b in intervals(row,level):
                    kept = np.unique(ids[(z >= a)&(z <= b)])
                    kept = kept[kept!=0]
                    if len(kept)>1:
                        raise ValueError("A connected face interval has multiple sampled components")
                    if len(kept):
                        plane = (axis,offset[axis]+side*count[axis])
                        sides[plane][side].append((offset[along]+a*count[along],
                                                   offset[along]+b*count[along],kept[0]))
    for lower,upper_faces in sides.values():
        lower.sort()
        upper_faces.sort()
        j = 0
        for a,b,k in lower:
            while j<len(upper_faces) and upper_faces[j][1]<=a:
                j += 1
            for c,d,h in upper_faces[j:]:
                if c>=b:
                    break
                if min(b,d)-max(a,c)>1e-12:
                    parents[root(k)] = root(h)
    if not anchors:
        raise ValueError(f"No phase patch contains the original anchor {anchor}")
    _,seed,actual = min(anchors)
    if not seed or actual < level:
        raise ValueError(f"Original analysis trace falls below {level} at {anchor}")
    connected = np.array([bool(i) and root(i)==root(seed) for i in range(len(parents))])
    x,y = -np.inf,np.nan
    for base,width,rows,u,v,groups in sorted(blocks,key=lambda b: b[0][0]+b[1][0],reverse=True):
        if base[0]+width[0] < x:
            break
        for j in np.flatnonzero(np.any(connected[groups], axis=1)):
            for a,b in intervals(rows[j],level):
                if np.any(connected[groups[j,(u>=a)&(u<=b)]]):
                    end = base[0]+width[0]*b
                    if end>x:
                        x,y = end,base[1]+v[j]*width[1]
    if not np.isfinite([x,y,upper]).all() or upper < x-1e-12:
        raise ValueError("Inconsistent connected front and global Bernstein envelope")
    return x,y,upper,np.linalg.norm(spacing)
