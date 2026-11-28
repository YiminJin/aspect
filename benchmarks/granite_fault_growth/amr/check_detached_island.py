"""Distinguish a coarse crossing on the backward seed ray from a connected extension."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import label

from trajectory import Case, bernstein


def check(folder, old_log, bins):
    case = Case(folder)
    pts, fields, _, _ = case.load(0)
    parent = case.phase_patches(pts,fields)
    pts, fields, _, ledger = case.load(1)
    candidate = case.phase_patches(pts,fields)
    assert ledger["time"].item() == .0025 and len(parent) == len(candidate)
    # N = 4k+2 places every coarse 4x4 sample exactly at a fine raster
    # bin center: h_coarse/8 = h_fine/4. Thus a point close to the .01
    # contour is evaluated at its own coordinates on both lattices.
    if bins < 10 or bins % 4 != 2:
        raise ValueError("Use 4k+2 bins per finest edge, e.g. 34 or 66.")
    spacing = min(np.min(size) for _, size, _ in candidate)/bins
    nx, ny = np.rint(case.size/spacing).astype(int)
    values = np.full((ny,nx),np.nan)
    crossing_points = []
    for (base, size, previous), (lo, width, patch) in zip(parent,candidate):
        np.testing.assert_allclose(lo, base, rtol=0, atol=1.e-13)
        np.testing.assert_allclose(width,size,rtol=0,atol=1.e-13)
        ix, iy = np.rint((lo-case.origin)/spacing).astype(int)
        cx, cy = np.rint(size/spacing).astype(int)
        bx, by = (bernstein((np.arange(c)+.5)/c) for c in (cx,cy))
        values[iy:iy+cy,ix:ix+cx] = by @ patch @ bx.T
        if max(size) > .25*(1+1.e-12):
            u = (np.arange(4)+.5)/4
            test = bernstein(u)
            old = test @ previous @ test.T
            new = test @ patch @ test.T
            jj,ii = np.nonzero((new>=.01)&(old<.01)&(new-old>=1.e-4))
            crossing_points.extend(lo + np.column_stack([u[ii],u[jj]])*size)
    assert np.isfinite(values).all() and crossing_points
    components,_ = label(values >= .01)
    def component(xy):
        i,j = np.floor((xy-case.origin)/spacing).astype(int)
        return components[j,i]
    seed = component(case.seed-case.t*case.ell)
    island = component(np.array([.25,.5]))
    assert island>0 and seed>0 and island!=seed
    for pt in crossing_points:
        delta=(pt-case.origin)/spacing-.5
        np.testing.assert_allclose(delta,np.rint(delta),rtol=0,atol=1.e-9)
        assert component(pt)==island
    assert np.all(np.asarray(crossing_points)[:,0] < 1.)
    # The former convention admitted every component along the backward
    # ray. Neither the old binary nor a second nonlinear solve is required
    # to discriminate that rule using this accepted field.
    ray_y = int(np.floor((case.seed[1]-case.origin[1])/spacing))
    end = int(np.floor((case.seed[0]-case.origin[0])/spacing))
    ray_components = set(components[ray_y,:end])-{0}
    legacy_crossings = sum(component(pt) in ray_components for pt in crossing_points)
    assert legacy_crossings == len(crossing_points)
    # The first solve is accepted without a mesh reaction; the former
    # whole-ray source reported one on this same trial.
    text = (folder/"log.txt").read_text()
    assert "Growing fault missed extension" not in text and "Repeating the current time step" not in text
    if old_log:
        assert "Growing fault missed extension" in old_log.read_text()
    return dict(cells=len(parent), bins_per_fine_edge=bins, threshold=.01, minimum_increment=1.e-4, coarse_crossings=len(crossing_points),
                leftmost_crossing_m=min(p[0] for p in crossing_points),rightmost_crossing_m=max(p[0] for p in crossing_points),
                seed_connected_crossings=0, whole_ray_crossings=legacy_crossings, accepted_s=ledger["time"].item())


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("--old-log",type=Path)
    p.add_argument("--bins",type=int,default=34)
    args = p.parse_args()
    print(json.dumps(check(args.output,args.old_log,args.bins),indent=2))
