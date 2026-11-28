"""Check a zero-duration native transfer, keyed by the old cell and material ID."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured as columns

from check_fracture_rows import Cell


TENSORS = ("xx", "yy", "zz", "xy", "Axx", "Ayy", "Azz", "Axy")
HISTORY = TENSORS + ("signed", "absolute")


def read_snapshot(path, prefix, ranks):
    cells = {}
    energy = np.zeros(3)
    metadata = ledger = None
    for rank in range(ranks):
        name = path/f"{prefix}-r{rank}"
        summary = name.with_suffix(".txt").read_text().splitlines()
        kind, *values = summary[0].split()
        assert kind == "energy"
        energy += np.array(values, float)
        data = name.with_suffix(".ledger")
        assert data.is_file(), "Use a nonzero-duration accepted parent with a work ledger."
        current_ledger = data.read_bytes()
        if metadata is None:
            metadata, ledger = summary[1:], current_ledger
        assert summary[1:] == metadata and current_ledger == ledger
        rows = np.atleast_1d(np.genfromtxt(name.with_suffix(".csv"), names=True, dtype=None, delimiter=",", encoding="utf8"))
        assert len(rows) and all(np.isfinite(rows[c]).all() for c in rows.dtype.names if c != "cell")
        for cid in np.unique(rows["cell"]):
            assert cid not in cells
            cells[cid] = rows[rows["cell"] == cid], rank
    assert np.isfinite(energy).all()
    return cells, energy, metadata, ledger


def geometry(rows, owner):
    row = rows[0]
    lo = np.array([row["x0"], row["y0"]])
    return Cell(str(row["cell"]), lo, lo+[row["hx"], row["hy"]], None, owner)


def phase_coefficients(rows, cell):
    xy = columns(rows[["x", "y"]])
    shape, _ = cell.shape(xy)
    coeff, _, rank, _ = np.linalg.lstsq(shape, columns(rows[["phase", "old_phase"]]), rcond=None)
    assert rank == 9
    return coeff


def assert_bounded(coeff, depth=4):
    if coeff.min() >= -2e-12 and coeff.max() <= 1.+2e-12:
        return
    assert depth > 0, (coeff.min(), coeff.max())
    left = np.array([[1,0,0],[.5,.5,0],[.25,.5,.25]])
    for tx in (left,left[::-1,::-1]):
        for ty in (left,left[::-1,::-1]):
            assert_bounded(ty@coeff@tx.T,depth-1)


def penalty_energy(cells, fields, ell, kf, penalty):
    """Physical face quadrature of 1/2 Kf sigma [phase]^2 only."""
    t, w = np.polynomial.legendre.leggauss(6)
    total = 0.
    for left in cells:
        for right in cells:
            for d in (0, 1):
                if abs(left.hi[d]-right.lo[d]) > 1e-13:
                    continue
                e = 1-d
                a, b = max(left.lo[e], right.lo[e]), min(left.hi[e], right.hi[e])
                if a >= b:
                    continue
                pts = np.empty((len(t), 2))
                pts[:, d], pts[:, e] = left.hi[d], (a+b)/2 + t*(b-a)/2
                jump = left.shape(pts)[0]@fields[left.name][:, 0] - right.shape(pts)[0]@fields[right.name][:, 0]
                sigma = 18*penalty*ell**2/min(left.size[d], right.size[d])
                total += .25*kf*sigma*(b-a)*(w@(jump*jump))
    return total


def check(path, step, pass_id, ranks):
    pre, e0, meta, ledger = read_snapshot(path, f"remesh-{step}-{pass_id}-before", ranks)
    post, e1, meta1, ledger1 = read_snapshot(path, f"remesh-{step}-{pass_id}-after", ranks)
    assert meta1 == meta and ledger1 == ledger and int(meta[1].split()[1]) == step
    _, ell, kf, penalty = meta[0].split()
    assert len(post) > len(pre)
    old_cells = {k: geometry(*v) for k,v in pre.items()}
    new_cells = {k: geometry(*v) for k,v in post.items()}
    old_phase = {k: phase_coefficients(pre[k][0], c) for k,c in old_cells.items()}
    new_phase = {k: phase_coefficients(post[k][0], c) for k,c in new_cells.items()}
    descendants = defaultdict(list)
    migrations = 0
    phase_error = np.zeros(2)
    for cid, child in new_cells.items():
        parent, = (c for c in old_cells.values() if np.all(c.lo <= child.lo+1e-13) and np.all(child.hi <= c.hi+1e-13))
        descendants[parent.name].append(post[cid][0])
        migrations += int(parent.owner != child.owner)
        if cid == parent.name:
            assert np.array_equal(pre[cid][0][list(HISTORY)], post[cid][0][list(HISTORY)]), cid
        rows = post[cid][0]
        actual = columns(rows[["phase", "old_phase"]])
        # Bernstein enclosures qualify bounds on the entire child, including
        # irreversible increase over the previous physical endpoint.
        coeff = new_phase[cid]
        for phase in coeff.T:
            assert_bounded(phase.reshape((3,3)))
        assert np.min(coeff[:,0]-coeff[:,1]) >= -2e-12
        expect = parent.shape(columns(rows[["x", "y"]]))[0]@old_phase[parent.name]
        phase_error = np.maximum(phase_error, abs(expect-actual).max(axis=0))

    moment_error = np.zeros(len(HISTORY))
    first_error = np.zeros(len(TENSORS))
    affine_error = np.zeros(len(TENSORS))
    signal = np.zeros(len(HISTORY))
    cellwise_constant_defect = 0.
    for cid, children in descendants.items():
        old = pre[cid][0]
        new = np.concatenate(children)
        c = old_cells[cid]
        for k in np.unique(old["strip"]):
            src, dst = old[old["strip"] == k], new[new["strip"] == k]
            assert len(dst)
            so, sn = src["area"], dst["area"]
            np.testing.assert_allclose(so.sum(), sn.sum(), rtol=5.e-12, atol=1e-18)
            vo, vn = columns(src[list(HISTORY)]), columns(dst[list(HISTORY)])
            mean = so@vo / so.sum()
            signal = np.maximum(signal, abs(vo).max(axis=0))
            moment_error = np.maximum(moment_error, abs(so@vo-sn@vn))
            def affine(rows):
                xy = (columns(rows[["x", "y"]])-c.lo)/c.size
                return np.c_[np.ones(len(rows)), xy]
            src_shape, dst_shape = affine(src), affine(dst)
            first_error = np.maximum(first_error, abs(src_shape.T@(so[:, None]*vo[:, :8])
                                                      -dst_shape.T@(sn[:, None]*vn[:, :8])).max(axis=0))
            if len(children) > 1:
                # Independent weighted QR/SVD, fitting all polynomial moments
                # to the old physical samples. Do not use the producer's slopes.
                coeff = np.linalg.lstsq(src_shape*np.sqrt(so[:, None]),
                                        vo[:, :4]*np.sqrt(so[:, None]), rcond=None)[0]
                phi = np.clip(src["phase"], 0., 1.)
                damaged = phi > 0.
                predicted_strain = np.zeros((len(dst),4))
                if damaged.any():
                    w = so[damaged]
                    f = phi[damaged]
                    coeff_strain = np.linalg.lstsq(src_shape[damaged]*np.sqrt(w*f)[:,None],
                                                   vo[damaged,4:8]*np.sqrt(w/f)[:,None],rcond=None)[0]
                    predicted_strain = np.clip(dst["phase"],0.,1.)[:,None]*(dst_shape@coeff_strain)
                predicted = np.column_stack((dst_shape@coeff,predicted_strain))
                affine_error = np.maximum(affine_error, abs(predicted-vn[:, :8]).max(axis=0))
                intact = vn[dst["phase"] <= 0.,4:]
                assert (intact == 0.).all()
                cellwise_constant_defect = max(cellwise_constant_defect, abs(vn[:, :4]-mean[:4]).max())
    # Absolute budgets scale with the measured field and area. Thin intersections
    # and physical/referential coordinate conversions enter the independent fits.
    assert np.all(phase_error < 2e-12), phase_error
    assert np.all(moment_error < 2e-12*np.maximum(signal, 1e-80)), (moment_error,signal)
    assert np.all(first_error < 2e-12*np.maximum(signal[:8], 1e-80)), (first_error,signal)
    assert np.all(affine_error < 3e-9*np.maximum(signal[:8], 1e-80)), (affine_error,signal)
    assert cellwise_constant_defect > max(signal[:4])*1.e-5
    assert signal[8] > 0. and signal[9] > 0.
    for rows, _ in post.values():
        assert np.all(rows["absolute"] >= abs(rows["signed"])-64*np.finfo(float).eps*max(signal[9],1e-80))
    delta_penalty = (penalty_energy(new_cells.values(),new_phase,float(ell),float(kf),float(penalty)) -
                     penalty_energy(old_cells.values(),old_phase,float(ell),float(kf),float(penalty)))
    np.testing.assert_allclose(e1[2]-e0[2],delta_penalty,rtol=2e-9,atol=4e-12*max(e0[2],1.))
    return dict(step=step,pass_id=pass_id,ranks=ranks,cells_before=len(pre),cells_after=len(post),
                cells_migrated=migrations,strip_count=sum(line.startswith("strip ") for line in meta),
                law_state_count=sum(line.startswith("state ") for line in meta),
                pressure_station_count=sum(line.startswith("pressure ") for line in meta),
                phase_defect=phase_error.tolist(),
                native_integral_defects=moment_error.tolist(),tensor_first_moment_defects=first_error.tolist(),
                tensor_affine_defects=affine_error.tolist(), constant_stress_error_Pa=cellwise_constant_defect,
                bulk_cohesive_fracture_before_J_m=e0.tolist(), bulk_cohesive_fracture_after_J_m=e1.tolist(),
                bulk_cohesive_fracture_change_J_m=(e1-e0).tolist(), independent_penalty_change_J_m=delta_penalty)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--pass-id", type=int, default=1)
    p.add_argument("--ranks", type=int, required=True)
    args = p.parse_args()
    print(json.dumps(check(args.output,args.step,args.pass_id,args.ranks),indent=2))
