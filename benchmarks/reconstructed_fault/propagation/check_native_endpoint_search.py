#!/usr/bin/env python3
"""Check an obstacle search prediction; physical energy/Hessian stay unchanged."""
import numpy as np
from scipy.linalg import solve
import native_endpoint as native


def predict(matrix, gradient, tolerance):
    active = gradient >= 0
    step = np.zeros_like(gradient)
    for iteration in range(1000):
        free = np.flatnonzero(~active)
        step[:] = 0
        if len(free):
            step[free] = solve(matrix[np.ix_(free, free)], -gradient[free], assume_a='pos')
        multiplier = gradient + matrix @ step
        changed = active.copy()
        changed[~active & (step < 0)] = True
        changed[active & (multiplier < -tolerance)] = False
        if np.array_equal(changed, active):
            assert min(step) >= 0
            assert max(abs(multiplier[~active]), default=0) < tolerance
            assert min(multiplier[active], default=0) >= -tolerance
            return step, iteration + 1
        active = changed
    raise RuntimeError('Search quadratic active set did not stabilize')


def main():
    e = native.E
    fields = np.load(e/'prefix-2-fine-1024-onset/fields.npz')
    times = fields['time']
    rows = native.old.read(e/'prefix-2-fine-1024-onset/history.json')
    p = native.SharedFE()
    matrix = p.r.F/native.old.A + np.diag(p.r.mass)
    results = []
    for time in [.52734375, .5283203125, .529296875, .53125]:
        j = int(np.flatnonzero(times == time)[0])
        lower = fields['phase'][j-1].copy()
        p.incoming(lower, fields['R'][j-1], fields['Z'][j-1],
                   rows[j-2]['slip'], time, time-times[j-1])
        value, gradient = p.evaluate(lower)
        step, iterations = predict(matrix, gradient, 2e-13)
        accepted = None
        for ls in range(35):
            trial = lower + 2.**(-ls)*step
            if max(trial) >= 1:
                continue
            v, _ = p.evaluate(trial)
            if v <= value + 1e-4*min(0., gradient@(trial-lower)) + 32*np.finfo(float).eps*max(1., abs(value)):
                accepted = trial
                break
        assert accepted is not None
        result = p.r.refine(accepted, lower, p.evaluate, p.hessian)
        defect = p.r.defect(p.evaluate(result)[1], result, lower)
        error = float(max(abs(result-fields['phase'][j])))
        assert defect <= 1e-11 and error < 1e-8
        results.append(dict(time=time, search_iterations=iterations,
                            line_rejections=ls, search_energy_change=v-value,
                            search_phase_increment=float(max(accepted-lower)),
                            corrected_KKT=float(defect), corrected_phase_error=error))
    native.old.write(e/'search-prediction-check.json', results)
    print(native.old.json.dumps(results))


if __name__ == '__main__':
    main()
