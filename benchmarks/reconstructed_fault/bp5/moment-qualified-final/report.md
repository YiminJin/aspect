# Pressure compatibility correction and completed moment-cycle qualification

## Decision

**A, B and C pass four real steps; C also passes the two-rank replay.**
The former stop rejected numerical continuity noise before checking that the
base iterate was already converged. The correction retains the existing mixed
nonlinear target, independent surface target, fresh linear checks, FGMRES,
active-set/Armijo rules, physical pressure normalization and history lifecycle.
It changes the internal compatibility allowance and avoids unused directions.
No BP5 trajectory or general stress-transfer method was introduced.

## 1. Audit before correction

`../moment-pressure-audit-local/native_history_reference/run.log` reproduces the
original failure with observational output only. It checks

\[
 b_c=-R_b+B K_{FF}^{-1}R_\Gamma,\qquad q_L^T b_c,
\]

after homogeneous Stokes constraints, inside each Newton/active-set direction
solve and before FGMRES. In this fixture all surface rates are prescribed,
so the condensed correction is zero. The pressure block is the scaled
continuity equation; physical pressure is `p = pressure_scaling * p_solver`.
Here pressure_scaling = 40000000.000005908. Physical base constraints are lifted
once before residual evaluation; directions have homogeneous constraints.

At B step 2 / Newton 0:

| Quantity | Value |
|---|---:|
| Signed incompatible component | 1.60474642096447e-14 |
| Condensed RHS / fresh bulk norm | 1.32716371513922e-11 |
| Velocity residual norm | 1.32663433674344e-11 |
| Scaled continuity residual norm | 3.74815068457159e-13 |
| Zero-velocity reference norm | 58.4771207110079 |
| Old `100 epsilon * max(initial, reference, rhs)` allowance | 1.29845291654291e-12 |
| Old relative-only cap | 8.71377001866816e-15 |
| Existing fixed bulk precision allowance | 1.75295950964101e-10 |
| Existing mixed nonlinear target | 1.75304664734120e-10 |
| Free surface residual | 0 |

The **relative-only cap**, not the old roundoff estimate, caused rejection.
The bulk residual was already 13.2 times below the mixed target. The old ordering
also solved an unused direction at the preceding converged iteration; its
direction was never applied. This was not a failed mechanical linearization.

### Compatibility is established independently

- Prescribed velocities are `(0.01*y,0)` at top/bottom: their normal flux is zero.
- The x boundaries are periodic; their realized fluxes cancel. Native face
  integration reports **zero total flux** at every accepted state on one and
  two ranks, not merely an inference from the parameter file.
- The material is incompressible. Resolved parameters have additional Stokes RHS
  and prescribed dilation disabled; there is no mass source.
- The algebraic pressure null vector has constrained entries removed. Its
  physical FE representation is reconstructed with homogeneous constraints.
- Both actual constrained null identities are tested. At the audited state,
  `||C q_R|| = ||C^T q_L|| = 8.37210761906e-11`; their respective block scales
  are approximately 1.58455118855e7 (relative defect 5.28e-18).

The left identity is **not inferred from the right identity**. Since B has no
pressure rows, `q_L^T B=0`, hence `C^T q_L=A^T q_L`; the source explicitly tests
the transposed pressure-row blocks with constraints. The right test executes
the full condensed action, including G and the restricted surface inverse.
Their equality here follows from this fixture, not an assumed symmetry of
general fault coupling. An additional nonsymmetric algebraic test rejects an
RHS orthogonal to the right nullspace but not the left nullspace.

## 2. Narrow correction

The compatibility scale is evaluated before cancellation using actual native
FE gradients/JxW, the physical base velocity, pressure scaling, and the verified
left pressure mode distributed through homogeneous constraints:

\[
 S_p=\sum_{K,q}|s_pJxW|\Big(\sum_i|q_i^K\phi_i^p|\Big)
      \sum_{j,d}(|u_j^K|+|u_{origin,d}^K|)|\partial_d\phi^u_{j,d}|.
\]

The origin accounts for the componentwise constant subtraction in the actual
residual evaluation. The conservative operation count
`m=6*n_local_dofs+2*n_q+4*dim+16+n_global_cells` covers local coefficient
arithmetic, quadrature, constraints/cell accumulation and MPI summation. With
`gamma(m)=m*epsilon/(1-m*epsilon)`, the guard uses

\[
 \min\{\gamma_m S_p+
 \gamma_{2n_p}\|q\|_\infty\|b_p\|_1,
             \epsilon_{nl}s_b+\rho_b\}.
\]

For the captured step-2 state, S_p=657737.710251137, m=1492, final-dot scale
1.42535545644137e-13 and dot count=2210. The two roundoff contributions are
2.17902288145449e-7 and 6.99448401139551e-26. The unchanged mixed nonlinear
target caps the resulting allowance at **1.75304664734120e-10**. This is a
conservative bound, not an estimate of the typical measured noise and not a
coefficient fitted to the failed residual.

The old **relative-only compatibility cap** is explicitly replaced in both
authoritative documents. Neither nonlinear nor linear convergence targets
change. The guard validates the component before projecting a private RHS copy;
the accepted physical solution is not pressure-shifted by this operation.
The signed removed component is recorded as `rhs null` in each audit record.
For an already-converged base, the validated copy is unused: no extra solve or
state mutation occurs. When a direction is needed, every returned direction
still passes the existing fresh full-operator residual check and null-component
check within the original iteration budget.

An early check includes **all unprescribed surface rows**, so it cannot hide a
bound reaction by prematurely labeling nodes active. If it does not pass, the
original active-set algorithm runs. Its stabilized-free-set RMS floor is retained
when finalizing the surface normalization, preserving the original criterion.

`compatibility_audit.csv` exports both bound contributions, old relative cap,
mixed target, signed component, block norms, null defects/scales and skip status.
`qualification.json` additionally retains fresh linear residuals and face fluxes.

## 3. Four-step results

All runs start independently from the same initial data and retain dt=0.1 s.

| Real step | A jump (Pa m) | B jump (Pa m) | C jump (Pa m) | A/C reduction |
|---:|---:|---:|---:|---:|
| 1 | 0.0672938746323 | 1.77150e-14 | 1.02720e-11 | 6.551e9 |
| 2 | 0.116882455500 | 3.54300e-14 | 1.04687e-11 | 1.116e10 |
| 3 | 0.168485606323 | 4.50120e-14 | 1.05089e-11 | 1.603e10 |
| 4 | 0.218677546437 | 7.08600e-14 | 1.05347e-11 | 2.076e10 |

A reproduces the original accepted fields and transfer jumps. B's jumps remain
below its independently measured reverse-summation assembly floor at every
step. C passes both the original 1e4 reduction gate and
`max(1e-9 Pa m,1e-10 F_abs)` absolute gate at every step. Independent polynomial
weak-load integration confirms the native diagnostics, including both velocity
components, periodic constraints and signed projection/map cross terms.

C's mean-shear increments differ from B by at most **1.30e-13 relative**;
boundary reactions differ by at most **1.04e-14 relative**, versus the 1e-6 gate.
At step 4 the retained mean shear is 2000.00305074590 Pa in B and
2000.00305074584 Pa in C. A instead retains 2000.88265819697 Pa.

The resolved difference is decisive: C and B have **identical native velocity
and all four gradients** on one rank at all common states. At step 4 A differs
from B by 7.1330e-7 m/s maximum tangential velocity (weighted RMS 4.5584e-7 m/s).
No additional Newton update is needed for B/C at steps 2–4; their freshly
assembled bulk/surface residuals genuinely pass the unchanged criteria.

C's retained parent xy within-cell RMS stays at 3.77e-12–4.44e-12 Pa; A grows
from 1.54453 to 6.17999 Pa. B's **native tensor Frobenius RMS** grows from
3.10893 to 12.43570 Pa, while C's current tensor regenerates approximately
3.10893 Pa each solve. These are deliberately different statistical measures.
The step-4 B/C pointwise shear difference reaches 21.8859 Pa, but preserves
resolved loads: smoothing B's native tensor is not required for equilibrium.

## 4. MPI, lifecycle and tests

C on two ranks passes all four publication gates. At real steps 1–4,
one/two-rank velocity differences are at most **1.73473e-18 m/s** and current
tensor differences at most **6.95126e-11 Pa**. Initial particle count/hash is
identical in all branches/rank counts: `9216 3806466698461680840`.

Timestep zero's **evaluated**, uncommitted tensor differs by up to
7.55250e-5 Pa between ranks; velocity differs by 2.32683e-12 m/s. Both initial
nonlinear solves meet their original criteria and both retained tensors are
exactly zero. An initial analysis mistakenly applied the real-step pointwise
stress comparison to this uncommitted response; the final analysis separates
it explicitly rather than loosening a criterion or rerunning initialization.
The real-step discrepancy is six orders smaller in velocity and negligible
in stress as quantified above. No history is reset or committed twice.

Final build: `cmake --build build-pf-cpdi --target aspect -j4`.
`aspect-release --test 'Stage-I*'`: **20,136 assertions in 14 test cases passed**,
including roundoff acceptance, non-mutating rejection of deliberately
incompatible data, explicit nonsymmetric/nullspace checks and existing
active-set, line-search exhaustion and rollback helper tests. The three focused
Python test scripts pass **6 tests** total. No full ASPECT suite was run.

The pre-change audit failure is preserved. The first sandbox attempt could not
initialize MPI sockets and performed no mechanics; its log remains in
`../moment-pressure-audit/`. Intermediate successful validation is preserved in
`../moment-qualified-20260924/`; the results in this directory use the final
build consistently after the active-set scale-ordering review.

| Final run | Wall s | Reported child peak RSS MiB |
|---|---:|---:|
| A, one rank | 6.111 | 286.05 |
| B, one rank | 7.524 | 285.34 |
| C, one rank | 5.514 | 285.20 |
| C, two ranks | 5.596 | 253.03 |

RSS is the launcher's reported maximum child value, **not** summed MPI memory.
Each simulation had a 120-s hard cap and a new output directory.

## Files and reproduction

Production changes: `source/simulator/solver.cc` and
`include/aspect/simulator/solver/reconstructed_fault_linear.h`.
Tests: `unit_tests/reconstructed_fault.cc`.
Benchmark diagnostics: `../moment_cycle.cc`, `../run_moment_cycle.py`,
`../qualify_moment_cycle.py`. Existing complete mode PRMs are unchanged.
The authority pair records the guard revision; unrelated working changes remain.

Every final run uses ASPECT SHA256
`d053c7a596887b6e630574c8db660520957a890d672efa8a83ec22087392006c`
and plugin SHA256
`3fd10ed25696d27ad9b3582b91b5a4c560b95655b0d10f58845a2adba8708587`.
Per-run `execution.json`, copied source and PRM preserve exact provenance.
From repository root, the run pattern is:

```sh
python3 benchmarks/reconstructed_fault/bp5/run_moment_cycle.py MODE \
  --binary build-pf-cpdi/aspect-release --compatibility-audit --root NEW_DIRECTORY
```

Use each of the three existing mode names; C's MPI check adds `--ranks 2` only
after the serial gates pass. The runner refuses output overwrite. Offline:

```sh
python3 benchmarks/reconstructed_fault/bp5/qualify_moment_cycle.py \
  benchmarks/reconstructed_fault/bp5/moment-qualified-final
```

**Supported conclusion:** consistent history publication breaks the measured
forcing cycle in this horizontal, uniform-coefficient, prescribed-slip,
fixed-particle fixture while preserving physical loading. This does not select
a production transfer correction or qualify inclined faults, free RSF,
advection, mesh transfer, restart or long-time energy behavior. No further
experiment was launched.
