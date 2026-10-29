# Saved-state angular diagnosis (no native advancement)

Run from the repository root:

```sh
python3 benchmarks/reconstructed_fault/particle_free/bulk_cohesive/assess_angular.py benchmarks/reconstructed_fault/particle_free/evidence/bulk_angular
env MPLCONFIGDIR=/tmp/bulk-angular-mpl python3 benchmarks/reconstructed_fault/particle_free/bulk_cohesive/plot_angular.py benchmarks/reconstructed_fault/particle_free/evidence/bulk_angular
```

This reads the existing wide, fine-G8 and square-G6 trajectories. It does not
call ASPECT, evaluate a new return map or publish/modify a history. NumPy and
Matplotlib suffice. The exact profile recipe, equations, first-node table,
reference checks and interpretation are in
`doc/reconstructed_fault/particle_free/BULK_ANGULAR_REPORT.md`.

Each native `state-####.bin` is decoded as its original double vector:
all three DG coefficient fields (total count from fixture.csv), then the
numeric-CellId-ordered local a/k/L/q/rate records at native QPs, physical phase,
step/time/publication count. Active-cell iteration order and lexical CellId
order are not substituted for the native map comparator. The geometry is fixed;
the complete step-52 CSV supplies the same cell/QP points and positive JxW.
The code first reproduces its a/k/L/q columns exactly and checks stored phase
and clocks. It then recovers the previously unexported dense precursor samples.

At each original QP it computes Y from the saved q and that state's Q1 phase.
Native weak mechanical gradients are independently reproduced. Arc qbar and
Ybar are **separate** averages with the same positive normalized Q1 test as
the fracture/total residual densities. No averaged q or interpolated history
is fed back into a constitutive law. Arc samples include native grid crossings;
otherwise narrow residual minima at a first-growing node can be missed by a
uniform angular sample. Initial fracture values use the unchanged unloaded
Q1 seed, verified identical at step 1; initial mechanical driving is zero.

For the smooth reference, the analytical profile equation is integrated in the
regular cosine parameter. Weak fracture forces use cell-edge integration by
parts (Delta(Q1 basis)=0), splitting at support and band/cap interfaces. The
actual table and the differentiable ideal are distinguished throughout.

`../evidence/bulk_angular/diagnostic-inputs.tar.xz` contains the required original
phase/history/geometry files and read-only scripts at their repository-relative
paths. Extract it into an **empty directory** and run the commands there with a
fresh output directory to replay without touching the accepted original runs.
No build or binary executable is needed for that replay. The archive manifest
and checksums identify exactly which existing states were used. C++ source and
binary identities are retained for provenance; full raw runs remain in their
original build directories. Prior native checkpoints and physical-energy
failures are preserved, and no new orientation trajectory is selected here.
