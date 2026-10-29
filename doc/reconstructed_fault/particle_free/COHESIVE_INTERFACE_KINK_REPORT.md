# Non-collinear admission and transfer control

**A prescribed 26.565° append passes unloaded and nonzero loaded tied transfer.**
The native vector-displacement/pressure representation, Gauss histories and
physical storage are preserved. An append that needs damage is rejected before
publication. This is a bounded component qualification, not selected-direction
propagation or a cure for the earlier early-growth spatial failure.

[Formulation and gates](COHESIVE_INTERFACE_KINK_FORMULATION.md) ·
[figures/data/reproduction](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_kink/INDEX.md).
Branch `pf-rsf-particle-free`, revision `f1f9f29cf40c32d1ddcea000b724375b64853532`
with existing local work retained. Production sources, BP3 and calibration are
unchanged.

## What was actually exercised

The old segment is (0.25,0)→(0.5,0) m; the append ends at (0.75,0.125) m in a
1 m square confined block. A fixed fitted native Q2/P1disc mixed mesh represents
both components of the displacement discontinuity and independent pressure
traces. This reference discretization differs from the earlier Q2/Q1 fixture;
there is no claim that its results are a new spatial reference for that specimen.
The pressure space is fixed before/after insertion. Existing material tangents,
nonuniform damage, bond reference, age, signed/absolute slip and full Maxwell
memory remain unchanged. New records are exactly tied, with zero damage/storage
and the accepted virgin age. No physical time advances during insertion.

The independent harness solves the actual native assembled mixed operator and
checks the unchanged C++ cohesive–RSF return. A separate native adapter imports
that verified state and exercises real `ReconstructedFaultManager` geometry/
history swaps and ASPECT checkpoint/restart. This is **not** a production
outer-solver topology transaction. Bulk coefficient, pressure and trace-layout
payloads participate in the same checkpoint. Native manager swaps invalidate
associations; this test does not claim to rebuild populated production caches.

| Check | Measured result |
|---|---:|
| Loaded physical stress, old histories, bulk/bond/fracture storage across insertion | exactly unchanged |
| Old peak geometric jump, 32×16 | 81.806 μm |
| Loaded weak tied certificate, 32×16 | 1033.288 Pa required; residual 1.53e−13 Pa |
| Minimum loaded sided normal pressure | 9520.787 Pa |
| Unloaded insertion | zero stress, slip and storage |
| Native geometry publication | exactly one; zero physical steps |
| Rejection/retry, preinstalled control, checkpoint continuation | bitwise identity |
| Complete pressure-coupled tangent, 32×16 | 4.15e−10 relative FD error; tolerance 1e−6 |
| Independent homogeneous affine patch | 1.82e−12 Pa maximum error |

The loaded snapshot is synthetic, mechanically equilibrated and below its old
damage obstacle. Its nonuniform old bond references and ages are preserved;
it is not asserted to be the outcome of the preceding straight-growth trajectory.
The fine old inventory is bulk .4899858563, bond .0018424027 and fracture
.1159154943 J/m. Every inventory is unchanged at tied insertion. The Gauss3
physical bulk energy differs from independent Gauss6 integration by 2.37e−10
relative for this state. A new-branch virtual slip variation gives zero normal
gap, a 56.81 Pa nonzero pressure response and positive 6.54e−5 J/m storage;
its Gauss6 energy/work difference is 3.59e−8 relative. It tests actual oblique
mechanical modes, rather than only a geometry drawing or scalar coordinate rename.

A genuinely overloaded new direction is isolated with trace-free axial loading:
tau_xx=5000 Pa, tau_yy=−5000 Pa and tau_xy=0. The existing horizontal plane has
weak shear below 4e−13 Pa. The oblique append requires **4000 Pa**, exceeding the
unchanged 3000 Pa strength; contact remains compressed at 13000 Pa. Both the
reference and native owner reject, restore the accepted state and publish zero
geometry changes. Damage admission is not simulated by resetting histories.

A subsequent pressure-coupled reference step uses the same accepted state before
and after restart/preinstallation. The full tangent includes the pressure term;
there is no pressure lag. Omitting the pressure term gives 1.58e−5 relative
FD error, above the retained 1e−6 tolerance, while the complete tangent passes.
The mass-scaled symmetric part is positive (minimum 5.46e7 Pa/m); this is a
local fixed-damage mechanical diagnostic, not a phase/free-path stability
certificate. On the fine mesh its work is .950482136 J/m, physical
sliding loss .000117608 J/m and Maxwell loss .0000504153 J/m. The BE bulk remainder
is .179859738 J/m: this deliberately large diagnostic loading increment is **not**
a physical-energy accuracy qualification. Ledger error is 3.33e−16 J/m, reported
separately from these numerical remainders. No new fracture energy is created.

## Resolution, direction and the retained straight-growth failure

One 16×8→32×16 mechanical comparison uses fresh sampling of the same prescribed
old histories, not transferred coarse quadrature states. A common physical
radius-.1 m weighted tensor average around the old endpoint changes by **.9345%**;
its principal axis changes **.09622°**. A Gauss6 assembly control changes the
coarse principal axis by only 1.3e−6°. These meet the declared 5%/5° diagnostic
gates. The averaged fine maximum-shear planes are **0.0153° and 90.0153°** from
+x. They do **not** select the prescribed 26.565° kink. The kernel is diagnostic,
not an altered strength or nonlocal constitutive law.

The local required tied multiplier changes from **811.525 to 1033.288 Pa**
(27.3% relative to the coarse value). The smooth average is therefore not a
qualification of pointwise tip stress or first activation. No precise onset time
or unique free direction is claimed, and this synthetic check does not qualify
the older first-onset trajectories.

The revised `comparison.png` retains the original straight-growth data, adds an
early inset and absolute/growth-relative differences, and ends every curve at
**2.42 s**. At 1.8 s the spatial growth difference remains **0.0055521 m / 65.4%**.
The process-zone a≥.05 front and continuous fully decohered front remain separate.
No law change or new convergence campaign was used to erase that failure.

## Path contract and limitations

Prospective tied geometry is distinct from fracture formation. One new edge
has immutable material/Gauss IDs and a declared orientation; old damaged normals
never rotate to maximize instantaneous driving. Zero-length, backwards/crossing
and unsupported multi-edge proposals reject. Virgin directions must eventually
satisfy maximum-shear intact admission, a stated preference only for genuinely
tied directions, compressive contact and the weak strength certificate. An
already overloaded candidate needs event-resolved or joint energy-accounted
physical admission; it cannot be a zero-work geometry update.

Surface a alone owns Gc∫a ds, using oblique arclength. Phi remains provisional
geometric metadata, with no second fracture reservoir or bulk degradation.
The exact-jump representation does not require diffuse normal-profile completion.

At a closed sharp corner, continuous vector traces and zero opening on two
nonparallel branches imply **zero vector jump at the corner**. The control
retains this consequence. It does not qualify freely sliding around a kink,
opening, junctions, secondary fracture, evolving paths, MPI, AMR, transport,
arbitrary DG-history import or changes to the bulk pressure approximation.

The next justified task is a bounded event-resolved admission test for a
prespecified oblique candidate inserted while still tied, followed by the
unchanged damage/contact solve. Any proposal to permit sliding through the
corner must first specify its normal-contact/kinematic treatment; that is a
physical-model decision, not a tolerance fix. No next experiment starts here.

## Implementation checks and cost

Native kernel costs were 9.72 s (16×8), 18.95 s (32×16), with a separate small
Gauss6 control and axial overload kernel; these include compact response exports.
The independent checks cost about 2.8 s. Five small native owner/checkpoint
invocations take approximately 1.5–3.5 s each. They were run sequentially.

A first owner restart comparison exposed implicit fresh shear-sense metadata
versus explicit +1 after deserialization. The fixture now sets +1 explicitly;
archives match bitwise without tolerance changes. A preliminary shear-overload
control was discarded because it also overloaded the old tied plane; the axial
control above corrects that test design. Both diagnoses are retained. Production
algorithms/defaults and previous failed scientific evidence were not changed.
