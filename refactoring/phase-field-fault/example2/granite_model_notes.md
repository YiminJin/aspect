# Example 2 · A growing laboratory fault in Westerly granite

The proposed specimen is a 1.5 m × 0.75 m block under in-plane shear. A horizontal seed enters from the left boundary and ends at (1.125, 0.375) m, leaving 0.375 m of intact rock ahead. Its last 0.125 m taper to zero phase. A single forward tip can turn within the specimen. The scientific target is two or more slip bursts under constant drive, with measurable new fault length during the bursts and arrest during reloading. First-event growth, stable sliding and complete failure on the first event are all possible outcomes.

The physical estimates, preload history and mesh restrictions below describe the original zero-stress pilot, retained in `granite_preload_reference.prm`. The current `granite_fault_growth.prm` initializes a uniform native deviatoric shear of 52,500,004.2031734 Pa and applies a steady 1 mm/s relative wall speed from time zero. Its companion `granite_prestress_hold.prm` holds all four walls fixed on a 48 × 24 grid for 60 µs.

## Inputs and provenance

| Quantity | Input | Basis |
|---|---:|---|
| Bulk shear modulus and density used for wave speed | 26.67 GPa, 2,650 kg/m³ | Passelègue et al., Table 2, report E = 64 GPa, ν = 0.20 and this density for Westerly granite. G = E/[2(1+ν)]. The growing solver imposes incompressibility, equivalent to ν = 0.5 for the elastic plane-strain stiffness calculation. |
| Effective friction pressure | 70 MPa | A normal stress tested by Kilgore et al. The constant adiabatic-pressure function supplies this value to the resistance law. Zero gravity and prescribed wall velocities make this a fixed-effective-pressure idealization. It does not simulate triaxial confining platens. |
| RSF direct and evolution effects | a = 0.0137, b = 0.0156 | Means measured together on bare Westerly granite at 70 MPa for 0.01–0.1 µm/s, Table 1 of Kilgore et al. The simulation extends that single pair across slip speeds. At 100–1,000 µm/s the paper also reports delayed strengthening after the initial weakening. |
| Characteristic slip distance | 1 µm | Lower end of the 1–10 µm range measured in the same study. This choice permits the smaller specimen. It is not a separate measurement for the particular 70 MPa velocity step. |
| Reference coefficient, speed and initial age | 0.75, 1 µm/s, 1 s | Representative bare-granite friction level, reference convention, and the corresponding steady-state age Dc/V₀. The seed starts held at zero bulk shear stress. |
| Radiation damping | 4.2032 MPa s/m | Calculated as G/(2cs) with cs = √(G/ρ) = 3.17 km/s. The branch uses the entered damping coefficient. The growth solver does not integrate seismic waves. |
| Intact shear fracture energy | 10 kJ/m² | Order of magnitude reported by Wong for Westerly granite in compression. This is an effective input for forming new surface. Partitioning experimental breakdown work between new damage and friction requires calibration. |
| Cohesive strength | 85 MPa | Aben and Brantut report a representative 85 MPa cohesion loss at 40 MPa effective pressure for their notched granite specimens. Its transfer to the model’s strength input at 70 MPa is a modeling assumption. |
| Maxwell viscosity | 3.17 × 10¹⁶ Pa s | Elastic-limit surrogate. The relaxation time η/G is about 1.19 × 10⁶ s against 0.35 s of loading. |
| AT1 length, seed centerline phase | 62.5 mm, 0.95 | Numerical regularization and an imposed mature seed. Neither is a measured granite damage-band width. Keep the AT1 length fixed in a mesh comparison. Vary it separately to assess regularization sensitivity. |

Friction and elasticity sources are [Kilgore, Blanpied and Dieterich (1993), Table 1](https://doi.org/10.1029/93GL00368) and [Passelègue et al. (2016), Table 2](https://doi.org/10.1002/2015JB012694). Fracture sources are [Wong (1982)](https://doi.org/10.1029/JB087iB02p00990) and [Aben and Brantut (2021)](https://doi.org/10.1016/j.epsl.2021.117174).

## How small can the specimen be?

The linearized spring-slider critical stiffness at low speed is (b−a)pₙ/Dc = 133 GPa/m. A corresponding leading-order correction from radiation damping is ηᵈV/Dc, or 4.2 GPa/m at 1 mm/s. The specimen has a finite seed and four kinematically controlled outer walls. To account for their stiffness, I independently solved a sharp-interface, incompressible elastic problem for three geometrically similar sizes.

| Width × height | Seed length | Lowest elastic stiffness against relative slip | Fraction of 133 GPa/m |
|---|---:|---:|---:|
| 2.00 × 1.00 m | 1.50 m | 62.5 GPa/m | 0.47 |
| **1.50 × 0.75 m** | **1.125 m** | **83.3 GPa/m** | **0.63** |
| 1.00 × 0.50 m | 0.75 m | 125 GPa/m | 0.94 |

For the calculation, all outer-wall displacement increments vanish. Mode-II reflection allows a solve on the upper half. Its centerline has zero incremental normal traction, zero tangential displacement ahead of the tip, and an applied tangential traction along the seed. Q2 displacement and Q1 pressure enforce incompressibility. The largest eigenvalue of the map from shear traction to the full slip jump gives the reciprocal of the softest stiffness. In the 2 m case, 48 × 12 and 96 × 24 elements on the half-domain gave 62.53 and 62.49 GPa/m. The other rows follow the inverse-length scaling of geometrically similar elastic bodies. This screens small perturbations about sliding. The nonlinear phase-field fault still has to develop an event from the selected initial state.

The distinction matters here. For an unbounded near-steady aging-law crack, L∞ = (2/π)G′Dc b/[pₙ(b−a)²] gives a **half-length** of about 2.10 m with G′ = 2G for the incompressible model. A different scale, Lb = G′Dc/(b pₙ), is only 48.8 mm. The 1.125 m seed passes the finite-body stiffness screen but does not accommodate the full near-steady limiting nucleation zone. The input tests whether instability develops before that limit is reached. A larger mesh at the same physical dimensions cannot cure a specimen that is too stiff or short. [Ampuero and Rubin (2008)](https://doi.org/10.1029/2007JB005082)

## Loading, cost and outputs

The native growing material starts without shear prestress. A smooth 25 ms setup adds 1.35 mm of relative wall displacement to the steady 1 mm/s drive. In an intact locked block, the extra displacement contributes 48.0 MPa of shear stress. Including the steady drive, the estimate at 25 ms is 48.9 MPa, compared with reference friction of 52.5 MPa. The setup spans about 53 shear-wave transit times across the box. The quasi-dynamic simulation uses this comparison only to characterize its loading protocol. Local stress concentration can cause slip or damage earlier. Inspect the end-of-setup state. Treat events during setup as forced responses and repeat with a lower setup displacement if it materially grows the seed. Setting Dload to zero provides a constant-drive control from time zero. A locked block reaches reference friction in about 1.48 s under that drive.

The current midpoint scalar solve enforces ηᵈ ≥ pₙ b (Δt/2)/Dc even at negligible slip speed. The full-step ceiling is 7.698 µs. With the selected 7.5 µs cap the 0.35 s window needs **at least 46,667 accepted steps**, plus shorter steps during acceleration. Growing mechanics factors the coupled matrix on MPI rank zero. AMR could reduce the work per solve. A separate change to the midpoint root solve and its admissibility test would be needed to use substantially larger locked-phase steps with these physical parameters.

For the historical pilot, place granite_fault_growth.prm, granite_preload_reference.prm and granite_startup.prm in the same directory and invoke these commands there with the current pf-rsf executable. The first runtime check covers eight nominal steps and writes the native diagnostics in a separate directory. Its elapsed time gives a lower-cost way to measure startup performance. The second input is the event-search case.

```sh
aspect --validate granite_fault_growth.prm
mpirun -np 4 aspect granite_startup.prm
mpirun -np 4 aspect granite_preload_reference.prm
```

The survey writes phase snapshots every 1 ms. For each **accepted** physical step, the controller log gives max|V| = slip_fraction × Dc / dt and the station index slip_id. Match retry records to the final accepted attempt before plotting. Use elapsed physical time and compare several seed-connected phase contours. For a short event replay, select the built-in `phase field fault history` postprocessor. Its sections tables carry signed midpoint rate, accumulated slip, station position and the formed flag. Its balance tables carry endpoint time and work. Use the nearest **previously formed** station behind the advancing tip to measure local slip acceleration. A formed flag can respond to a small positive phase, and station-by-station counting naturally makes a staircase. Verify an event using extension over several stations, a moving connected contour, unloading, and a return to low slip rate. A matched velocity-strengthening control can use a = 0.0175 with b = 0.0156, leaving the rest of the input fixed. Time convergence and mesh convergence are separate checks.

## Fault-following mesh refinement

The supplied calculation uses 96 × 48 Cartesian cells, 15.625 mm per side. Thus ℓ/h = 4 and Lb/h ≈ 3.1. This is a pilot resolution. The native path explicitly rejects initial adaptive passes, periodic remeshing and hanging faces. Its phase-face residual assumes a single active neighbor, its kinematic constraints assume simple prescribed values, and its material checkpoint keys histories to strip edges, cells, vertices and quadrature points. The branch constructs initial strip spacing from the smallest bulk-cell altitude. Selecting a composition indicator or loading an external marking plugin cannot satisfy these constraints. [Midpoint solver](https://github.com/YiminJin/aspect/blob/ee6d59cb7e59b871b176a1e7042577aff9b4edfc/source/simulator/solver/midpoint_fault.cc) · [Phase-face assembly](https://github.com/YiminJin/aspect/blob/ee6d59cb7e59b871b176a1e7042577aff9b4edfc/source/phase_field/midpoint_fracture.cc) · [Material histories](https://github.com/YiminJin/aspect/blob/ee6d59cb7e59b871b176a1e7042577aff9b4edfc/source/material_model/phase_field_fault/strip_material_history.cc)

My first implementation would use a permanently refined corridor wide enough to cover the prescribed seed and the region where its tip may grow. This isolates coarse–fine Stokes constraints and DG phase-face integration before testing history transfer on a moving mesh. Start with the same finest h as the uniform reference. Integrate the phase jumps and normal fluxes on each physical fine subface with the appropriate penalty and check their contribution to the energy ledger.

For dynamic adaptation, retain the fine mesh across the existing damage band and within a disk around the forward tip. An initial marking rule for this specimen is a band of half-width 2ℓ and a tip disk of radius 3ℓ. The current tip-direction calculation samples a disk of radius 2ℓ. The extra ℓ gives it fine cells in intact rock before it chooses a direction. Evaluate whether the gradient of along-fault slip rate requires further resolution on the formed fault. Keep the physical distance between fault stations independent of unrelated bulk refinement. Begin with refinement only and implement coarsening once transfer on refinement has been verified.

An illustrative balanced-quadtree count starts from 12 × 6 cells and uses three levels of refinement. Refine any cell intersecting the horizontal seed band or the tip disk to h = 15.625 mm and balance touching neighbors to 2:1, including corner neighbors. The initial mesh has **1,788 cells**, compared with **4,608** on the uniform grid. That is 61% fewer cells. The count excludes future tip motion and carries no wall-time estimate.

Remesh from an accepted state. Preserve the DG phase bounds and irreversibility, native stress and cohesive-strain history, formed geometry, rate-and-state age, and accumulated slip. Rebuild quadrature and transfer weights on the new mesh, then check the energy and frictional work before continuing. On refinement, the DG polynomial on a parent can be restricted exactly to its children. Material-point histories require an explicit transfer rule. If a trial burst leaves the fine region, retry that interval from its accepted parent on the enlarged fine mesh. Compare uniform and adaptive runs at the same local h and ℓ using event timing, signed tip-adjacent rate, extension per event, and fracture work. A mesh change alone should produce no new fault length or slip. [Heister, Wheeler and Wick (2015)](https://doi.org/10.1016/j.cma.2015.03.009)

The parameter paths and solver restrictions were checked against pf-rsf commit ee6d59cb7e59. The stiffness calculation and quadtree count are independent calculations. An ASPECT executable built from that branch was unavailable here. Startup, event occurrence and full-run wall time remain unverified.
