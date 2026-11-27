# Maintainer review of the phase-field fault branch after C2

**Request changes and keep this implementation.** The first repair is a user input that ASPECT accepts and then ignores. The most important numerical repair for BP3 is on the boundary-spanning fault. The outstanding strip-8 result also has a much narrower explanation now. These are three separately testable changes.

Reviewed remote tip: `14358d2a56c2542aace26e91adafbc69e1ee71e2`. C1: `5cd01f7f8d8011001b635724155861d1aff4422e`. ASPECT 3.1 comparison base: `0d97d2d3311154ca3f01d801b18f3945dc8b8bad`. Paths below are relative to the root of this worktree.

## 1. Changing the modulus in the advertised material can have no effect

**Fix before accepting the C2 interface.** Start with the installed growing seed, keep `Model name = phase field fault`, and add the following.

```text
subsection Material model
  subsection Phase field fault
    set Shear modulus = 2e6
  end
end
```

The two million pascals pass parameter validation. The coupled Maxwell response still uses the one million pascals inherited from the example's `Growing fault` subsection. The Simulator selects the growing implementation before material parsing. That implementation checks three entries in the public subsection, switches to the older subsection, and reads its modulus and resistance selector there. With the test law loaded, a user selecting the linear dashpot under the public subsection can likewise get rate-and-state friction. The installed interface test selects the growing dashpot under the older subsection, so its passing result is consistent with this defect. The guide documents the split, but a successful run with the wrong modulus is a poor failure mode even for a user who has read the guide.

Give shared physical settings one canonical address. Resolve legacy values and route-specific defaults once. Accept an unambiguous old spelling, accept equal duplicates, and diagnose conflicting explicit values with both paths. Entries that make sense only for one route should say so if supplied on the other. A compact material-factory helper for the two built-in routes is reasonable if it removes the two public material names from the general Simulator constructor. The two numerical implementations can keep their own state and initialization.

**Acceptance.** Vary the public modulus with the old entry absent and measure a physical stress change against the equivalent legacy input. Supply two different explicit moduli and require a diagnostic. Select the stateless law through the same public subsection for mature and growing runs and check both the force and the absence of age. Preserve historical defaults for each route.

In file order, see lines 228–232, 56–81, 285–289 and 321–329, 63, and 115–119.

```text
source/simulator/core.cc
source/material_model/growing_fault.cc
source/material_model/phase_field_fault.cc
benchmarks/growing_fault/onecut.prm
benchmarks/growing_fault/check_interface.py
```

## 2. The two directions of mature fault coupling do different amounts of discrete work

**Fix or quantify before using the boundary-spanning model for BP3. Inherited from before C2.** The fault equations sample the bulk with eight Gauss points per coordinate. When the geometry is boundary spanning, the ordinary Q2 Stokes assembler samples the returned fault load with three. A label-to-label linear slip rate typically has a change of slope *inside* a bulk cell. Two tensor Gauss rules over that cell can give different integrals even when the bulk strain, stiffness and phase are all constant. The interior-fault branch already takes explicit steps to make the two discrete actions agree. The boundary-spanning path bypasses that projection.

Here is a small counterexample to exact work agreement. Use one unit Cartesian cell, a full-width straight vertical trace, a uniform positive admissible phase, fitting disabled, and labels at heights 0, ½ and 1. Change only the middle label's speed by one unit about a positive base speed. The interpolated change is the hat function `1 − |2y − 1|`. The full-width normalization gives constant χ. Pair the returned stress with a constant virtual shear, for example the local Q2 test field `v = (y, 0)`. After removing the shared stiffness and tensor factors, the actions are:

| Integration | Work divided by the common physical factor |
|---|---:|
| Exact integral, splitting at the middle label | 0.500000000000 |
| Ordinary Stokes load, Gauss 3 | 0.569668517088 |
| Fault equation, Gauss 8 | 0.494235968293 |

I evaluated these quadratures separately. This is a counterexample for the implemented *local operators*. It does not give an error estimate for an earthquake simulation. It does demonstrate that a passing section tangent, or a Stokes RHS checked against another Q3 evaluation at constant slip rate, cannot establish the paired-work property. The cut-cell projection used when publishing the final DG stress also comes too late to change the force in the Stokes solve.

**Acceptance.** Inject the label perturbation in a test, read the production local Stokes RHS before boundary constraint elimination, and contract it with the test velocity. Compare that against the reverse section action assembled from that velocity. Also compare each to the independent split integral on successively resolved meshes. Distinguish discrete transpose agreement from accuracy against the integral. Use a common integration or projection in the fault contribution. Raising the quadrature order of every ASPECT Stokes calculation would be an unnecessarily broad change.

In file order, see lines 92–96, 209–215, 285–305, 378–404 and 427–439 in the first file, 195–223 in the second, 25–34 in the third, 157 in the fourth, 804 in the fifth, and 143–187 and 384–405 in the last.

```text
source/reconstructed_fault/stokes_coupling.cc
source/reconstructed_fault/phase_profile.cc
source/simulator/assemblers/fault_stress.cc
source/simulator/introspection.cc
source/simulator/assembly.cc
benchmarks/fixed_fault/phase_ridge/phase_ridge.cc
```

## 3. On strip 8 the two inputs have essentially the same preferred angle

**Keep the C1 scientific comparison open.** The recap correctly identifies the first measurable difference in the solved midpoint fields. That observation alone leaves two explanations: the change in the fields may actually move the optimum, or the production calculation may locate practically the same optimum differently. The archived inputs let us tell these apart.

I reproduced the committed replay output byte for byte. I then independently bracketed the signed creep root at each trial angle, directly summed the squared point tractions over the 180 native samples with 60-decimal arithmetic, scanned the same forward sector, and located the peak within the winning bracket on each layout. Both scans chose grid index 52 of 0–128. The calculations used each run's own captured field and pressure. Repeating the derivative calculation at angular spacings of 1e−7 and 1e−9 rad changed each reference angle by about 2.2e−17 rad. I also obtained the same peaks with the repository's high-precision point-score implementation.

| At 5 ms, radians | One MPI rank | Two MPI ranks |
|---|---:|---:|
| Production angle | −0.147887429773508 | −0.147886573135828 |
| Direct-score peak in the selected bracket | −0.147885945931339 | −0.147885945935012 |
| Production minus direct peak | −1.48384e−6 | −6.27201e−7 |

The direct peaks differ by **3.67e−12 rad**. Production differs by **8.57e−7 rad**, against **1.15e−7 rad** allowed. Both production answers fall on the same side of their respective peaks. The discrete score has negative local curvature, approximately −1.11e−8 in score units per radian squared. At the production angles the published signed rates agree with their respective independent roots to within 7.11e−14 relative. Reversing a published strip-8 rate in a scratch copy made the replay fail as it should.

For this snapshot, perturbations in the solved field account for very little movement of the mathematical peak. The combined finite-precision score evaluation and search deserve the next numerical investigation. These measurements do not apportion error between moment accumulation, scalar root evaluation and the final angle comparisons. They establish a reference against which those operations can be tested.

**Next experiment.** Apply the same direct-peak calculation to the historical constant-pressure capture at the first divergent decision. Compare production, the isolated candidate and the direct peak on *the same input*, then follow the accepted arclength. The candidate already passes the affine case and misses the frozen nine-strip arclength by 2.86e−8 m against 2.51e−8 m. If the more accurate peak changes the historical trajectory, bring the quantified choice back to the author before replacing a scientific reference. Keep the four older strip-15 timestep findings separate. This investigation can start on archived points on a laptop.

The sector scan and comparison are at lines 193–253 of the first file. The second file contains the expanded and centered formulas. The remaining two files reproduce the C2 investigation. An independent reproduction of the new peak calculation appears in the appendix.

```text
source/simulator/solver/midpoint_direction.cc
include/aspect/reconstructed_fault/onset_score.h
refactoring/phase-field-fault/results/interface-c2/replay.py
refactoring/phase-field-fault/results/interface-c2/c1-captures.tar.gz
```

## 4. The mature material and the energy observer can read different fields

**Fix with the public parameter repairs.** The mature material looks up the literal name `phase_field`. The growing material and surface-energy observer look up the configured `Phase composition`. Renaming the physical field to `damage` consistently in the field list and phase settings makes mature initialization fail. Conversely, leaving a field named `phase_field` present and pointing the observer at a different existing field can make the reported energy describe something other than the fault. With zero stress a wrong stress-component selection can even look like a valid zero energy.

Use the resolved phase name throughout mature mechanics, keeping `phase_field` as the default. Check the selected slot against the stress slots. Test a renamed field with the energy observer enabled and disabled. Automatic inference of interior geometry from the old phase role is a separate compatibility rule. When geometry is explicit, reconsider whether the physical meaning of the very same phase should need to change between `generic` and `phase field`; first verify the basis and transport selected in both cases. The general energy observer should remain usable in a fault-free run.

In file order, see lines 280 and 389–403, 47–51, 21–28 and 65–75, and 99–105.

```text
source/material_model/phase_field_fault.cc
source/material_model/growing_fault.cc
source/phase_field/surface_energy_model.cc
source/postprocess/phase_field_surface_energy.cc
```

## 5. Two details still bind the friction interface to rate-and-state assumptions

**Mature interior restart omits the law parameters.** The resistance interface explicitly exposes the physical parameters that must agree on restore. Growth checks them. The mature interior archive checks the law key and the adiabatic profile, then restores the label history without comparing that parameter vector. Keeping the same law name while changing its impedance or characteristic slip distance can silently continue the prior history under different physics. Add a versioned signature for newly written mature interior archives and keep the existing legacy-read policy explicit. Exercise an unchanged restart and one with a changed parameter of the *same* law. If intentional changes on restart are desired, define that separately from an ordinary reproducible continuation.

**A stateless mature law still depends on the RSF reference speed.** The mature material reads `Reference speed` before creating the law, initializes all trace rates with it, and later insists it lies above the minimum. That entry is declared by RSF. The installed dashpot can be rejected solely because an inactive RSF entry is zero. A recoverable design is a physically named initial slip rate owned by the trace/material initialization, with the historic RSF value as the compatibility fallback. Keep the RSF reference speed in its logarithmic friction law. Test the two speeds independently and show a stateless run with no RSF-specific input.

The interface contract is at lines 67–68. In the remaining files see lines 327–329, 363–375 and 426–485 of the mature material, 149–151 and 240–249 of the growing material, and 179–197 of the RSF implementation.

```text
include/aspect/material_model/rheology/fault_resistance.h
source/material_model/phase_field_fault.cc
source/material_model/growing_fault.cc
source/material_model/rheology/rate_and_state_friction.cc
```

## 6. The installed growing-history exporter can report success with missing or stale data

The new postprocessor opens per-rank point and coefficient files, a memory report, and root-owned strip, pressure and balance files. It returns success without checking creation or final writes. A path collision or storage failure can leave the scientist with an accepted timestep and a missing physical-stress file. The mature label exporter already checks the stream after closing it.

There is also a reproducible issue when reusing an output directory. Overwrite an existing two-rank frame with a one-rank frame of the same step. The old rank-1 file survives. I exercised the shipped reader on precisely this combination from the captures: the new file has 4,968 points, the stale shard has 2,484, and the reader raises a duplicate-key error on cell `16_0:`, point 0. A reader that concatenates CSVs would double-count part of the domain.

**Acceptance.** Report collective write failure with the filename and rank, including failure detected on close. Ensure readers can identify exactly the shards of the most recently completed frame when rank count changes, preserving the public filenames. Test a blocked point-output filename in an otherwise writable directory and a repeated two-to-one-rank frame in the same directory. Keep the useful stress, strip and work outputs. Dumping all FE coefficients and peak memory on each output is a separate, opt-in verification need worth moving off the default scientist-facing path while retaining benchmark reproducibility.

In file order, see lines 31–34 and 63–103, 40–52, and 18–30 and 53–57.

```text
source/postprocess/growing_fault_history.cc
source/postprocess/mature_fault_statistics.cc
benchmarks/growing_fault/compare.py
```

## 7. The canonical-versus-legacy test passes a reversal of the creep direction

The installed interface comparison uses the same absolute floor, 1e−12, for almost every growing point and strip quantity. At strip 8 the rate is about +1.33e−13 m/s. I fed the comparator the archived table and a copy with just that rate negated. It accepted the 2.65e−13 m/s difference. The dedicated signed-root replay rejected the same mutation. These tests answer different questions: the root replay gives good evidence for the captured run, while the interface check should detect a change of physics introduced by either public spelling.

For inputs intended to be equivalent on the same layout, compare exact discrete identities, formation flags, and signs. Use exact equality where the operation order is promised identical, and physical column-specific tolerances elsewhere. A near-zero rate can be checked relative to a signed root with a separate zero-rate case. Make this one-value mutation fail in the canonical-versus-legacy test itself.

The checks are at lines 50–59 of the first file, 33–43 of the second, and 80–93 of the third.

```text
benchmarks/growing_fault/check_interface.py
benchmarks/growing_fault/compare.py
refactoring/phase-field-fault/results/interface-c2/replay.py
```

## Your three questions about crack geometry

- **A class declared in a source file.** I would keep `PolynomialGeometry` private here. The public contract `CrackGeometry` is already declared in its header. The concrete class implements three polynomials from their initial slopes and normalizations, and only the selector constructs it. A second public declaration or three public subclasses would add places to read without giving any caller a capability it needs. If user-defined geometries become a real requirement, the design question is registration and lifetime.
- **The selector.** `AT1|AT2|CZM` is the grammar for alternatives in deal.II. The surrounding parameter pattern accepts zero or one entry. Zero supports an omitted canonical or legacy spelling. A user cannot select AT1 and CZM together. Naming the helper `get_valid_model_names_pattern()` or documenting that return value would make this easier to read.
- **CZM growth.** Your objection identifies the actual scientific limitation. The model knows the CZM crack-density function, but the growing operator has the AT1 equations compiled into it. The normalized local phase residual currently adds ∫Nᵢ, its local Jacobian contribution is zero, and the ledger books energy proportional to ∫d. For CZM, α(d) = 2d − d². At d = ¼ the three local quantities α, α′ and α″ are 0.4375, 1.5 and −2. In particular, the local phase Jacobian needs the contribution −2∫NᵢNⱼ in the stage variable. The endpoint ledger must use the actual change of α. For this quadratic α the midpoint derivative gives its exact discrete change. The SIPG gradient term has its own separate contribution.
- **The onset calibration changes too.** With the current cohesive degradation, the driving factor at intact phase is m. The homogeneous intact threshold satisfies Kf α′(0) = m f²/(2G), where f denotes the effective driving shear in the local Maxwell response and Kf = Gc/(c₀ℓ). At fixed intended strength c, AT1's m = 2G Kf/c² would give CZM onset at √2 c. Matching that strength requires the α′(0) factor or a separately justified cohesive law. AT2 has α′(0) = 0 and needs its own physical interpretation for inception. This is why I would keep the guard until the implementation and its tests are ready. The affine AT2 test in the recap qualifies the independent energy calculation.

Assign CZM as a separate scientific change. Require a homogeneous inception test at the specified cohesion, independent residual and tangent actions at nonzero phase on inactive obstacle rows, and an endpoint energy/irreversibility check. Retain the AT1 reference. A registration test or an isolated comparison of the three scalar polynomials cannot qualify a coupled growth law.

In file order, see lines 19–75, 21–28, 7–12 and 33–39, 393–411 and 738–743, 76–80, and 23–41.

```text
source/phase_field/crack_geometry.cc
source/phase_field/surface_energy_model.cc
source/phase_field/midpoint_fracture.cc
source/simulator/solver/midpoint_fault.cc
source/material_model/growing_fault.cc
source/material_model/rheology/cohesive_maxwell.cc
```

## Architecture and work that can wait

| Area | Maintainer judgment |
|---|---|
| Geometry, rheology and the attempt | Keep the matching cohesive-Maxwell header/source pair, the shared graph used by strip pressure and direction selection, the standalone DG operator, the law-owned optional history and the solver-owned attempt. A new universal fault manager would have to reconcile Q1 material labels with growing strips, two pressure locations, and different time integrations. The common user name does not require that inheritance hierarchy. |
| Degradation | The point response and plane score now call the same bounded fraction. There is one remaining identity to document and verify: for this particular law, `fraction_derivative()` and `driving_factor()` are mathematically identical. Both equal m(1+3d)(1−d)/[md(1+d)+(1−d)²]². The current class maintains the two expressions separately and exposes writable numerator/factor members alongside private phase and magnitude. Give the evaluated state a consistent read-only interface and test the identity, including endpoints. Preserve arithmetic ordering where the current reference is sensitive until a numerical change is assessed separately. |
| Installed SDK | The install now distinguishes generated World Builder headers for Debug and Release. Its generated plugin package still embeds the configure-time install prefix for its headers. Installing with a different `cmake --install … --prefix …` or moving the tree can send an external plugin to the wrong SDK. Some executable-path assumptions predate this branch. Before upstreaming the install change, derive the installation's own paths from the installed package location and compile a plugin after hiding the original prefix. The reported fresh install exercised a single chosen prefix. |
| Upstream restart compatibility | A fault-free, unmarked ASPECT 3.1 **2-D** checkpoint with degree-two DG composition, without splitting or melt, falls into the branch's blanket ambiguity rejection. There was real nodal/Bernstein ambiguity among historical *branch* writers. Preserve that safety check for unknown producers and supply a tested import or explicit producer-basis mechanism for users who can verify a 3.1 checkpoint. This affects ordinary ASPECT users and deserves separate upstream review. |
| AMG and GMG | Mature mechanics explicitly requires block AMG. On growth, both parameter choices have been exercised, but both reach the rank-zero UMFPACK factorization for the coupled matrix. Neither result qualifies a growing GMG solver. Gathering that matrix is also a scaling limitation when choosing sizes for growth experiments. |
| Public examples | The growing seed includes a pressure override which itself includes an old four-field fixture. The reader has to undo inherited fields and interpret long generated trigonometric expressions to recover a simple inclined coordinate system and a boundary velocity. The layered input still embeds a long table of nearly equal-spaced coordinates. In Stage D provide readable primary inputs, a named strike/dip coordinate definition or named constants, line-broken velocity components, and vertex files. Make the independent variables, units and positive slip direction visible beside the input. Keep the historical fixtures as comparisons. Apply neighboring ASPECT declaration/indentation style to edited parameter blocks. |
| Subduction | The vertex reader is useful now for straight BP3 seeds. All initial vertices must lie on a **straight** ordered seed even though an evolved trace can bend. The guide says this; the parameter description and recap can easily suggest arbitrary polylines. A curved slab geometry, the role of temperature and density, gravity in bulk momentum, and frictional compression on the interface need an explicit qualification before the realistic subduction example. Selecting adiabatic friction pressure alone supplies only the last of these. |

Additional source locations, in the order shown below, are lines 14–25, 30–45, 994–1009, 26–37 and 67–101, 155–182, 239–244 and 376–381, 210–230 and 503–548, 47–58, 2–15, 53 and 83, and 85–88.

```text
include/aspect/phase_field/degradation.h
source/phase_field/degradation.cc
CMakeLists.txt
cmake/AspectConfig.cmake.in
source/simulator/checkpoint_restart.cc
source/material_model/phase_field_fault.cc
source/simulator/solver/midpoint_fault.cc
source/reconstructed_fault/material_fault_trace.cc
benchmarks/growing_fault/growing_seed.prm
benchmarks/growing_fault/onecut.prm
benchmarks/layered_moving_fault/layered.prm
```

## What passed, and what to do next

The remote tip matches the attached recap. The branch diff from 3.1 spans 258 files, including 28 headers and 30 implementation files under the main include and source trees. I reviewed the branch-level changes and followed the public inputs through parsing, dispatch, FE and fault assembly, acceptance, history, output, and their checkers. The recorded mature file/inline equivalence, repartitioned label identity, installed plugins, 2,574 Release assertions, affine AT2 energy test and frozen nine-strip control remain valuable. The unchanged four strip-15 failures are an explicitly retained comparison from 5B. The separate strip-8 angle and arclength checks still fail. This review environment had no ASPECT executable or deal.II development installation. Native build, CTest, installed-run and MPI-solve results above come from the submitted record. I reran the Python capture replay, a stationary replay unit test and the calculations and negative controls described here. This was a read-only review of the branch.

1. **First Codex assignment — finish the public interface.** Address findings 1, 4, 5 and 7 with isolated commits. Address finding 6 in a separate output commit. Demonstrate changes through the installed public inputs, including the two negative controls. Keep established numerical trajectories for unchanged inputs. This assignment can proceed on the laptop.
2. **Separate scientific assignment — mature weak work.** Build the matrix-action test in finding 2 before editing the assembler. Preserve its pre-fix result. Fix discrete work and quantify integration accuracy on a resolved, inclined ridge, then rerun the existing mature controls. This result matters for BP3.
3. **Separate numerical investigation — strip 8.** Use the archived local-maximum comparison on both pressure controls. A frozen trajectory change needs a scientific decision supported by the first divergent angle, its direct reference and the resulting arclength. Keep this separate from C2 interface edits.
4. **Then Stage D and further physics.** Rewrite the public examples against the settled parameter interface. Scope CZM growth, curved subduction geometry, and solver scalability separately. The current maintainer baseline is sufficient. The practical improvement is to require a value-changing input test, an independently assembled weak-work test, or a deliberately corrupted result for each corresponding claim.

## Reproducing the direct-peak calculation

From the reviewed checkout, extract the archived inputs into a scratch directory. Save the Python block below as `direct_peak.py` there, then run the following with Python and mpmath. It uses the original binary-double values encoded in the sampled output, matching the input to the repository's direct reference. It independently brackets the root between zero and the free rate for either slip sense, evaluates the pointwise score, and checks the peak on both layouts. The field snapshots and material constants correspond to this first-turn specimen.

```sh
work=$(mktemp -d)
tar -xzf refactoring/phase-field-fault/results/interface-c2/c1-captures.tar.gz -C "$work"
# Save the Python block below in "$work/direct_peak.py".
python3 "$work/direct_peak.py" "$work/build-interface-c2/cap1/output" "$work/build-interface-c2/cap2/output"
```

```python
import csv
import sys
from pathlib import Path
import mpmath as mp

mp.mp.dps = 60
h, G, eta, dc, V0, a, b, mu, damping = map(mp.mpf,
    (0.0025, 1e6, 1e8, 0.01, 0.001, 0.015, 0.02, 0.6, 1000.))
B = 1 / (1 + h * G / eta)

def read_table(path):
    with path.open() as f:
        return {int(row['section']): row for row in csv.DictReader(f)}

def inspect(path):
    width = mp.mpf(float(read_table(path/'sections-0000.csv')[8]['ds']))
    actual = mp.mpf(read_table(path/'sections-0001.csv')[8]['beta'])
    pressure = mp.mpf((path/'root-input').read_text().split()[-1])
    samples = {}
    for shard in path.glob('turn-points-*'):
        for line in shard.read_text().splitlines():
            cell, q, *data = line.split()
            key = cell, int(q)
            assert key not in samples
            samples[key] = tuple(mp.mpf(float(v)) for v in data[:5])
    assert len(samples) == 180
    pts = [samples[key] for key in sorted(samples)]
    weighted_u = mp.fsum(J*F*u for J,_,F,u,v in pts)
    weighted_v = mp.fsum(J*F*v for J,_,F,u,v in pts)
    weight = mp.fsum(J*F for J,_,F,u,v in pts)
    def shear_force(speed):
        x = h * abs(speed) / dc
        age = 10*mp.exp(-x) + h*(-mp.expm1(-x)/x if x else 1)
        return pressure*a*mp.asinh(speed/(2*V0)*mp.exp((mu + b*mp.log(age*V0/dc))/a)) + damping*speed
    def score(beta):
        s, c = mp.sin(2*beta), mp.cos(2*beta)
        scale = G*B*h*width/mp.cos(beta)
        free = (weighted_u*s+weighted_v*c)/scale
        mobility = weight/scale
        lo, hi = min(0, free), max(0, free)
        for _ in range(180):
            v = (lo+hi)/2
            if v + mobility*shear_force(v) < free:
                lo = v
            else:
                hi = v
        f = shear_force((lo+hi)/2)
        return mp.fsum(J*kernel*(1-F)**2*(u*s+v*c-f)**2 for J,kernel,F,u,v in pts)
    scan = [score(-mp.pi/4+mp.pi*i/256) for i in range(129)]
    print(path, 'best grid point', max(range(129),key=scan.__getitem__), 'interior grid peaks',
          [i for i in range(1,128) if scan[i]>max(scan[i-1], scan[i+1])],flush=True)
    for step in ('1e-7','1e-9'):
        delta=mp.mpf(step)
        derivative=lambda beta:(score(beta+delta)-score(beta-delta))/(2*delta)
        peak=mp.findroot(derivative,('-.14789','-.14788'),tol=mp.mpf('1e-35'),verify=False)
        curvature=(score(peak+delta)-2*score(peak)+score(peak-delta))/delta**2
        print(step,'peak',mp.nstr(peak,23),'selected minus reference',mp.nstr(actual-peak,13),
              'curvature',mp.nstr(curvature,12),'derivative',mp.nstr(derivative(peak),6),flush=True)
    return peak

one,two = [inspect(Path(path)) for path in sys.argv[1:]]
print('two minus one',mp.nstr(two-one,16))
```
