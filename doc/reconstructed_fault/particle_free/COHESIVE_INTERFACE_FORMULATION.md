# Recoverable fracture compliance and geometric RSF/contact

Independent experimental formulation. This is a constitutive replacement, not
an exact transformation of the distributed bulk law or an enabled native path.
The selected mature limit is a registered geometric interface. Production/BP3,
its Γ implementation and all old checkpoint formats remain unchanged.

## 1. State, energy and intact admission

On an oriented material surface Γ let n be the normal, t=(n_y,-n_x,0), and
S=sym(t⊗n). The tangential displacement jump δ has units m; V=δdot is the
**one geometric interface velocity**, not a separate rate at each cross-band
quadrature point. The gap is zero. Total bulk stress is σ=τ−p I, with a full
three-dimensional plane-strain deviator τ. Normal contact pressure is
pn=−n·σn=p−n·τn>0, solved with mechanics. Loss of closed contact stops these
references; they do not implement opening.

New surface damage a∈[0,1] is **not** the former bulk phase φ. With an immutable
stress-free bond reference b, e=δ−b, define

    c(a)=c0 a/(1−a),  c0=2Gc/τp²,  δf=2Gc/τp,
    ψb(e,a)=e²/(2c(a)),  q=∂eψb=e/c(a),  Efracture=Gc∫Γ a ds.

The endpoint definitions are exact: a=0 imposes e=0 as a constraint; a=1 gives
q=ψb=0 for every finite e. No finite penalty at a=0, phase floor, normalization
floor, or remnant spring at a=1 is used. The damage force and irreversible law are

    Y=−∂aψb=e²/(2c0 a²),
    adot≥0, Y≤Gc, (Y−Gc)adot=0   (0<a<1),
    a_new=max(a_old,min(|e_new|/δf,1)).

At the tied intact endpoint q is the mechanical multiplier. Its one-sided
complementary-energy release is Y0=c0 q²/2: admission occurs at |q|=τp,
independently of dt or a pre-existing diffuse support. Along monotonic growth,
q=sign(e)τp(1−a); ∫q de=Gc through complete decohesion. At partial damage,
unloading is recoverable at the current compliance and a cannot decrease.
The upper obstacle permits Y≥Gc at a=1; the vanishing bond gives no further
cohesive resistance or storage.

Gc is irreversible surface creation energy, counted once. It may equivalently
be included as a fracture-energy inventory or as dissipation, **never both**.
This reference records Gc a as inventory and friction/viscosity separately.
A geometry/display lift φ or a Γ/I integration kernel must not add another
Gc∫Γ_bulk. Thus this proposal changes the old bulk phase energy and intact
admission semantics. It is not claimed to preserve the old degradation m-law.

## 2. Contact and thermodynamics

In parallel with the bonds, damaged contacts satisfy

    s=q+a R(V,pn,Θ),
    R=sign(V)[pn μ(|V|,Θ)+ηrad |V|],
    μ=a_RSF asinh[ |V|/(2V0) exp((μ0+b_RSF log Θ)/a_RSF) ],
    Θdot=(V0/Dc)(1−|V|Θ/V0).

The contact-activation interpolation **a** is a new physical assumption. It
represents a progressively active contact mechanism, not a fictitious width.
No dilation/volumetric plastic flow is added. Recoverable motion of partially
broken bonds can dissipate in their parallel contact mechanism. This is not
an additive series elastic-spring/plastic-slider split: making such a series
spring vanish would also destroy the mature frictional resistance.

For closed contact the local work identity is

    s V = ψbdot + Gc adot + a pn μ |V| + a ηrad V².

At fixed a the first term includes reversible unloading; during damage the
KKT law supplies Gc adot. At a=1 the identity reduces exactly to geometric
interface RSF/radiation work, with no accumulated-slip hardening, cohesive
backstress, or residual energy in diffuse flanks. At a=0 contact power is zero.
The inherited ηrad=G/(2 sqrt(G/ρ)) is the selected radiation impedance, not a
new numerical stabilizer. Quasistatic Maxwell bulk contributes τ:τ/(2η).
The frictional state has no stored energy in this phenomenological RSF law.
Full pn-dependence makes the coupled tangent nonsymmetric. This is a positive
power law under closed contact, not an associated volumetric flow potential.

### Fixed calibration (no pressure-dependent stored-energy parameter)

| Quantity | Value | Meaning |
|---|---:|---|
| G, η | 1e6 Pa, 1e8 Pa s | retained Maxwell material; r=G/η=.01/s |
| Gc | 1 J/m² | irreversible new surface energy |
| ell | .046875 m | retained geometric-lift scale; not another Gc term |
| old m | 4 | provenance only; old sqrt(3 G Gc/(4 ell m))=2000 Pa |
| p_ref, μ_ref | 10000 Pa, .1 | reference calibration, not a pressure lag |
| τp | 3000 Pa | 2000+μ_ref p_ref intact shear strength |
| c0 | 2.222222222e−7 m/Pa | compliance scale |
| δf | .000666666667 m | bond decohesion jump |
| a_RSF,b_RSF | .005,.01 | unchanged direct/evolution coefficients |
| V0,Dc | 5e−6 m/s,.001 m | physical interface rate and state length |
| ρ,ηrad | 2670 kg/m³, 25836.02137 Pa s/m | radiation impedance |

The nominal 2000 Pa strength drop is not residual cohesion. At maturity residual
cohesion is exactly zero; residual strength is actual pn μ(V,Θ)+ηrad |V|.
It need not equal 1000 Pa at other rates, states or pressures. τp is fixed at
the reference calibration; imposing a pressure-dependent τp would require
re-deriving the energy and normal coupling. No such recalibration is hidden.

## 3. New records, direction and history

For a **new** closed material plane, maximize the intact recoverable release
c0(t·τn)²/2 among allowed planes. For the tested confined in-plane states this
is the maximum-shear pair. It is not the old maximization of frictional flow
speed or a Coulomb plane-angle claim. A declared geometric/history preference
selects among exact ties; a coordinate change transports the preference and
all tensors. With no preference the conjugate symmetry is reported unresolved.
Once formed, the normal is a material structural record; reversal changes
signed V, not the normal. Small-strain kinematics are retained. Basis covariance
does not qualify finite rotations. Secondary fracture across an existing record
and general junctions remain undefined here.

At birth a=0, δ=b, signed/absolute slip=0, ψb=0. The full existing bulk τ is
unchanged; creating tied geometric metadata changes neither stress nor energy.
Then the joint formation/mechanical solve pays ψb+GcΔa+contact dissipation.
No finite damaged segment is inserted for free. All tested candidate records
share the same material contact-age clock: while tied, Θ=Θ_init+V0 t/Dc.
Newly admitted records inherit that value; neither formation nor a=1 resets it.
The prospective-plane clock is an explicit physical assumption, with Θ_init=1
in fresh references. This avoids silently declaring every new patch steady at
an arbitrary velocity. An existing native Γ record should retain its accepted
state and signed/absolute slip; its state is authoritative.

The seeded test prescribes a pre-existing mature interval and records its
initial fracture inventory .25 J/m separately. The surrounding path is exactly
intact and tied. Every path point is eligible to form; no tip-growth mask exists.
It is a path-restricted cohesive-front demonstration, not unconstrained 2-D
bulk phase evolution or off-path branching.

### Energy-accounted conversion of loaded old variables

The independent mapping check retains the old full T,A,K arrays and sets new
physical τ=T−2G A, preserving all components including zz. For a coherent old
mode A=χ e_old S+A_perp with orthogonal residual, define

    C_old=G∫χ²/h dy, q_old=C_old e_old,
    a_new=I_old (for the tested 0<I_old<1 fracture inventory),
    e_new=c(a_new) q_old, b=δ_old−e_new,
    δ_old=2∫S:K dy; retain K_perp and all original tensor records.

This preserves q, physical τ, geometric slip and fracture inventory. The
storage change is explicitly

    Wswitch=½ c(a_new) q_old² + E_perp − E_old.

A conversion requesting zero switch work is rejected unless this difference
vanishes. The compatible loaded control is chosen by solving c(a)=1/C_old
on a smooth fixed-support interval; the generic loaded example requires
negative switch work (recoverable energy extraction). That is **not** free
fracture energy or unreported heat. No existing production checkpoint is
converted. A full native conversion would also have to re-establish mechanical
compatibility and boundary work; this test only qualifies the instantaneous
mapping and its ledger.

A_perp is retained, not projected away. Its old matched Maxwell decay and
storage loss can continue at the existing r, with explicit dissipation and BE
remainder; there is no new relaxation parameter. This is a declared legacy
residual reservoir, not new cohesive resistance on the mature interface. The
newly accumulated slip is a geometric surface quantity; residual old K modes
are preserved separately. The mapped example uses a uniform old Θ, so its
surface age is identical. Arbitrary heterogeneous old cross-band ages do not
have a uniquely equivalent single RSF age; that conversion is **not qualified**.
Fresh runs and existing authoritative Γ histories avoid that ambiguity.

## 4. Mixed mechanical reference and complete return derivative

Two homogeneous incompressible Maxwell layers of heights ht,hb have clamped
outer walls and a closed interface. Fourier modes in x solve the actual mixed
incompressible layer equations. With upper/lower 2×2 Dirichlet-to-Neumann maps,

    Krel(k)=[Kup(k)^−1+Kdown(k)^−1]^−1,
    K=Krel_tt, N=Krel_nt,
    mdot+r m=Udot−V, s=K m, pn=p0−N m.

Here m is the elastic interface mismatch (not the old degradation parameter).
Unequal ht=.075 and hb=.05 m give nonzero N. Normal pressure is the solved
reaction at zero normal gap: its perturbation includes solved p′ and deviatoric
τnn separately. The mean normal pressure p0 is the imposed reaction/gauge.
The uniform mode K0=G/(ht+hb); large |k| gives K~G|k| and N→0. Mechanics has
no finite-element approximation in the normal direction; tangential Fourier
collocation remains a discretization. Real Nyquist modes have no sine partner,
so the odd normal symbol is zero there.

At an increment, β=1/(1+r dt), m_new=β(m_old+ΔU−Δδ), V=Δδ/dt and
Θ_new=(Θ_old+V0 dt/Dc)/(1+|V|dt/Dc). The nonlinear free-interface residual is
F=K m−q−a R. Its complete tangent is

    ∂F/∂Δδ = −βK − diag(qδ+aδ R+a RV_total/dt)
                    −β diag(a sign(V) μ) N,
    RV_total=pn dμ/d|V| + ηrad,
    dμ/d|V| = fraction [a_RSF/|V| − b_RSF (dt/Dc)/(1+|V|dt/Dc)].

The analytic zero-speed limit is used, not a rate floor. qδ=(1−a)/(c0 a) on
fixed-damage branches, −τp/δf during formation, and zero at maturity;
aδ=sign(e)/δf during formation and zero otherwise. Changing active sets use
one-sided derivatives. Intact variables are eliminated exactly. Independent
finite differences include the actual current pressure response. No contribution
is condensed twice and no artificial intact-row stiffness enters stability.

The increment is the Newton unknown. Subtracting almost equal absolute jumps
at reversal caused a 1.103e−8 Pa force residual in an early implementation;
using the increment directly preserves the unchanged 1e−8 Pa requirement.
The Fourier kernel is constructed with exact reciprocal/odd circulant symmetry;
independent-column FFT roundoff is recorded rather than weakening a symmetry
assertion. Underlying layer reciprocity and bulk-energy identities are checked
independently before this construction.

## 5. Work, stability and limits

At each step, endpoint external work equals bulk/bond storage changes, GcΔa,
Maxwell/friction/radiation loss and two separately reported numerical terms:

    Bbulk=½ Δm·KΔm,
    Rbond=q_new Δδ−Δψb−GcΔa.

Rbond can be negative during softening. It is a signed endpoint quadrature
remainder, **not** physical negative dissipation. Trapezoidal reaction work and
the sum of absolute stepwise physical-energy remainders are also checked.
Loading/hold/reversal times align with step boundaries. Exact BE closure alone
is not the physical energy qualification.

On each active seeded state, the symmetric part of −∂F/∂Δδ is tested for
positive eigenvalues. This is a sufficient local incremental strong-monotonicity
check on the tested branches; it is not a global stability certificate for all
formation configurations. The mature continuous-time interface symbol retains
pressure coupling and Θ perturbations. For V>0, let
h=(K+μN)/RV, RV=pn ∂Vμ|Θ+ηrad, RΘ=pn ∂Θμ. The (m,Θ) linearization is

    [ −r−h             RΘ/RV                ]
    [ −Θ h/Dc         −V/Dc+Θ RΘ/(Dc RV)   ].

Short waves have K~G|k|, N→0; one branch decays increasingly rapidly and the
state branch tends to −V/Dc. Fixed-plane RSF may still have physical
finite-wavelength instabilities in other regimes; the calculation does not
remove them or claim stability for arbitrary evolving directions. It also
cannot inherit the old distributed two-system acoustic defect: its kinematics
are different. Direction formation, interface junctions and global birth remain
separate qualifications before native propagation.

Γ reuse is geometrical and work-adjoint: with ∫χ dn=1, a lift Dslip=χ V S
satisfies ∫τ:Dslip dn=V∫χ τ:S dn. The independent polynomial check verifies
this identity. Admission itself uses geometric surface support and tied traces,
not division by zero phase normalization. A native diffuse Γ operator and the
sharp layer operator are **not** assumed discretely identical. Establishing their
consistent trace/compliance realization is the next implementation question.
