# Mechanical localization audit before phase advancement

Retain the selected Coulomb return and all physical parameters/tolerances.
Audit the full mixed displacement–pressure principal symbol, with actual
pressure-dependent resistance and changing plane directions. Distinguish
algorithmic finite-step derivatives from the continuous-rate active-system
limit. Sample phi=0,.2,.6,.75,.8,.9,.95,.99, stick and single/double slip on
the retained small-strain point loading path. Verify an independent active-plane
rate construction and a constant-resistance control. Do not infer mixed
ellipticity from fixed-pressure stiffness or the symmetric part alone.

For a wave normal n and admissible transverse amplitude a, put
A n = [tau_e:D3 sym(a*n)] n, b=(tau_P-Id)n. Incompressibility is n.a=0.
The relevant mixed traction matrix has columns (A a,b); pressure may have a
transverse constitutive action. Analyze its determinant/singular values and,
where n.b is nonzero, its pressure-eliminated transverse coefficient. Use the
full 3D plane-strain stress convention. A gradient in phi does not automatically
control a perturbation with delta phi=0.

Then use the existing confined .25 m block on only 4×4, 8×8 and 16×16 meshes
at unchanged physical ell=.046875 m. Select the phase/state and short loaded
window from the point audit before launching. Use a common physical loading
history, freshly initialized histories on each mesh, and one focused timestep
halving. Record physical slip concentration/width, contact, mechanical residual,
phase driving/resistance, and the actual nonsymmetric mixed tangent. Normalize
conditioning against the same mesh's elastic H1/pressure L2 Riesz metric to
separate mesh scaling/pressure-reference coordinates from constitutive loss.
All native runs have bounded timeouts; failures and checkpoints are retained.

If mechanical loss is found, retain the fixed-phase evolution guard, do not
begin a phase-active specimen, and propose an energetic/dissipative slip
regularization versus a fully consistent dilatant alternative. Do not select
a new constitutive parameter or loosen acceptance. If the audit supports the
mechanics, continue automatically to the small phase-active qualification.
No full specimen, AMR, transport, limiter, RSF or BP3 change is selected.

Selected after the point audit, before native launches: phi=.8 (h=144),
fully compressive and with positive uniform-obstacle phase residual, versus
phi=.2 controls. The actual finite-step single-system tangent is already
nonelliptic at dt=.0125 s while fixed-pressure stiffness remains positive.
Run fresh 4/8/16 grids through .125 s, with the retained shear-to-normal change
at .1 s; compare dt=.0125 and .00625 over this same short history. Add only
4/16-grid phi=.2 controls. Preserve numerical failures and compare only common
accepted times. Phase remains fixed because the mechanical prerequisite fails;
measure zero phase increment and the actual phase-force/obstacle balance,
rather than enabling an unsupported phase solve to manufacture growth.
