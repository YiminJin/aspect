# Stage 3

## Purpose

construct the initial particle-based crack-driving force \(H\) from prescribed fault geometry and prescribed core phase-field values.
Implement the first phase-field-aware construction of an initial sharp fault.

Inspect the existing phase-field implementation first. Reuse the existing classes and functions:
* Stationary-profile formula `PhaseField::PhaseFieldProfile`;
* Geometric model parameters `PhaseField::GeometricFunction`;
* Acceptable phase-field range `MaterialModel::PhaseFieldModel::get_phase_field_range()`;
* Value of $H$ in stationary profile: `PhaseFieldHandler<dim>::stationary_crack_driving_force(volume_fractions, phi, phi_hat)`.

For the current 2-D implementation, represent each prescribed initial fault as an ordered polyline. For a particle position $\boldsymbol x_p$, compute the closest point $\boldsymbol x^*_p$ on the polyline. Initialize $H$ according to the core phase-field $\hat\phi(\boldsymbol x^*_p)$ and the phase-field $\phi(|\boldsymbol x_p - \boldsymbol x^*_p|)$. The volume fractions can be obtained from particle properties (assume that all chemical fields are advected by particles).

Support multiple non-overlapping prescribed faults. Throw an error if more than one fault contributes at a particle.T he prescribed profile is only used to initialize $H$. The sharp fault will be reconstructed later from the actual solved $Q_1$ phase field.

This task must not reconstruct the sharp fault from the prescribed stationary profile. 

Do not follow the instructions strictly. For example, if you find it unnecessary to compute the coordinate $\boldsymbol x^*_p$, then do what you think is the most efficient to implement the above functions.

Do not implement propagation, smoothing, RSF coupling, generic fault properties, particle-to-fault projection, fault-to-QP interpolation, or 3-D functionality.

Do not remove or refactor the Stage-2 distributed phase-field sampling functions in this task; they will be reconsidered when the Q1 reconstruction stage is implemented.

Before editing, report:

1. how the existing infrastructure will be used;
2. how the value of $H$ will be calculated;
3. which files you plan to modify;
4. any ambiguity that affects scientific behavior.

