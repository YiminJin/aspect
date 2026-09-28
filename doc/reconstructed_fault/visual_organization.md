Please do a readability-only organization pass on the reconstructed-fault and phase-field-fault code.

This task is **not** about changing algorithms, APIs, assertions, numerical behavior, MPI behavior, or introducing new helper functions. The goal is only to make the existing code much easier for a human reader to navigate.

Apply the organization style commonly used in ASPECT headers such as `simulator.h`.

## 1. Group related declarations in headers with Doxygen groups

For large classes such as `ReconstructedFaultManager` and `PhaseFieldFault`, group related public/private members using:

```cpp
/**
 * @name <descriptive group name>
 * @{
 */

// declarations

/**
 * @}
 */
```

Use a small number of meaningful groups, not one group per function.

For `ReconstructedFaultManager`, use groups roughly corresponding to:

- Construction and parameter handling
- Fault initialization and reconstruction
- Generic reconstructed-fault properties
- Slip-rate nonlinear state
- Particle-to-fault projection
- Fault access and diagnostics
- Serialization and restart
- Private projection/cache helpers
- Persistent/reconstructible state

For `PhaseFieldFault`, use groups roughly corresponding to:

- Material-model interface
- Maxwell constitutive helpers
- Cohesive constitutive helpers
- Initial cohesive-state setup
- Normalization-integral evaluation
- Normalization-profile integration
- Material parameters and state

Keep related structs, aliases, helper declarations, and state variables close to the group where they are conceptually used.

Do not create excessive nesting or dozens of tiny groups.

## 2. Mirror the same organization in source files

In the `.cc` files, visually separate major implementation sections with clear divider comments.

Use a consistent style such as:

```cpp
// -----------------------------------------------------------------------------
// Fault reconstruction
// -----------------------------------------------------------------------------
```

or, if that matches the local ASPECT style better:

```cpp
/* -------------------------------------------------------------------------- */
/* Fault reconstruction                                                       */
/* -------------------------------------------------------------------------- */
```

Choose one style and use it consistently within each file.

The major source-file sections should mirror the header organization where practical.

For `reconstructed_fault.cc`, use sections approximately like:

- File-local helpers
- Prescribed-fault parsing and initialization
- Fault reconstruction
- ReconstructedFault implementation
- Property registration
- Slip-rate state
- Particle-projection cache
- Particle-to-fault projection
- Fault access and diagnostics
- Restart reconstruction / serialization support

For `phase_field_fault.cc`, use sections approximately like:

- File-local helper types
- Maxwell constitutive law
- Material-model interface
- Cohesive constitutive law
- Initial cohesive-state initialization
- Normalization phase-field utilities
- Adaptive normalization-profile integration
- Surface composition / normalization-profile construction
- Projection of normalization integrals
- Material parameters and parsing

## 3. Preserve logical execution order

Within each section, order functions so that a human can read from higher-level workflow to lower-level implementation where possible.

Prefer this ordering:

1. public/top-level workflow;
2. major helper routines called by it;
3. lower-level numerical helpers.

Do not mechanically sort functions alphabetically.

## 4. Keep visual hierarchy stronger than comment density

Do not add comments before every function.

The section dividers and Doxygen groups should provide the main navigation.

Function comments should still explain only:
- purpose;
- non-obvious numerical meaning;
- important MPI/state assumptions.

Do not add comments that merely repeat function names.

## 5. Keep the diff strictly organizational

Do not:
- rename functions or variables unless required for section consistency;
- change function bodies;
- change assertions;
- change helper decomposition;
- change public/private visibility;
- move logic between classes;
- reformat unrelated code;
- alter numerical expressions.

This pass should make the code easier to scan without changing behavior.

## 6. Desired result

After this pass, a developer opening the header should be able to understand the subsystem by reading the group names.

A developer opening the `.cc` file should be able to scroll quickly to a clearly labeled major section without searching for function names.

Please first show me the proposed group/section layout for each of the four files before editing.