# Reconstructed-fault state timestep limiter

Implemented in `source/time_stepping/reconstructed_fault.cc`; rebuild **ASPECT**,
not only the BP3 shared library. No BP5 timestep plugin is required.

```text
subsection Time stepping
  set List of model names = convection time step, reconstructed fault time step
  subsection Reconstructed fault time step
    set Maximum logarithmic state change = 0.1
  end
end
```

The default `std::numeric_limits<double>::max()` is a disabled sentinel;
the literal `infinity` is also accepted and mapped to that sentinel. Either
returns exactly the previous law-specific proposal. A finite value must be positive. The bound
is dimensionless and **unweighted**: no BP5 b/a factor. Stateless friction ignores
it. Existing production PRMs have not been opted in.

For finite bounds the plugin caps its law-specific proposal by the existing
global maximum timestep, reads every fault's committed Q1 velocity and Theta,
and calls the existing `FaultFriction::update_state(V,Theta,dt)`. This is the
same exponential implementation called by the mixture overload at commit; Dc
is still the material model's existing global value. If any vertex violates
the bound, halving finds a positive safe bracket and bisection returns its safe
endpoint. No current/trial velocity or constitutive history is modified.
The existing time-stepping manager combines controllers and MPI-reduces the
proposal. Its minimum-timestep floor is unchanged; keep that floor zero when
controller restrictions must not be overridden. This forward predictor does
not constrain the actual state change at the next newly solved velocity.

## Executed checks

The subsequent sentinel fix uses `std::numeric_limits<double>::max()` consistently
in the header, parameter default and bypass comparison. It initializes the member,
rejects zero, and preserves `infinity` as an alias. `sentinel-fix/build.log` and
`plugin-build.log` record successful Release builds; `sentinel-fix/parameter-tests.log`
passes 16 assertions including the maximum-double round-trip. The expanded
one-rank `sentinel-fix/integration.log` checks that both disabled spellings return
the unchanged proposal, in addition to the existing numerical checks. The prior
MPI qualification below was not repeated for this parameter-only fix.

Local Release GCC 12.4 / OpenMPI 5.0.6 / deal.II 9.6.2 build:
`build-final-range.log`, `plugin-build-final-range.log`.

- `parameter-tests.log`: 14 assertions; default infinity, positive finite
  values, and rejection of zero, negative, NaN and malformed values.
- `stateful-free.log`: real plugin on the initialized small Stage-I fixture,
  comparing increasing/decreasing predictions against an independent analytic
  inverse of the ODE solution. Includes equilibrium, a 1e-30 initial state
  requiring a timestep far below the proposal, a large finite bound with
  overflow of the raw state ratio, the last-vertex maximum, repeated calls,
  trial versus committed velocity, and preservation of Theta.
- `mpi-free.log`: the same assertions pass on two ranks; proposals agree.
- `stateless-final.log`: finite/infinite settings preserve the old stateless
  proposal and never request a missing Theta property.
- `screen-output`: deterministic two-rank success marker, matched against
  `tests/phase_field_fault_state_limiter/screen-output`.

The integration observer temporarily installs synthetic nodal probes and then
restores the original state/rates through the existing manager APIs. It is
test-only, not a production plugin. Tests run through initialization at t=0;
they do not claim a BP3 trajectory or a server/compiler qualification. The
existing global manager's combination/floor behavior was inspected, not changed.

Initial fixture wiring failures and the historical all-Dirichlet Stage-I
rate/state pressure-compatibility failure are preserved in earlier logs. That
pressure failure occurred during mechanics, before the limiter was called.
The new stateful fixture uses a free top boundary to avoid that unrelated
nullspace issue; no core pressure checks were relaxed. The stateless fixture
retains the original boundaries. MPI needed permission to use local sockets
outside the sandbox. No Python script was used for this task.

## Reproduce

From the repository root, with the matching compiler/MPI environment loaded:

```sh
cmake --build build-tmp -j2
build-tmp/aspect-release --test '[fault_state_limiter]'
cmake -S benchmarks/reconstructed_fault/state_limiter -B /tmp/aspect-state-limiter \
  -DAspect_DIR="$PWD/build-tmp" -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/aspect-state-limiter -j2
```

The `stateful.prm`, `stateless.prm` and `mpi.prm` wrappers identify the executed
test inputs and local library. Use new output directories when repeating them,
preserving the recorded evidence. The standard test suite also has
`tests/phase_field_fault_state_limiter.prm` and its deterministic output filter.
