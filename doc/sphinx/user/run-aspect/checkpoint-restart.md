(sec:checkpoint-restart)=
# Checkpoint/restart support

If you do long runs, especially when using parallel computations, there are a
number of reasons to periodically save the state of the program:

-   If the program crashes for whatever reason, the entire computation may be
    lost. A typical reason is that a program has exceeded the requested
    wallclock time allocated by a batch scheduler on a cluster.

-   Most of the time, no realistic initial conditions for strongly convecting
    flow are available. Consequently, one typically starts with a somewhat
    artificial state and simply waits for a long while till the convective
    state enters the phase where it shows its long-term behavior. However,
    getting there may take a good amount of CPU time and it would be silly to
    always start from scratch for each different parameter setting. Rather,
    one would like to start such parameter studies with a saved state that has
    already passed this initial, unphysical, transient stage.

To this end, ASPECT creates a set of files in
the output directory (selected in the parameter file) every N time steps
(controlled by the number of steps or wall time as specified in
`subsection Checkpointing`, see
{ref}`parameters:Checkpointing`) in which the entire state of
the program is saved so that a simulation can later be continued at this
point. The previous checkpoint files will then be deleted. To resume
operations from the last saved state, you need to set the `Resume computation`
flag in the input parameter file to `true`, see
{ref}`parameters:global`.

:::{note}
It is not imperative that the parameters selected in the input file are exactly the same
when resuming a program from a saved state than what they were at the time when this state
was saved. For example, one may want to choose a different parameterization of the material law,
or add or remove postprocessors that should be run at the end of each time step. Likewise, the
end time, the times at which some additional mesh refinement steps should happen, etc., can be
different.
Yet, it is clear that some other things can’t be changed: For example, the geometry model that
was used to generate the coarse mesh and describe the boundary must be the same before and
after resuming a computation. Likewise, you can not currently restart a computation with a
different number of processors than initially used to checkpoint the simulation. Not all invalid
combinations are easy to detect, and ASPECT may not always realize immediate what is going on
if you change a setting that can’t be changed. However, you will almost invariably get nonsensical
results after some time.
:::

### Compositional finite-element bases

Each checkpoint slot records the finite-element name for every composition in
field order in its `metadata.txt`. A nodal DGQ2 field and a quadratic Bernstein
phase store nine coefficients per cell in 2-D and 27 in 3-D. The same numeric
entries can represent different polynomials in these two bases. ASPECT checks
the saved bases before loading the solution vectors. A change of transport method is compatible with this check when the
field keeps the same basis.

Older slots omit this record. They can resume directly when the saved
discretization rules out a Bernstein field, for example with continuous
compositions or DGQ1. An unmarked 2-D DGQ2 field in a configuration that
could have used Bernstein requires evidence from the *producing* executable
and input. When that evidence establishes the bases, append a
`composition_bases` record to a copy of the slot's metadata, giving the field
count followed by the quoted FE names in their original order. For example,
a verified one-field nodal DGQ2 checkpoint takes
`composition_bases 1 "FE_DGQ<2>(2)"`. If the producing bases cannot be
established, the checkpoint is ambiguous and continuation is rejected.

Before the introduction of 3-D Bernstein transport, an unmarked 3-D DGQ2
composition was nodal. A 3-D checkpoint from that generation can resume into
a nodal field, including a signed explicit tracer. Selecting a Bernstein phase
for the same stored field is diagnosed before solution transfer. New 3-D
archives carry a format version. They require the per-field basis record when
the stored discretization admits a Bernstein field.
