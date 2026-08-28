# Reconstructed-fault visual validation

This minimal two-dimensional example initializes one prescribed fault on
particles, solves the initial Q1 diffuse phase field, reconstructs a sharp
fault from that solution, and writes both bulk and reconstructed-fault VTU
output.

Run from the ASPECT source directory:

```sh
./build-pf-rsf/aspect cookbooks/reconstructed_fault/reconstructed_fault.prm
```

The model uses a prescribed zero Stokes solution and ends at time zero. It
does not evolve RSF state, propagate the fault, or solve the Stokes equations.

In ParaView, open `output-reconstructed-fault/solution.pvd` and color the bulk
mesh by `phase_field`. Then open
`output-reconstructed-fault/reconstructed_faults.pvd`.
The reconstructed line should follow the centerline of the diffuse band and
approximately overlay the prescribed segment from `(35 km, 24 km)` to
`(65 km, 76 km)`. Mesh-scale endpoint offsets are expected. The line output
contains only `fault_id`, `vertex_id`, and `cell_id` identifiers.
