set pagination off
set confirm off
set print elements 24
set debuginfod enabled off
# Read-only diagnostics just before the coupled convergence check. No program
# variables, history, tolerances, directions or line-search limits are changed.
break /home/ein/repository/aspect/source/simulator/solver.cc:1351
commands
silent
printf "COUPLED_RESIDUAL_RECORD\n"
print this->timestep_number
print this->time
print nonlinear_iteration
print current_bulk_norm
print current_surface_norm
print bulk_scale
print surface_scale
print relative_bulk_residual
print relative_surface_residual
continue
end
run
quit $_exitcode
