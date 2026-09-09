set pagination off
set confirm off
set breakpoint pending on
set print elements 24
set debuginfod enabled off
set $accepted_zero = 0
# The helper is entered for each Newton direction; ignore initial mechanics.
break /home/ein/repository/aspect/source/simulator/solver.cc:1411 if $accepted_zero
commands
silent
printf "SEARCH_BASE\n"
print nonlinear_iteration
print maximum_step_length
print bulk_scale
print surface_scale
print current_bulk_norm
print current_surface_norm
print current_merit
print slip_rate
print slip_rate_direction
print active_set
printf "BASE_SURFACE\n"
print surface_system.surface_linearization->residual.values
printf "SURFACE_MASS_DIAGONAL\n"
print surface_system.surface_linearization->mass_diagonal
printf "SURFACE_MASS_OFF_DIAGONAL\n"
print surface_system.surface_linearization->mass_off_diagonal
continue
end
break /home/ein/repository/aspect/source/simulator/solver.cc:1473 if $accepted_zero
commands
silent
printf "SEARCH_TRIAL\n"
print step_length
print trial_residual.bulk_norm
print trial_relative_bulk
print trial_relative_surface
print trial_merit
print current_merit
print trial_slip_rate
printf "TRIAL_SURFACE\n"
print trial_residual.surface.values
continue
end
break /home/ein/repository/aspect/source/simulator/solver.cc:1496 if $accepted_zero
commands
silent
printf "SEARCH_EXHAUSTION\n"
print line_search_result
continue
end
# At postprocessing all initial mechanical/state exports have completed.
break /home/ein/repository/aspect/benchmarks/reconstructed_fault/uniform_shear/uniform_shear.cc:229
commands
silent
set $accepted_zero = 1
printf "INITIAL_EXPORT_COMPLETE\n"
continue
end
run
