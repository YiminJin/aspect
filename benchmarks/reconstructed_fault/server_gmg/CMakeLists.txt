cmake_minimum_required(VERSION 3.13.4)
find_package(Aspect REQUIRED HINTS ${Aspect_DIR})
DEAL_II_INITIALIZE_CACHED_VARIABLES()
project(fault_gmg_server_debug)

# Reuse the production coupled lifecycle verifier, not a BP3 model copy.
add_library(fault_gmg_stage_i SHARED ../../../tests/phase_field_fault_stage_i.cc)
ASPECT_SETUP_PLUGIN(fault_gmg_stage_i)

add_library(fault_gmg_coarse_probe SHARED coarse_diagonal_probe.cc)
ASPECT_SETUP_PLUGIN(fault_gmg_coarse_probe)

add_executable(check_gmg_diagnostics check_diagnostics.cc)
DEAL_II_SETUP_TARGET(check_gmg_diagnostics)
