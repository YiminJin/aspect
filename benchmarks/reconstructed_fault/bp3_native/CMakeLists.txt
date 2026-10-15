cmake_minimum_required(VERSION 3.13.4)
find_package(Aspect REQUIRED HINTS ${Aspect_DIR})
DEAL_II_INITIALIZE_CACHED_VARIABLES()
project(bp3_native_history_checks)
add_library(bp3_native_checks SHARED history_checks.cc)
ASPECT_SETUP_PLUGIN(bp3_native_checks)
target_compile_options(bp3_native_checks PRIVATE -fno-finite-math-only -ffp-contract=off)

# Unmodified repository controls for shared history/publication and completion.
add_library(legacy_history_checks SHARED ../../../tests/phase_field_fault_stage_j.cc)
ASPECT_SETUP_PLUGIN(legacy_history_checks)
add_library(boundary_completion_checks SHARED ../../../tests/phase_field_fault_boundary_completion.cc)
ASPECT_SETUP_PLUGIN(boundary_completion_checks)

# ASPECT loads plugins locally: this observer explicitly links the exact BP3
# library used in its inputs. Existing first-pass targets need no such library.
set(BP3_COUPLED_PLUGIN "" CACHE FILEPATH "Matching maintained BP3 library for the serial 7B observer")
if(BP3_COUPLED_PLUGIN)
  add_library(bp3_coupled_checks SHARED coupled_checks.cc)
  ASPECT_SETUP_PLUGIN(bp3_coupled_checks)
  target_link_libraries(bp3_coupled_checks "${BP3_COUPLED_PLUGIN}")
  option(BP3_TRANSITION_CHECK "Build the existing observer against the ordinary BP3 transition" OFF)
  if(NOT BP3_TRANSITION_CHECK)
    target_compile_definitions(bp3_coupled_checks PRIVATE ASPECT_BP3_LOCAL_BOTTOM_TEST)
  endif()
  target_compile_options(bp3_coupled_checks PRIVATE -fno-finite-math-only -ffp-contract=off)
endif()
