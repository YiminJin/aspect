cmake_minimum_required(VERSION 3.13.4)
find_package(Aspect REQUIRED HINTS ${Aspect_DIR})
DEAL_II_INITIALIZE_CACHED_VARIABLES()
project(bp5_initialization)
add_library(bp5_initialization SHARED ../bp3/bp3.cc startup_time_step.cc)
target_compile_definitions(bp5_initialization PRIVATE ASPECT_BP5_WEAK_INITIALIZATION)
ASPECT_SETUP_PLUGIN(bp5_initialization)

# A separate plugin keeps old inverse-state fixtures reproducible. Never load
# both plugins: they register the same benchmark names with different initial data.
add_library(bp5_steady_initialization SHARED ../bp3/bp3.cc startup_time_step.cc)
target_compile_definitions(bp5_steady_initialization PRIVATE ASPECT_BP5_STEADY_INITIALIZATION)
ASPECT_SETUP_PLUGIN(bp5_steady_initialization)

# Initialization-only normal-feedback control; never used by the server fixture.
add_library(bp5_normal_control SHARED EXCLUDE_FROM_ALL ../bp3/bp3.cc startup_time_step.cc)
target_compile_definitions(bp5_normal_control PRIVATE ASPECT_BP5_STEADY_INITIALIZATION ASPECT_BP5_NORMAL_CONTROL)
ASPECT_SETUP_PLUGIN(bp5_normal_control)

add_library(bp5_normal_stress_diagnostic SHARED EXCLUDE_FROM_ALL normal_stress_diagnostic.cc normal_filter_clock.cc)
ASPECT_SETUP_PLUGIN(bp5_normal_stress_diagnostic)

add_library(bp5_stress_cycle SHARED EXCLUDE_FROM_ALL stress_cycle_diagnostic.cc)
ASPECT_SETUP_PLUGIN(bp5_stress_cycle)

add_library(bp5_clean_stress_cycle SHARED EXCLUDE_FROM_ALL clean_stress_cycle.cc)
ASPECT_SETUP_PLUGIN(bp5_clean_stress_cycle)

add_library(bp5_moment_cycle SHARED EXCLUDE_FROM_ALL moment_cycle.cc stress_only_interpolator.cc)
ASPECT_SETUP_PLUGIN(bp5_moment_cycle)

# Offline traction audit: call the actual production projection solver during
# --validate. No Simulator, mechanics, or history evolution is constructed.
add_library(test_traction_projection SHARED EXCLUDE_FROM_ALL test_traction_projection.cc)
ASPECT_SETUP_PLUGIN(test_traction_projection)

add_executable(test_normal_stress_clock EXCLUDE_FROM_ALL test_normal_stress_clock.cc)
target_compile_features(test_normal_stress_clock PRIVATE cxx_std_17)
add_library(test_restart_clock SHARED EXCLUDE_FROM_ALL test_restart_clock.cc)
ASPECT_SETUP_PLUGIN(test_restart_clock)
