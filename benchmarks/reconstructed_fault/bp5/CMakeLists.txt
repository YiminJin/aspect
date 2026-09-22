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
