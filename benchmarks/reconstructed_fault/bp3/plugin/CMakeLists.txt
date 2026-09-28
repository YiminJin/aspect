cmake_minimum_required(VERSION 3.13.4)
find_package(Aspect REQUIRED HINTS ${Aspect_DIR})
DEAL_II_INITIALIZE_CACHED_VARIABLES()
project(reconstructed_fault_bp3_restore)

# This directory is the complete runtime source package. No mesh generators,
# test executables, BP5 variants, or sibling-directory sources are required.
add_library(bp3_restore_150x50 SHARED
  bp3.cc
  mesh.cc
  monitor.cc
  output.cc
  work_audit.cc
)
ASPECT_SETUP_PLUGIN(bp3_restore_150x50)
