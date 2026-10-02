cmake_minimum_required(VERSION 3.13.4)
find_package(Aspect REQUIRED HINTS ${Aspect_DIR})
DEAL_II_INITIALIZE_CACHED_VARIABLES()
project(particle_replenishment)
add_library(replenishment SHARED replenishment.cc)
ASPECT_SETUP_PLUGIN(replenishment)
add_library(coupled_replenishment SHARED coupled.cc)
ASPECT_SETUP_PLUGIN(coupled_replenishment)
# ASPECT loads plugins with local symbol scope. Explicit dependency keeps the
# fixture-only H-audit adapter bound to the accepted maintained BP3 library.
target_link_libraries(coupled_replenishment ${Aspect_DIR}/../benchmarks/reconstructed_fault/refactoring_r6b/plugin-build/bp3/libbp3_restore_150x50.release.so)
