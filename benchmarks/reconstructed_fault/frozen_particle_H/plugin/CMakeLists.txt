cmake_minimum_required(VERSION 3.13.4)
find_package(Aspect REQUIRED HINTS ${Aspect_DIR})
DEAL_II_INITIALIZE_CACHED_VARIABLES()
project(frozen_particle_H)
add_library(frozen_birth_observer SHARED
  ../../particle_lifecycle/plugin/coupled.cc birth_check.cc)
ASPECT_SETUP_PLUGIN(frozen_birth_observer)
target_link_libraries(frozen_birth_observer ${CMAKE_CURRENT_BINARY_DIR}/../bp3/libbp3_restore_150x50.release.so)
# Standalone guard loads the maintained property implementation, without the
# frozen BP3 simulator callbacks (which intentionally reject evolving BP3).
add_library(evolving_birth_guard SHARED
  ../../bp3/plugin/particle_initialization.cc evolving_guard.cc)
ASPECT_SETUP_PLUGIN(evolving_birth_guard)
