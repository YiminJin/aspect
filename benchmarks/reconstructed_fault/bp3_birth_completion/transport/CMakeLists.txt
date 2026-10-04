cmake_minimum_required(VERSION 3.13.4)
find_package(Aspect REQUIRED HINTS ${Aspect_DIR})
DEAL_II_INITIALIZE_CACHED_VARIABLES()
project(bp3_birth_identity_transport)
set(BP3 ../../bp3/plugin)
add_library(bp3_birth_transport SHARED audit.cc ../../bp3_local_tests/transport/transport.cc
  ${BP3}/geometry.cc ${BP3}/profile.cc ${BP3}/mesh.cc
  ${BP3}/particle_initialization.cc ${BP3}/particle_history.cc)
target_compile_definitions(bp3_birth_transport PRIVATE ASPECT_BP3_LOCAL_OSCILLATION_TEST)
ASPECT_SETUP_PLUGIN(bp3_birth_transport)
