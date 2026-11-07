cmake_minimum_required(VERSION 3.13.4)
find_package(Aspect REQUIRED HINTS ${Aspect_DIR})
DEAL_II_INITIALIZE_CACHED_VARIABLES()
project(dgq2_phase_checks CXX)
find_package(MPI REQUIRED COMPONENTS CXX)
get_filename_component(REPOSITORY "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
add_executable(dgq2_phase_checks operator_checks.cc
  "${REPOSITORY}/source/simulator/assemblers/scalar_dg.cc"
  "${REPOSITORY}/source/phase_field/dg_transport.cc"
  "${REPOSITORY}/source/phase_field/dg_sipg.cc"
  "${REPOSITORY}/source/phase_field/obstacle.cc")
target_include_directories(dgq2_phase_checks PRIVATE ${Aspect_INCLUDE_DIRS})
deal_ii_setup_target(dgq2_phase_checks RELEASE)
target_link_libraries(dgq2_phase_checks MPI::MPI_CXX)
target_compile_options(dgq2_phase_checks PRIVATE -fno-finite-math-only -ffp-contract=off)
add_library(dg_native_phase_checks SHARED native_checks.cc)
ASPECT_SETUP_PLUGIN(dg_native_phase_checks)
target_compile_options(dg_native_phase_checks PRIVATE -fno-finite-math-only -ffp-contract=off)
