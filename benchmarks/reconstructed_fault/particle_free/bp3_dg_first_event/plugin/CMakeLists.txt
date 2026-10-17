cmake_minimum_required(VERSION 3.13.4)
find_package(Aspect REQUIRED HINTS ${Aspect_DIR})
DEAL_II_INITIALIZE_CACHED_VARIABLES()
project(bp3_dg_first_event CXX)

if(NOT ASPECT_BUILD_RELEASE OR ASPECT_BUILD_DEBUG)
  message(FATAL_ERROR "Build the supplied ASPECT source with CMAKE_BUILD_TYPE=Release.")
endif()
get_filename_component(_aspect_parent "${Aspect_DIR}" DIRECTORY)
set(BP3_ASPECT_SOURCE "${_aspect_parent}/source" CACHE PATH "Matching supplied ASPECT source tree")
set(_native "${BP3_ASPECT_SOURCE}/benchmarks/reconstructed_fault/bp3_native")
if(NOT EXISTS "${_native}/coupled_checks.cc")
  message(FATAL_ERROR "Set BP3_ASPECT_SOURCE to the supplied ASPECT source tree.")
endif()
option(BP3_LOCAL_OSCILLATION_TEST "Local diagnostic fixture; forbidden in this package" OFF)
if(BP3_LOCAL_OSCILLATION_TEST)
  message(FATAL_ERROR "This package uses the ordinary BP3 physical parameters.")
endif()

# Use the maintained ordinary BP3 implementation, not a copied formulation.
add_subdirectory("${_native}/server/plugin" "${CMAKE_CURRENT_BINARY_DIR}/native")
add_library(bp3_coupled_checks SHARED "${_native}/coupled_checks.cc")
ASPECT_SETUP_PLUGIN(bp3_coupled_checks)
target_link_libraries(bp3_coupled_checks bp3_restore_150x50)
if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
  target_compile_options(bp3_coupled_checks PRIVATE -fno-finite-math-only -fp-model=precise -ffp-contract=off)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(bp3_coupled_checks PRIVATE -fno-finite-math-only -ffp-contract=off)
endif()

# Objects/caches remain in plugin/build. Relative runtime paths are in plugin/.
# This permits the requested direct 'ibrun ./plugin/aspect-release' launch.
foreach(_target bp3_restore_150x50 bp3_coupled_checks)
  set_target_properties(${_target} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN")
endforeach()
add_custom_target(bp3_runtime_path ALL
  COMMAND ${CMAKE_COMMAND} -E create_symlink "${Aspect_DIR}/aspect-release" "${CMAKE_CURRENT_SOURCE_DIR}/aspect-release"
  DEPENDS bp3_restore_150x50 bp3_coupled_checks
  COMMENT "Expose the matching Release executable at plugin/aspect-release")
