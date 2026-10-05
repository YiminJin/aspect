cmake_minimum_required(VERSION 3.16)
find_package(deal.II 9.6 REQUIRED HINTS ${DEAL_II_DIR})
DEAL_II_INITIALIZE_CACHED_VARIABLES()
project(post_r6_installed_headers LANGUAGES CXX)

# Deliberately do not find the source-tree AspectConfig or add its include paths.
set(headers
  reconstructed_fault/normal_filter_internal.h
  reconstructed_fault/surface_direct_internal.h
  reconstructed_fault/surface_system_internal.h
  material_model/phase_field_fault/history_diagnostics.h
  simulator/solver/reconstructed_fault_bound_diagnostics.h
  simulator/solver/stokes_operators.h
  simulator/reconstructed_fault_interface_preconditioner.h
  simulator/reconstructed_fault_residual_audit.h)
set(sources)
foreach(header IN LISTS headers)
  get_filename_component(name ${header} NAME_WE)
  set(source "${CMAKE_CURRENT_BINARY_DIR}/${name}.cc")
  file(WRITE ${source} "#include <aspect/${header}>\nvoid check_${name}() {}\n")
  list(APPEND sources ${source})
endforeach()
add_library(installed_header_consumer SHARED ${sources})
target_include_directories(installed_header_consumer PRIVATE "${ASPECT_INSTALLED_INCLUDE_DIR}")
DEAL_II_SETUP_TARGET(installed_header_consumer)
