cmake_minimum_required(VERSION 3.16)
find_package(deal.II 9.6 REQUIRED HINTS /opt/dealii/9.6-local)
deal_ii_initialize_cached_variables()
project(bulk_tip_controls CXX)
get_filename_component(ASPECT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
add_executable(bulk_tip_points ../bulk_cohesive/points.cc
  ${ASPECT_SOURCE_DIR}/unit_tests/bulk_tip.cc
  ${ASPECT_SOURCE_DIR}/unit_tests/phase_field_dg_bounds.cc
  ${ASPECT_SOURCE_DIR}/unit_tests/bulk_cohesive.cc
  ${ASPECT_SOURCE_DIR}/source/material_model/phase_field_fault/bulk_tip.cc
  ${ASPECT_SOURCE_DIR}/source/material_model/phase_field_fault/bulk_cohesive.cc
  ${ASPECT_SOURCE_DIR}/source/material_model/phase_field_fault/bulk_cohesive_tensor.cc)
target_include_directories(bulk_tip_points PRIVATE ${ASPECT_SOURCE_DIR}/include ${ASPECT_SOURCE_DIR}/contrib/catch)
deal_ii_setup_target(bulk_tip_points)
target_compile_options(bulk_tip_points PRIVATE -fno-finite-math-only -ffp-contract=off)
if(Aspect_DIR)
  find_package(Aspect REQUIRED HINTS ${Aspect_DIR})
  add_library(bulk_tip_checks SHARED checks.cc)
  ASPECT_SETUP_PLUGIN(bulk_tip_checks)
  add_library(stress_advection_checks SHARED stress_advection_checks.cc ../support/serial_tensor_advection.cc)
  ASPECT_SETUP_PLUGIN(stress_advection_checks)
  target_compile_options(stress_advection_checks PRIVATE -fno-finite-math-only -ffp-contract=off)
endif()
