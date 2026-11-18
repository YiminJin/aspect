cmake_minimum_required(VERSION 3.16)
find_package(deal.II 9.6 REQUIRED HINTS /opt/dealii/9.6-local)
deal_ii_initialize_cached_variables()
project(single_turn_reference CXX)
get_filename_component(ASPECT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
# An oracle for the independent reference, not a second simulator driver.
add_executable(tip_point_oracle point_oracle.cc
  ${ASPECT_SOURCE_DIR}/source/material_model/phase_field_fault/bulk_tip.cc)
target_include_directories(tip_point_oracle PRIVATE ${ASPECT_SOURCE_DIR}/include)
deal_ii_setup_target(tip_point_oracle)
target_compile_options(tip_point_oracle PRIVATE -fno-finite-math-only -ffp-contract=off)
