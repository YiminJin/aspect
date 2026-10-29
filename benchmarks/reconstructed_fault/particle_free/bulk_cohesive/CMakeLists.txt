cmake_minimum_required(VERSION 3.16)
find_package(deal.II 9.6 REQUIRED HINTS /opt/dealii/9.6-local)
deal_ii_initialize_cached_variables()
project(bulk_cohesive_material_points CXX)
get_filename_component(ASPECT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
add_executable(bulk_cohesive_points points.cc
  ${ASPECT_SOURCE_DIR}/source/material_model/phase_field_fault/bulk_cohesive.cc
  ${ASPECT_SOURCE_DIR}/unit_tests/bulk_cohesive.cc)
target_include_directories(bulk_cohesive_points PRIVATE ${ASPECT_SOURCE_DIR}/include)
deal_ii_setup_target(bulk_cohesive_points)
target_compile_options(bulk_cohesive_points PRIVATE -fno-finite-math-only -ffp-contract=off)
add_library(bulk_strip_oracle SHARED strip_oracle.cc
  ${ASPECT_SOURCE_DIR}/source/material_model/phase_field_fault/bulk_cohesive.cc)
target_include_directories(bulk_strip_oracle PRIVATE ${ASPECT_SOURCE_DIR}/include)
deal_ii_setup_target(bulk_strip_oracle)
target_compile_options(bulk_strip_oracle PRIVATE -fno-finite-math-only -ffp-contract=off)
add_executable(bulk_cohesive_tensor_points tensor_points.cc
  ${ASPECT_SOURCE_DIR}/source/material_model/phase_field_fault/bulk_cohesive.cc
  ${ASPECT_SOURCE_DIR}/source/material_model/phase_field_fault/bulk_cohesive_tensor.cc)
target_include_directories(bulk_cohesive_tensor_points PRIVATE ${ASPECT_SOURCE_DIR}/include)
deal_ii_setup_target(bulk_cohesive_tensor_points)
target_compile_options(bulk_cohesive_tensor_points PRIVATE -fno-finite-math-only -ffp-contract=off)
add_executable(bulk_cohesive_coulomb_points coulomb_points.cc coulomb_point.cc
  ${ASPECT_SOURCE_DIR}/source/material_model/phase_field_fault/bulk_cohesive.cc
  ${ASPECT_SOURCE_DIR}/source/material_model/phase_field_fault/bulk_cohesive_tensor.cc)
target_include_directories(bulk_cohesive_coulomb_points PRIVATE ${ASPECT_SOURCE_DIR}/include)
deal_ii_setup_target(bulk_cohesive_coulomb_points)
target_compile_options(bulk_cohesive_coulomb_points PRIVATE -fno-finite-math-only -ffp-contract=off)
add_executable(coulomb_block_reference coulomb_block_reference.cc
  ${ASPECT_SOURCE_DIR}/source/material_model/phase_field_fault/bulk_cohesive_coulomb.cc
  ${ASPECT_SOURCE_DIR}/source/material_model/phase_field_fault/bulk_cohesive.cc
  ${ASPECT_SOURCE_DIR}/source/material_model/phase_field_fault/bulk_cohesive_tensor.cc)
target_include_directories(coulomb_block_reference PRIVATE ${ASPECT_SOURCE_DIR}/include)
deal_ii_setup_target(coulomb_block_reference)
target_compile_options(coulomb_block_reference PRIVATE -fno-finite-math-only -ffp-contract=off)
add_executable(coulomb_localization coulomb_localization.cc
  ${ASPECT_SOURCE_DIR}/source/material_model/phase_field_fault/bulk_cohesive_coulomb.cc
  ${ASPECT_SOURCE_DIR}/source/material_model/phase_field_fault/bulk_cohesive.cc
  ${ASPECT_SOURCE_DIR}/source/material_model/phase_field_fault/bulk_cohesive_tensor.cc)
target_include_directories(coulomb_localization PRIVATE ${ASPECT_SOURCE_DIR}/include)
deal_ii_setup_target(coulomb_localization)
target_compile_options(coulomb_localization PRIVATE -fno-finite-math-only -ffp-contract=off)
add_library(coulomb_gradient_point SHARED gradient_bridge.cc gradient_return.cc
  ${ASPECT_SOURCE_DIR}/source/material_model/phase_field_fault/bulk_cohesive_gradient.cc
  ${ASPECT_SOURCE_DIR}/source/material_model/phase_field_fault/bulk_cohesive_coulomb.cc
  ${ASPECT_SOURCE_DIR}/source/material_model/phase_field_fault/bulk_cohesive.cc
  ${ASPECT_SOURCE_DIR}/source/material_model/phase_field_fault/bulk_cohesive_tensor.cc)
target_include_directories(coulomb_gradient_point PRIVATE ${ASPECT_SOURCE_DIR}/include)
deal_ii_setup_target(coulomb_gradient_point)
target_compile_options(coulomb_gradient_point PRIVATE -fno-finite-math-only -ffp-contract=off)
add_executable(coulomb_gradient_space gradient_space.cc)
target_include_directories(coulomb_gradient_space PRIVATE ${ASPECT_SOURCE_DIR}/include)
target_link_libraries(coulomb_gradient_space coulomb_gradient_point)
deal_ii_setup_target(coulomb_gradient_space)
target_compile_options(coulomb_gradient_space PRIVATE -fno-finite-math-only -ffp-contract=off)
