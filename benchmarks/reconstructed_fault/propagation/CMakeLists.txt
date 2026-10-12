cmake_minimum_required(VERSION 3.16)
find_package(Aspect REQUIRED HINTS ${Aspect_DIR})
DEAL_II_INITIALIZE_CACHED_VARIABLES()
project(propagation_p1)
add_library(controlled_history SHARED ../../../tests/phase_field_controlled_history.cc)
ASPECT_SETUP_PLUGIN(controlled_history)
target_compile_options(controlled_history PRIVATE -fno-finite-math-only -ffp-contract=off)
add_library(bound_diagnostic SHARED bound_diagnostic.cc)
ASPECT_SETUP_PLUGIN(bound_diagnostic)
target_compile_options(bound_diagnostic PRIVATE -fno-finite-math-only -ffp-contract=off)

add_library(production_integration SHARED production_integration.cc)
ASPECT_SETUP_PLUGIN(production_integration)
target_compile_options(production_integration PRIVATE -fno-finite-math-only -ffp-contract=off)

add_library(mature_compat SHARED ../../../tests/phase_field_fault_boundary_completion.cc)
ASPECT_SETUP_PLUGIN(mature_compat)

add_library(coupled_feedback SHARED coupled_feedback.cc)
ASPECT_SETUP_PLUGIN(coupled_feedback)
target_compile_options(coupled_feedback PRIVATE -fno-finite-math-only -ffp-contract=off)

add_library(material_feedback SHARED material_feedback.cc)
ASPECT_SETUP_PLUGIN(material_feedback)
target_compile_options(material_feedback PRIVATE -fno-finite-math-only -ffp-contract=off)

add_library(history_driving SHARED history_driving.cc)
ASPECT_SETUP_PLUGIN(history_driving)
target_compile_options(history_driving PRIVATE -fno-finite-math-only -ffp-contract=off)

add_library(recoverable_feedback SHARED recoverable_feedback.cc recoverable_local.cc)
ASPECT_SETUP_PLUGIN(recoverable_feedback)
target_compile_options(recoverable_feedback PRIVATE -fno-finite-math-only -ffp-contract=off)

add_library(recoverable_projection SHARED recoverable_projection.cc)
ASPECT_SETUP_PLUGIN(recoverable_projection)
target_compile_options(recoverable_projection PRIVATE -fno-finite-math-only -ffp-contract=off)

add_library(coupling_feedback SHARED coupling_feedback.cc coupling_local.cc)
ASPECT_SETUP_PLUGIN(coupling_feedback)
target_compile_options(coupling_feedback PRIVATE -fno-finite-math-only -ffp-contract=off)

add_library(coupling_consistency SHARED coupling_consistency.cc)
ASPECT_SETUP_PLUGIN(coupling_consistency)
target_compile_options(coupling_consistency PRIVATE -fno-finite-math-only -ffp-contract=off)

add_library(irreversibility_feedback SHARED irreversibility_feedback.cc)
ASPECT_SETUP_PLUGIN(irreversibility_feedback)
target_compile_options(irreversibility_feedback PRIVATE -fno-finite-math-only -ffp-contract=off)

add_library(accuracy_stress SHARED accuracy_stress.cc)
ASPECT_SETUP_PLUGIN(accuracy_stress)
target_compile_options(accuracy_stress PRIVATE -fno-finite-math-only -ffp-contract=off)

add_library(native_fe_strain SHARED native_fe_strain.cc)
ASPECT_SETUP_PLUGIN(native_fe_strain)
target_compile_options(native_fe_strain PRIVATE -fno-finite-math-only -ffp-contract=off)

add_library(native_endpoint_verification SHARED native_endpoint_verification.cc)
ASPECT_SETUP_PLUGIN(native_endpoint_verification)
target_compile_options(native_endpoint_verification PRIVATE -fno-finite-math-only -ffp-contract=off)
