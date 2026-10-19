################### top matter ################
cmake_minimum_required(VERSION 3.13.4)

project(unittests CXX)

set(ASPECT_BINARY "${ASPECT_BINARY}" CACHE STRING "" FORCE)

enable_testing()

message(STATUS "creating unit_tests ctest in ${CMAKE_CURRENT_BINARY_DIR} binary=${ASPECT_BINARY}")

add_test(NAME unit_tests
         COMMAND ${ASPECT_BINARY} --test
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      )

# Exercise the Stage C adaptive profile collectives with unequal profile
# ownership. The tagged tests also run in serial as part of unit_tests.
add_test(NAME phase_field_fault_ih_accuracy_mpi
         COMMAND mpirun -np 2 ${ASPECT_BINARY} --test [phase_field_fault_ih_accuracy]
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      )
set_tests_properties(phase_field_fault_ih_accuracy_mpi PROPERTIES PROCESSORS 2)

# One prescribed append, without advancing physical time. Exercise native DG
# ownership/ghosts and the synthetic atomic geometry/history transition only.
add_test(NAME reconstructed_fault_extension_mpi
         COMMAND mpirun -np 2 ${ASPECT_BINARY} --test [p3_extension]
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(reconstructed_fault_extension_mpi PROPERTIES
                    PROCESSORS 2 TIMEOUT 300)
