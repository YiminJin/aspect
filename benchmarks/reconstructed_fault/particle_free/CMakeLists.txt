cmake_minimum_required(VERSION 3.16)
# Historical PF1--PF6 and replicated cohesive plugins use the retired Q1/nodal
# phase or serial material transport interfaces. Their sources/inputs/evidence
# are retained with the old executable; they are not active build dependencies.
# Configure bulk_tip/, dgq2_phase_field/ or transported_objective/native/ for
# maintained independent checks and current coupled controls.
project(particle_free_historical_checks NONE)
message(STATUS "Historical particle-free plugin umbrella retired; use the maintained per-control CMake projects.")
