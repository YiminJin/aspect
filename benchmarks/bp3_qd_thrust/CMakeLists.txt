cmake_minimum_required(VERSION 3.13.4)
find_package(Aspect 3.1 REQUIRED HINTS ${Aspect_DIR} $ENV{ASPECT_DIR})
DEAL_II_INITIALIZE_CACHED_VARIABLES()
project(bp3_qd)
add_library(bp3_qd SHARED bp3_qd.cc)
ASPECT_SETUP_PLUGIN(bp3_qd)

# Ship the existing event definition with the installed benchmark.
if(ASPECT_INSTALL_EXAMPLES)
  install(FILES "${CMAKE_SOURCE_DIR}/refactoring/phase-field-fault/results/bp3-preparation/history_events.py"
          DESTINATION benchmarks/bp3_qd_thrust COMPONENT examples)
endif()
