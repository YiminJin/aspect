cmake_minimum_required(VERSION 3.13.4)
find_package(Aspect 3.1 REQUIRED HINTS ${Aspect_DIR} $ENV{ASPECT_DIR})
DEAL_II_INITIALIZE_CACHED_VARIABLES()
project(explicit_dg_compositions)

# Build the installed slab observer from the CTest source and ship that source
# beside the benchmark input and checker.
set(probe "${CMAKE_CURRENT_SOURCE_DIR}/explicit_dg_compositions.cc")
if(NOT EXISTS "${probe}")
  set(probe "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/explicit_dg_compositions.cc")
  install(FILES "${probe}" DESTINATION benchmarks/explicit_dg_transport COMPONENT examples)
endif()
add_library(explicit_dg_compositions SHARED "${probe}")
ASPECT_SETUP_PLUGIN(explicit_dg_compositions)
