cmake_minimum_required(VERSION 3.13)
find_package(Aspect REQUIRED HINTS ${Aspect_DIR})
project(k1_initialization_diagnostics)
# Keep the included pilot input's postprocessor registered. Its phase-freeze
# signal is inactive at zero, and its postprocessor is not executed here.
add_library(k1_initialization_diagnostics SHARED diagnostic.cc ../uniform_shear.cc)
ASPECT_SETUP_PLUGIN(k1_initialization_diagnostics)
