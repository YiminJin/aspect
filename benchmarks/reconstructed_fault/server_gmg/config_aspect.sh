#!/bin/bash

cmake \
  -D CMAKE_CXX_COMPILER=icpx \
  -D DEAL_II_DIR="$DEAL_II_DIR" \
  -D ASPECT_WITH_VORO=ON \
  -D VORO_DIR="$VORO_ROOT" \
  -D ASPECT_ADDITIONAL_CXX_FLAGS="-fno-finite-math-only -fp-model=precise -ffp-contract=off" \
  ..
