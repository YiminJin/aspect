#!/bin/sh
output=$("${0%/*}/cmake/default" "$@")
printf '%s\n' "$output" | grep 'Phase-field nonlinear iteration budget exhausted:' || exit 1
printf '%s\n' "$output" | grep 'Aborting simulation as requested.' || exit 1
! printf '%s\n' "$output" | grep -q 'Reconstruction initial faults'
