#!/bin/sh
output=$("${0%/*}/cmake/default" "$@" 2>&1 || true)
printf '%s\n' "$output" | grep -q 'Fault linear solve: iterations=1,' || exit 1
if printf '%s\n' "$output" | grep -q 'Reconstructed-fault line search accepted'; then exit 1; fi
printf '%s\n' "$output" | grep -q '^Stage-I rollback after an accepted Newton update: verified$' || exit 1
printf '%s\n' 'Fault linear iteration budget and rollback: verified'
