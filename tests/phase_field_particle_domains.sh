#!/bin/sh
output=$("${0%/*}/cmake/default" "$@")
status=$?
test "$status" -eq 0 || exit 1
test "$(printf '%s\n' "$output" | grep -Ec 'Particle-domain regeneration:[[:space:]]+verified')" -eq 2 || exit 1
printf 'Particle-domain regeneration after advection: verified\n'
