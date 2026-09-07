#!/bin/sh
output=$("${0%/*}/cmake/default" "$@")
status=$?
count=$(printf '%s\n' "$output" | grep -c 'Frozen-input phase-field residual and Jacobian: verified')
printf 'Frozen-input phase-field verification count: %s\n' "$count"
test "$status" -eq 0 && test "$count" -eq 3
