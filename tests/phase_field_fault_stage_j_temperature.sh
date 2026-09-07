#!/bin/sh
output=$("${0%/*}/cmake/default" "$@")
status=$?
test "$status" -eq 0 || exit 1
printf '%s\n' "$output" | grep -q 'Stage-J controlled transverse-temperature evaluation: verified' || exit 1
printf 'Stage-J controlled transverse-temperature evaluation: verified\n'
