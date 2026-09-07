#!/bin/sh
output=$("${0%/*}/cmake/default" "$@")
status=$?
test "$status" -eq 0 || exit 1
printf '%s\n' "$output" | grep -q 'Stage-J checkpoint histories, V, geometry, and bulk: verified' || exit 1
printf '%s\n' "$output" | grep -q 'Stage-J resumed feedback versus uninterrupted run: verified' || exit 1
printf 'Stage-J checkpoint preservation and resumed feedback: verified\n'
