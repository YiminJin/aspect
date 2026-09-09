#!/bin/sh
output=$("${0%/*}/cmake/default" "$@" 2>&1 || true)
printf '%s\n' "$output" | grep -q 'Reconstructed-fault line search accepted' || exit 1
printf '%s\n' "$output" | grep '^Stage-I rollback after an accepted Newton update: verified$' | sort -u
