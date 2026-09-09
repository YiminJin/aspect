#!/bin/sh
output=$("${0%/*}/cmake/default" "$@" 2>&1) || { printf '%s\n' "$output"; exit 1; }
printf '%s\n' "$output" | grep -E '^(First coupled iterate boundary error|.*Reconstructed-fault prescribed velocity:)'
