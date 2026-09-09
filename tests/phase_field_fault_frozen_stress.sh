#!/bin/sh
output=$("${0%/*}/cmake/default" "$@" 2>&1) || { printf '%s\n' "$output"; exit 1; }
printf '%s\n' "$output" | grep 'Frozen Maxwell bulk load:'
