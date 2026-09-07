#!/bin/sh

output=$("${0%/*}/cmake/default" "$@" 2>&1 || true)

printf '%s\n' "$output" \
  | grep '^      Reconstructed-fault line search accepted after 0 rejected candidates\.$'
printf '%s\n' "$output" \
  | grep '^Stage-I rollback after an accepted Newton update: verified$'

if printf '%s\n' "$output" \
   | grep -q 'Coupled reconstructed-fault Newton line search exhausted'; then
  exit 1
fi
