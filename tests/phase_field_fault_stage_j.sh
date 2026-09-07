#!/bin/sh

output=$("${0%/*}/cmake/default" "$@" 2>&1)
status=$?

lifecycle_count=$(printf '%s\n' "$output" \
  | grep -Ec 'Reconstructed-fault Stage-I solve:[[:space:]]+verified')
feedback_count=$(printf '%s\n' "$output" \
  | grep -Ec 'Reconstructed-fault Stage-J history feedback:[[:space:]]+verified')
history_count=$(printf '%s\n' "$output" \
  | grep -c 'Cohesive traction profile variation for fault 0:')

printf 'Stage-J lifecycle postprocess count: %s\n' "$lifecycle_count"
printf 'Stage-J feedback postprocess count: %s\n' "$feedback_count"
printf 'Stage-J history commit diagnostics: %s\n' "$history_count"

test "$status" -eq 0 \
  && test "$lifecycle_count" -eq 3 \
  && test "$feedback_count" -eq 3 \
  && test "$history_count" -eq 2
