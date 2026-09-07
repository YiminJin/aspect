#!/bin/sh

"${0%/*}/cmake/default" "$@" \
  | grep '^    Rate-and-state reconstructed-fault friction requires'
