#!/bin/sh

if test "$1" = "screen-output"; then
  grep "Reconstructed-fault surface system:"
else
  cat
fi
