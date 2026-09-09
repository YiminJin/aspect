#!/bin/sh
if test "$1" = "screen-output"; then
  grep "Reconstructed-fault surface and bulk coupling:"
else
  cat
fi
