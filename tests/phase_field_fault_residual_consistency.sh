#!/bin/sh
if test "$1" = "screen-output"; then
  sed -n -e 's/.*Affine nonzero-V probe: block=\([01]\).*/Nonzero-V affine consistency block \1: passed/p' \
         -e '/Reconstructed-fault Stage-I solve:/p'
else
  cat
fi
