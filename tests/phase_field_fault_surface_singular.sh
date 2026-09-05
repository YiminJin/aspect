#!/bin/sh

if test "$1" = "screen-output"; then
  grep "Verified Stage-F singular K_V factorization diagnostic." | tail -1
else
  cat
fi
