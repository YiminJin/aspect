#!/bin/sh

if test "$1" = "screen-output"; then
  grep "Adiabatic fault-friction pressure requires initialized adiabatic" | tail -1
else
  cat
fi
