#!/bin/sh
if test "$1" = "screen-output"; then
  grep 'Particle projection cold/warm cache:'
else
  cat
fi
