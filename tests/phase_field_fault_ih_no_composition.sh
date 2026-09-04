#!/bin/sh

if test "$1" = "screen-output"; then
  grep "Distributed I_h:"
else
  cat
fi
