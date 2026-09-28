#!/bin/sh
# Numerical assertions carry the checks; retain only their deterministic result.
awk '/^State limiter:/ {print}'
