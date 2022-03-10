#!/bin/sh

# Dataset cleaning step
# This file receives as input multiple files and merges them and fixes any
# errors in the characters' encoding.

[ -z "$1" ] && echo "Usage: $0 FILE [FILE...]" >&2 && exit 1

head -n1 "$1"
tail -q -n+2 "$@" | \
  LC_ALL=C sed 's/\xc3\x83\xc2\x85/\xc3\x85/g;s/\xc3\x83\xc2\x98/\xc3\x98/g' # Fix Å (0xc383) and Ø (0xc398)
