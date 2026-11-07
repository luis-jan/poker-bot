#!/bin/bash
# usage: tools/batch.sh OTHER "seeds" cfg1 cfg2 ...   (runs 2 at a time)
OTHER=$1; SEEDS=$2; shift 2
for cfg in "$@"; do
  while [ $(jobs -r | wc -l) -ge 2 ]; do sleep 1; done
  python tools/h2h.py "$OTHER" --seeds $SEEDS --params "$cfg" | tail -1 &
done
wait
