#!/bin/bash
# score each config vs teammate (seeds 5-10) and v1 (seeds 34-39), 2 jobs at a time
for cfg in "$@"; do
  ( a=$(python tools/h2h.py /home/claude/teammate/repo --seeds 5 6 7 8 9 10 --params "$cfg" | tail -1 | awk '{print $NF}');
    b=$(python tools/h2h.py /home/claude/v1 --seeds 34 35 36 37 38 39 --params "$cfg" | tail -1 | awk '{print $NF}');
    echo "$cfg  teammate $a  v1 $b" ) &
  while [ $(jobs -r | wc -l) -ge 1 ]; do sleep 1; done
done
wait
