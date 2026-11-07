#!/bin/bash
# score configs vs teammate (seeds 5-10) and Royal Flush (201-206); 2 configs in parallel
TM=/home/claude/teammate/repo; RF=/home/claude/scouted/SahithReddy-GitHub_poker-bot-tamu-datathon
run() { a=$(python tools/h2h.py $TM --seeds 5 6 7 8 9 10 --params "$1" | tail -1 | awk '{print $NF}');
        b=$(python tools/h2h.py $RF --seeds 201 202 203 204 205 206 --params "$1" | tail -1 | awk '{print $NF}');
        echo "$1  teammate $a  royalflush $b  sum $(python -c "print(f'{$a+$b:+.3f}')")"; }
for cfg in "$@"; do
  while [ $(jobs -r | wc -l) -ge 2 ]; do sleep 2; done
  run "$cfg" &
done
wait
