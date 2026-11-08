#!/bin/bash
# CYH (8 seeds) + teammate (4) + Royal Flush (4); 2 configs in parallel
CY=/home/claude/scouted/yahwehc/td26-poker-bot-main; TM=/home/claude/teammate/repo; RF=/home/claude/scouted/SahithReddy-GitHub_poker-bot-tamu-datathon
run() { c=$(python tools/h2h.py $CY --seeds 805 806 807 808 809 810 811 812 --params "$1" 2>/dev/null | tail -1 | awk '{print $NF}');
        t=$(python tools/h2h.py $TM --seeds 5 6 7 8 --params "$1" 2>/dev/null | tail -1 | awk '{print $NF}');
        r=$(python tools/h2h.py $RF --seeds 201 202 203 204 --params "$1" 2>/dev/null | tail -1 | awk '{print $NF}');
        echo "$1  CYH $c  teammate $t  royalflush $r"; }
for cfg in "$@"; do
  while [ $(jobs -r | wc -l) -ge 2 ]; do sleep 2; done
  run "$cfg" &
done
wait
