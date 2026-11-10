#!/bin/zsh
# Phase 15: run the Allen fetch (resumable) and restart it when its log has not changed for 12 min (hung stream read).
# Then fetch Gabor receptive-field positions. Usage: scripts/phase15/fetch_watchdog.sh
LOG=data/phase15/allen_fetch15.log
for attempt in 1 2 3 4 5 6; do
  data/allen-venv/bin/python scripts/phase15/fetch_allen15.py fetch >> $LOG 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 30
    age=$(( $(date +%s) - $(stat -f %m $LOG) ))
    if [ $age -gt 720 ]; then echo "watchdog: no progress for ${age}s, restarting (attempt $attempt)" >> $LOG; kill $pid; sleep 2; kill -9 $pid 2>/dev/null; touch $LOG; break; fi
  done
  wait $pid; rc=$?
  [ $rc -eq 0 ] && break
done
data/allen-venv/bin/python scripts/phase15/fetch_gabor15.py > data/phase15/gabor15.log 2>&1
echo "fetch-all-done" >> $LOG
