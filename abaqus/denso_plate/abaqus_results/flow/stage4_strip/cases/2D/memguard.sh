#!/bin/bash
# kill every openInjMoldSimF of this user on this node if their resident memory together passes GUARD_GB (default 16);
# a job's OOM takes the node's pbs_mom down (systemd OOMPolicy=stop, frontale01 4 Nov, frontale03 9 Nov)
lim=$(( ${GUARD_GB:-16} * 1048576 ))
while true; do
  rss=$(ps -u "$(id -u)" -o rss=,comm= | awk '$2 ~ /openInjMold/ {s += $1} END {print s + 0}')
  if [ "$rss" -gt "$lim" ]; then echo "memguard: ${rss} kB over ${GUARD_GB:-16} GB, killed $(date)"; pkill -9 -u "$(id -u)" openInjMoldSimF; fi
  sleep 10
done
