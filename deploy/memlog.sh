#!/bin/bash
# Logs "unix-time used-MB" of this node once per second for $1 seconds (default 1200). Read-only.
#   bash memlog.sh 1200 > mem.log
end=$(( $(date +%s) + ${1:-1200} ))
while [ "$(date +%s)" -lt "$end" ]; do
    echo "$(date +%s.%N) $(free -m | awk '/Mem:/{print $3}')"
    sleep 1
done
