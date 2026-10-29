#!/bin/bash
# Container start-up (runs as root): give mpiuser the mounted folders, start sshd on port 2222, and on the master
# (ROLE=master) run the web server as mpiuser. A worker only keeps sshd running in the foreground.
mkdir -p /run/sshd /data
chown mpiuser:mpiuser /data 2>/dev/null
if [ -d /home/mpiuser/.ssh ]; then                 # sshd and ssh refuse keys that others can read or write
    chown -R mpiuser:mpiuser /home/mpiuser/.ssh 2>/dev/null
    chmod 700 /home/mpiuser/.ssh
    chmod 600 /home/mpiuser/.ssh/* 2>/dev/null
fi

if [ "$ROLE" = "master" ]; then
    /usr/sbin/sshd -e || exit 1
    exec runuser -u mpiuser -- python3 /opt/server.py --port 8000
fi
exec /usr/sbin/sshd -D -e
