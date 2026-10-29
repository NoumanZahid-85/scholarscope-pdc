#!/bin/bash
# Read-only health check of one EC2 node (changes nothing). On the node:  bash check-node.sh
ok()  { echo "  [OK]   $*"; }
bad() { echo "  [FAIL] $*"; }
echo "== $(hostname)"
[ "$(nproc)" -ge 4 ] && ok "vCPUs $(nproc)" || bad "vCPUs $(nproc) (need 4)"
mem=$(awk '/MemTotal/{printf "%d", $2/1024/1024}' /proc/meminfo); [ "$mem" -ge 7 ] && ok "memory ${mem} GB" || bad "memory ${mem} GB"
. /etc/os-release; ok "OS $PRETTY_NAME, disk $(df -h / | awk 'NR==2{print $2" total, "$4" free"}')"
command -v docker >/dev/null && ok "$(docker --version)" || bad "docker missing (run setup-node.sh)"
docker compose version >/dev/null 2>&1 && ok "$(docker compose version)" || bad "docker compose plugin missing"
id -nG | grep -qw docker && ok "ubuntu is in the docker group" || bad "ubuntu not in docker group (log out and in again)"
for f in config.json settings.txt canary.jsonl; do [ -s /data/$f ] && ok "/data/$f" || bad "/data/$f missing"; done
want=$(python3 -c "import json; print(len(json.load(open('/data/config.json'))['shards']))" 2>/dev/null)
have=$(ls /data/shards 2>/dev/null | wc -l); [ "$have" = "$want" ] && ok "shards $have/$want ($(du -sh /data/shards | cut -f1))" || bad "shards $have/$want"
[ -s ~/mpi-ssh/id_ed25519 ] && [ -s ~/mpi-ssh/authorized_keys ] && ok "~/mpi-ssh has key + authorized_keys" || bad "~/mpi-ssh incomplete"
[ -s ~/scholar/docker-compose.yml ] && ok "compose: $(grep -o 'ROLE: .*' ~/scholar/docker-compose.yml)" || bad "~/scholar/docker-compose.yml missing"
[ -s /data/hosts ] && ok "/data/hosts: $(grep -c slots /data/hosts) hosts" || echo "  [--]   no /data/hosts (only the master needs it)"
