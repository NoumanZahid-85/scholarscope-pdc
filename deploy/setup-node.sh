#!/bin/bash
# Run ONCE on EACH EC2 node (master, w1, w2) as the ubuntu user, after copying the 3 kit files to ~ :
#     bash setup-node.sh
# Installs Docker, puts the kit files in /data and downloads YOUR shards (read from config.json) into /data/shards.
set -e

# 1. Docker Engine + compose plugin (official install script); ubuntu may use docker without sudo after re-login
if ! command -v docker >/dev/null; then
    curl -fsSL https://get.docker.com | sudo sh
    sudo usermod -aG docker ubuntu
fi

# 2. /data with the 3 kit files, owned by ubuntu (uid 1000 = mpiuser inside the container)
sudo mkdir -p /data/shards
sudo chown -R ubuntu:ubuntu /data
cp ~/config.json ~/settings.txt ~/canary.jsonl /data/

# 3. Your shards, 16 downloads at a time, from the public bucket (same region = fast, free)
cd /data/shards
python3 -c "import json; print('\n'.join(f'shard-{s:03d}.jsonl.gz' for s in json.load(open('/data/config.json'))['shards']))" > /tmp/myshards
todo() { for f in $(cat /tmp/myshards); do gzip -t "$f" 2>/dev/null || echo "$f"; done; }   # missing or broken
todo | xargs -r -P 16 -I{} curl -sSf --retry 3 -o {} https://arxiv-shards.s3.amazonaws.com/{} || true

# 4. Check: every listed shard present and a valid gzip (re-running the script fetches only what is still missing)
bad=$(todo | wc -l)
echo "shards expected $(wc -l < /tmp/myshards), present $(ls /data/shards | wc -l), bad/missing $bad, total $(du -sh /data/shards | cut -f1)"
mkdir -p ~/mpi-ssh ~/scholar
[ "$bad" = 0 ] && echo "NODE READY" || echo "NOT READY: run 'bash setup-node.sh' again"
