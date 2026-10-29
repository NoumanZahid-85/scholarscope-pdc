#!/bin/bash
# Laptop benchmark: copies data into the container's own disk (the Windows share is slow), compiles, times 1/2/4 threads.
#   docker run --rm -u root -v "${PWD}:/work" -v "${PWD}\data:/src:ro" pdc-dev bash /work/bench.sh 0 1 2 3 4 6
mkdir -p /data && cp -r /src/* /data/ && chown -R mpiuser /data
su mpiuser -c '
cd /work/engine && mpicxx -O3 -fopenmp -std=c++17 -Wall engine.cpp -o /tmp/engine -lz || exit 1
for t in 1 2 4; do
  echo "threads=$t"; /usr/bin/time -f "  wall %e s  cpu %U+%S s  maxrss %M KB" env OMP_NUM_THREADS=$t /tmp/engine index '"$*"' >/dev/null
done
echo "mpirun -np 3 x threads=4 (one machine, so this only shows it works; real gains come on 3 EC2 nodes)"
/usr/bin/time -f "  wall %e s" env OMP_NUM_THREADS=4 mpirun -np 3 --oversubscribe -x OMP_NUM_THREADS /tmp/engine index '"$*"' >/dev/null'
