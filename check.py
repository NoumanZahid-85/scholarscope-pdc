#!/usr/bin/env python3
"""
Correctness check: runs the C++ engine (index + search) and ref.py on the same shards and queries, compares top-10.
Ids must match exactly and in order; scores to 1e-6 relative (the grader's tolerance). Also checks the progress output.
Run INSIDE the container (needs /data and the engine):

    python3 check.py --shards 0 1                                  # sequential
    python3 check.py --shards 0 1 --threads 4                      # OpenMP
    python3 check.py --shards 0 1 --threads 2 --launch "mpirun -np 3 --oversubscribe"   # MPI
"""
import argparse, json, os, shlex, subprocess, sys, time
import ref

ap = argparse.ArgumentParser()
ap.add_argument("--shards", type=int, nargs="+", required=True)
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--launch", default="", help='e.g. "mpirun -np 3 --oversubscribe"')
ap.add_argument("--engine", default="/work/engine/engine")
ap.add_argument("--data", default="/data")
a = ap.parse_args()

s = ref.load_settings(os.path.join(a.data, "settings.txt"))
queries = json.load(open(os.path.join(a.data, "config.json")))["search_queries"] + ref.EDGE_QUERIES
t0 = time.time(); docs = ref.load_docs(a.data, a.shards); N, df, post = ref.build(docs, s)
print(f"ref.py built N={N} in {time.time() - t0:.1f}s")

env = dict(os.environ, OMP_NUM_THREADS=str(a.threads)); pre = shlex.split(a.launch)
t0 = time.time()
r = subprocess.run(pre + [a.engine, "index"] + [str(x) for x in a.shards], env=env, capture_output=True, text=True)
print(f"engine index: exit {r.returncode} in {time.time() - t0:.2f}s   stderr: {r.stderr.strip()[-300:]}")
lines = r.stdout.split()
ok = r.returncode == 0 and all(x.isdigit() for x in lines) and lines and lines[0] == "0" and lines[-1] == "100" \
     and all(int(x) <= int(y) for x, y in zip(lines, lines[1:]))
print(f"progress: {len(lines)} lines {'OK' if ok else 'BAD: ' + repr(r.stdout[:200])}")
fails = 0 if ok else 1

for q in queries:
    r = subprocess.run(pre + [a.engine, "search", q], env=env, capture_output=True, text=True)
    got = [l.split() for l in r.stdout.splitlines()]
    want = ref.search(q, docs, N, df, post, s)
    good = r.returncode == 0 and len(got) == len(want) and all(
        len(g) == 2 and g[0] == w[0] and abs(float(g[1]) - w[1]) <= 1e-6 * abs(w[1]) for g, w in zip(got, want))
    fails += not good
    print(f"{'OK  ' if good else 'FAIL'} {q!r:45} {len(want)} results" + ("" if good else f"\n  want {want}\n  got  {got}\n  err  {r.stderr.strip()}"))

print("ALL OK" if fails == 0 else f"{fails} FAILED"); sys.exit(fails > 0)
