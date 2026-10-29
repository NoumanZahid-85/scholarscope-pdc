#!/usr/bin/env python3
"""
Benchmark sweep through server.py's API (everything measured by the SERVER, not by the engine). ~12 min on 3 x m5.xlarge.
Never run it while the grader may be using the cluster: it occupies the server and replaces the index.

    python bench_api.py --url http://<master>:8000            -> results/bench.json (re-used by make_charts.py)

Stored per run: request, wall/cpu seconds, OS threads, engine processes, number of DISTINCT machines (no ids/hostnames),
the progress curve, and the latency + top results of a few searches.
"""
import argparse, json, os, time, urllib.error, urllib.request

QUERIES = ["graph neural network", "gravitational wave binary merger", "zq5109237"]

def call(url, path, body=None):
    req = urllib.request.Request(url + path, method="GET" if body is None else "POST", headers={"Content-Type": "application/json"},
                                 data=None if body is None else json.dumps(body).encode())
    try:
        with urllib.request.urlopen(req, timeout=900) as r: return r.status, json.load(r)
    except urllib.error.HTTPError as e: return e.code, json.load(e)

def run(url, group, mode, shards, threads=1, ranks=1):
    body = {"shards": shards, "threads": threads, "ranks": ranks}
    t_start = time.time()
    code, j = call(url, f"/api/run/{mode}", body)
    if code != 200: raise SystemExit(f"{mode}: HTTP {code} {j}")
    while call(url, f"/api/status?job={j['job']}")[1]["state"] == "running": time.sleep(0.5)
    r = call(url, f"/api/result?job={j['job']}")[1]
    if r["state"] != "done": raise SystemExit(f"{mode}: {r['state']} {r.get('error')}")
    out = {"group": group, "mode": mode, "shards": len(shards), "ranks": r["req"]["ranks"], "threads": r["req"]["threads"],
           "wall": r["wall_sec"], "cpu_master": r["server_cpu_sec"], "threads_max": r["observed"]["threads_max"],
           "procs_max": r["observed"]["engine_procs_max"], "machines": len({c.get("iid") for c in r["cluster"] or []}) or 1,
           "progress": r["progress"], "t_start": t_start, "searches": []}
    for q in QUERIES:
        _, s = call(url, "/api/search", {"q": q})
        out["searches"].append({"q": q, "latency_ms": s.get("latency_ms"), "results": s.get("results")})
    print(f"{group:8} {mode:3} {len(shards):3} shards {out['ranks']}x{out['threads']}: wall {out['wall']:7.2f}s "
          f"cpu {out['cpu_master']:7.2f}s threads {out['threads_max']} machines {out['machines']} "
          f"search {[x['latency_ms'] for x in out['searches']]} ms", flush=True)
    return out

ap = argparse.ArgumentParser(); ap.add_argument("--url", required=True); ap.add_argument("--data", default="data")
ap.add_argument("--out", default="results/bench.json")
ap.add_argument("--groups", default="size,threads,layout,repeat", help="comma list: size,threads,layout,repeat")
a = ap.parse_args()
G = a.groups.split(",")
mine = json.load(open(os.path.join(a.data, "config.json")))["shards"]
os.makedirs(os.path.dirname(a.out), exist_ok=True)
runs = []
def save(): json.dump({"vcpus": call(a.url, "/api/info")[1].get("vcpus"), "runs": runs}, open(a.out, "w"), indent=1)

# 1. data size: the three modes at 1, 20, 50, 100, 200 shards
for n in (1, 20, 50, 100, 200) if "size" in G else ():
    for mode, t, p in (("seq", 1, 1), ("omp", 4, 1), ("mpi", 4, 3)):
        runs.append(run(a.url, "size", mode, mine[:n], t, p)); save()
# 2. OpenMP thread scaling on 50 shards
for t in (1, 2, 3, 4) if "threads" in G else ():
    runs.append(run(a.url, "threads", "omp", mine[:50], t)); save()
# 3. cluster layouts on 50 shards (P ranks x T threads)
for p, t in ((1, 4), (2, 4), (3, 1), (3, 2), (3, 4), (6, 2), (6, 4)) if "layout" in G else ():
    runs.append(run(a.url, "layout", "mpi", mine[:50], t, p)); save()
# 4. repeatability: the size-M trio twice more
for _ in range(2) if "repeat" in G else ():
    for mode, t, p in (("seq", 1, 1), ("omp", 4, 1), ("mpi", 4, 3)):
        runs.append(run(a.url, "repeat", mode, mine[:20], t, p)); save()
print("saved", a.out)
