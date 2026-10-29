#!/usr/bin/env python3
"""
Drives server.py the way the grader does: runs seq / omp / mpi through the web API, prints what the SERVER measured
(wall time, CPU, OS threads, engine processes, machines per rank, progress), then compares every search with ref.py.
Runs on the laptop; ref.py needs the same shards in ./data.

    python api_check.py --url http://localhost:8000 --shards 0 1 2
    python api_check.py --url http://<master-public-ip>:8000 --shards 0 1 2 3 4 6 --threads 4 --ranks 3
"""
import argparse, json, os, sys, time, urllib.error, urllib.request
import ref

def call(url, path, body=None):
    req = urllib.request.Request(url + path, method="GET" if body is None else "POST", headers={"Content-Type": "application/json"},
                                 data=None if body is None else json.dumps(body).encode())
    try:
        with urllib.request.urlopen(req, timeout=600) as r: return r.status, json.load(r)
    except urllib.error.HTTPError as e: return e.code, json.load(e)

def run(url, mode, body):
    code, j = call(url, f"/api/run/{mode}", body)
    if code != 200: sys.exit(f"{mode}: HTTP {code} {j}")
    while True:
        time.sleep(0.5)
        _, s = call(url, f"/api/status?job={j['job']}")
        if s["state"] != "running": break
    _, r = call(url, f"/api/result?job={j['job']}")
    if r["state"] != "done": sys.exit(f"{mode}: {r['state']} {r.get('error')}")
    pcts = [p[1] for p in r["progress"]]
    machines = sorted({c["host"] + "/" + (c.get("iid") or "-") for c in r["cluster"] or []})
    print(f"{mode:4} {r['req']['ranks']} rank(s) x {r['req']['threads']} thr: wall {r['wall_sec']:7.2f}s  cpu {r['server_cpu_sec']:7.2f}s"
          f"  threads_max {r['observed']['threads_max']}  engine_procs {r['observed']['engine_procs_max']}"
          f"  progress {len(pcts)} values {pcts[:1]}..{pcts[-1:]}" + (f"\n     ranks on: {machines}" if machines else ""))
    return r["wall_sec"]

def searches(url, queries, model):
    fails = 0
    for q in queries:
        code, j = call(url, "/api/search", {"q": q})
        want = ref.search(q, *model)
        got = j.get("results", []) if code == 200 else None
        ok = got is not None and len(got) == len(want) and all(
            g[0] == w[0] and abs(g[1] - w[1]) <= 1e-6 * abs(w[1]) for g, w in zip(got, want))
        fails += not ok
        if not ok: print(f"     FAIL {q!r}\n       want {want}\n       got  {code} {j}")
    print(f"     search: {len(queries) - fails}/{len(queries)} OK")
    return fails

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8000"); ap.add_argument("--data", default="data")
ap.add_argument("--shards", type=int, nargs="+", required=True)
ap.add_argument("--threads", type=int, default=4); ap.add_argument("--ranks", type=int, default=3)
a = ap.parse_args()

_, info = call(a.url, "/api/info")
print("info:", {k: info.get(k) for k in ("roll", "server_sha256", "iid", "instance_type", "vcpus", "hostfile_hosts", "shards_present", "shards_expected", "image_digest")})
s = ref.load_settings(os.path.join(a.data, "settings.txt"))
docs = ref.load_docs(a.data, a.shards); N, df, post = ref.build(docs, s)
model = (docs, N, df, post, s)
queries = json.load(open(os.path.join(a.data, "config.json")))["search_queries"] + ref.EDGE_QUERIES

fails, t = 0, {}
for mode, body in [("seq", {}), ("omp", {"threads": a.threads}), ("mpi", {"ranks": a.ranks, "threads": a.threads})]:
    t[mode] = run(a.url, mode, dict(body, shards=a.shards))
    fails += searches(a.url, queries, model)
print(f"speedup vs seq: omp {t['seq'] / t['omp']:.2f}x   mpi {t['seq'] / t['mpi']:.2f}x")
print("ALL OK" if fails == 0 else f"{fails} FAILED"); sys.exit(fails > 0)
