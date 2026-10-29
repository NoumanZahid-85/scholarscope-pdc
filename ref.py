#!/usr/bin/env python3
"""
Reference implementation of the assignment's exact definitions (PDF section 4).
Slow but simple: it is the ANSWER KEY we compare the C++ engine against. Not part of the submission.

    python ref.py selftest                                     # checks the PDF's worked example
    python ref.py --data data search "graph neural network" "zq5109237" --shards 0 1     (queries BEFORE --shards)
"""
import argparse, gzip, json, math, os, re, sys, time
from collections import Counter, defaultdict

TOK = re.compile(r"[a-z0-9]+")

# Test queries used by check.py and api_check.py (on top of config.json's): word pairs, case/punctuation, repeats,
# canary words (exact ties), all-stop-word, unknown, empty, the Kelvin sign.
EDGE_QUERIES = ["graph neural network", "Graph-Neural NETWORKS!!", "neural network neural network", "zq5109237",
                "zq5109237 zqac39077", "the of and", "qwertyzzz", "", "quantum", "Kelvin temperature", "dark matter halo",
                "machine learning", "is in on"]

def load_settings(path):
    s = dict(l.strip().split("=", 1) for l in open(path, encoding="utf-8") if "=" in l)
    s["min_token_len"] = int(s["min_token_len"])
    s["stoplist"] = set(w for w in s.get("stoplist", "").split(",") if w)
    return s

def features(text, s):
    # 1. lower-case, tokens = maximal runs of [a-z0-9]   2. drop short, then stop-words (before stemming)
    toks = [t for t in TOK.findall(text.lower()) if len(t) >= s["min_token_len"] and t not in s["stoplist"]]
    # 3. optional plural_s stemming
    if s["stemming"] == "plural_s":
        toks = [t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t for t in toks]
    # 4. unigrams + bigrams of the REMAINING tokens
    return Counter(toks + [a + "_" + b for a, b in zip(toks, toks[1:])])

def tf(c, maxc, kind):
    return {"raw": c, "log": 1 + math.log(c), "aug": 0.5 + 0.5 * c / maxc}[kind]

def idf(N, df, kind):
    if kind == "plain": return math.log(N / df)
    if kind == "smooth": return math.log((1 + N) / (1 + df)) + 1
    if kind == "bm25": return math.log(1 + (N - df + 0.5) / (df + 0.5))
    raise ValueError(kind)

def build(docs, s):
    """docs: list of (id, text). Returns (df, postings feature -> [(doc_index, weight)])."""
    feats = [features(t, s) for _, t in docs]
    df = Counter(f for fc in feats for f in fc)
    N = len(docs); post = defaultdict(list)
    for i, fc in enumerate(feats):
        if not fc: continue                                   # no features -> all-zero vector
        maxc = max(fc.values())
        w = {f: tf(c, maxc, s["tf"]) * idf(N, df[f], s["idf"]) for f, c in fc.items()}
        if s["norm"] == "l2":
            L = math.sqrt(sum(x * x for x in w.values()))
            if L > 0: w = {f: x / L for f, x in w.items()}
        for f, x in w.items(): post[f].append((i, x))
    return N, df, post

def search(q, docs, N, df, post, s, k=10):
    qw = {f: c * idf(N, df[f], s["idf"]) for f, c in features(q, s).items() if df.get(f, 0) > 0}
    L = math.sqrt(sum(x * x for x in qw.values()))
    if L == 0: return []
    score = defaultdict(float)
    for f, x in qw.items():
        for i, w in post[f]: score[i] += (x / L) * w
    res = sorted(((docs[i][0], v) for i, v in score.items() if v > 0), key=lambda r: (-r[1], r[0]))
    return res[:k]

def load_docs(data, shards):
    docs = []
    for sh in shards:
        with gzip.open(os.path.join(data, "shards", f"shard-{sh:03d}.jsonl.gz"), "rt", encoding="utf-8") as f:
            docs += [(d["id"], d["text"]) for d in map(json.loads, f)]
    with open(os.path.join(data, "canary.jsonl"), encoding="utf-8") as f:
        docs += [(d["id"], d["text"]) for d in map(json.loads, f)]
    return docs

def selftest():
    s = {"tf": "log", "idf": "smooth", "norm": "l2", "min_token_len": 3, "stemming": "none", "stoplist": {"with", "is"}}
    docs = [("d1", "Graph neural networks classify nodes in a graph."), ("d2", "Neural networks learn graph structure."),
            ("d3", "Neural machine translation with attention."), ("d4", "Graph colouring is a hard problem.")]
    got = [(i, round(v, 4)) for i, v in search("graph neural", docs, *build(docs, s), s)]
    want = [("d1", 0.5242), ("d2", 0.2277), ("d3", 0.1195), ("d4", 0.1195)]
    assert got == want, got
    print("selftest OK:", got)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data")
    ap.add_argument("cmd", choices=["selftest", "search"]); ap.add_argument("--shards", type=int, nargs="*", default=[])
    ap.add_argument("queries", nargs="*")
    a = ap.parse_args()
    if a.cmd == "selftest": selftest(); sys.exit()
    s = load_settings(os.path.join(a.data, "settings.txt")); t0 = time.time()
    docs = load_docs(a.data, a.shards); N, df, post = build(docs, s)
    print(f"# N={N} features={len(df)} built in {time.time() - t0:.1f}s", file=sys.stderr)
    for q in a.queries:
        print(f"== {q}")
        for i, v in search(q, docs, N, df, post, s): print(f"{i} {v:.10g}")
