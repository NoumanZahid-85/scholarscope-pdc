#!/usr/bin/env python3
"""
Builds every README figure from MEASURED data (nothing typed in by hand):
    results/bench.json   <- bench_api.py (server-measured runs)
    results/mem.log      <- deploy/memlog.sh on the master ("unix-time used-MB", 1/s)
    results/logs/*.err   <- the engine's per-rank phase times (/data/logs on the master)
Writes Images/charts/<name>-light.svg and <name>-dark.svg (+ race-*.gif). Needs matplotlib, numpy, pillow.

    python make_charts.py
"""
import glob, json, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import PathPatch
from matplotlib.path import Path

OUT = "Images/charts"
# Colour-blind-checked palette: modes = 3 categorical hues, phases = one blue ramp light -> dark.
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  mode={"seq": "#2a78d6", "omp": "#eb6834", "mpi": "#1baf7a"}, phase=["#86b6ef", "#3987e5", "#1c5cab"]),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
                 mode={"seq": "#3987e5", "omp": "#d95926", "mpi": "#199e70"}, phase=["#86b6ef", "#3987e5", "#184f95"]),
}
LABEL = {"seq": "Sequential (1 thread)", "omp": "OpenMP (1 node × 4 threads)", "mpi": "Cluster (3 nodes × 4 threads)"}
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 11, "svg.fonttype": "path",
                     "axes.spines.top": False, "axes.spines.right": False})

def fig(t, w=8, h=4.2):
    f, ax = plt.subplots(figsize=(w, h), dpi=100)
    f.patch.set_facecolor(t["surface"]); ax.set_facecolor(t["surface"])
    for s in ("left", "bottom"): ax.spines[s].set_color(t["axis"]); ax.spines[s].set_linewidth(1)
    ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], length=0, pad=6)
    ax.grid(True, color=t["grid"], linewidth=1); ax.set_axisbelow(True)
    return f, ax

def titles(ax, t, title, sub):
    ax.set_title(title, loc="left", color=t["ink"], fontsize=14, fontweight="semibold", pad=26)
    ax.text(0, 1.02, sub, transform=ax.transAxes, color=t["ink2"], fontsize=10.5, va="bottom")

def save(f, name, theme):
    os.makedirs(OUT, exist_ok=True)
    f.savefig(f"{OUT}/{name}-{theme}.svg", facecolor=f.get_facecolor(), bbox_inches="tight", pad_inches=0.25)
    if os.environ.get("PREVIEW"):                                     # PNG copies for eyeballing the layout
        f.savefig(f"{os.environ['PREVIEW']}/{name}-{theme}.png", facecolor=f.get_facecolor(), bbox_inches="tight", pad_inches=0.25)
    plt.close(f)

def px(ax):
    """data units per pixel in x and y (limits must already be final)"""
    f = ax.figure; f.canvas.draw()
    bb = ax.get_window_extent(); x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    return (x1 - x0) / bb.width, (y1 - y0) / bb.height

def hbar(ax, y, x0, x1, color, t, rounded=True, thick=22):
    """horizontal bar <=24px thick, 4px rounded data end, square at the baseline - ONE shape, so no seams"""
    ux, uy = px(ax); h = thick * uy / 2
    if x1 - x0 <= 0: return
    rx, ry = (min(4 * ux, (x1 - x0) / 2), 4 * uy) if rounded else (0, 0)
    P = Path
    verts = [(x0, y - h), (x1 - rx, y - h), (x1, y - h), (x1, y - h + ry), (x1, y + h - ry), (x1, y + h), (x1 - rx, y + h),
             (x0, y + h), (x0, y - h)]
    codes = [P.MOVETO, P.LINETO, P.CURVE3, P.CURVE3, P.LINETO, P.CURVE3, P.CURVE3, P.LINETO, P.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), fc=color, ec="none", zorder=3))

# ------------------------------------------------------------------ data
B = json.load(open("results/bench.json"))["runs"]
size = {(r["mode"], r["shards"]): r for r in B if r["group"] == "size"}
NS = sorted({r["shards"] for r in B if r["group"] == "size"})
docs = {n: None for n in NS}   # documents per size, from the engine logs below when available

def phases():
    """engine phase times of the size-L runs (slowest rank) from results/logs, newest run per mode"""
    rx = re.compile(r"rank (\d+): (\d+) files, (\d+) docs, (\d+) threads \| read ([\d.]+)s parse ([\d.]+)s docs ([\d.]+)s buckets ([\d.]+)s")
    out = {}
    for path in sorted(glob.glob("results/logs/*.err")):
        rows = [m.groups() for m in map(rx.search, open(path)) if m]
        if not rows: continue
        ndocs = sum(int(r[2]) for r in rows); nfiles = sum(int(r[1]) for r in rows)
        if nfiles != 201: continue                                       # size L = 200 shards + canary
        mode = "mpi" if len(rows) > 1 else ("seq" if rows[0][3] == "1" else "omp")
        slow = max(rows, key=lambda r: sum(map(float, r[4:])))
        out[mode] = dict(read=float(slow[4]), parse=float(slow[5]), write=float(slow[6]) + float(slow[7]), docs=ndocs)
    return out
PH = phases()

# ------------------------------------------------------------------ 1. time vs data size
def chart_size(t, theme):
    f, ax = fig(t)
    for m in ("seq", "omp", "mpi"):
        xs = NS; ys = [size[(m, n)]["wall"] for n in xs]
        ax.plot(xs, ys, color=t["mode"][m], lw=2, solid_capstyle="round", zorder=3)
        ax.plot(xs, ys, "o", ms=8, color=t["mode"][m], mec=t["surface"], mew=2, zorder=4)
        sp = size[("seq", xs[-1])]["wall"] / ys[-1]
        ax.annotate(f"{LABEL[m]}  {ys[-1]:.0f} s" + ("" if m == "seq" else f"  ({sp:.1f}× faster)"), (xs[-1], ys[-1]),
                    xytext=(10, 0), textcoords="offset points", va="center", color=t["ink"], fontsize=10)
    ax.set_xlim(0, NS[-1] * 1.02); ax.set_ylim(0, None)
    ax.set_xlabel("Shards indexed (≈ 12,500 papers each)", color=t["ink2"]); ax.set_ylabel("Index build time (s)", color=t["ink2"])
    titles(ax, t, "Build time grows linearly with data — parallel modes grow far slower",
           "Wall time measured by server.py on 3 × m5.xlarge")
    save(f, "time-vs-size", theme)

# ------------------------------------------------------------------ 2. speedup at size L
def chart_speedup(t, theme):
    f, ax = fig(t, h=2.9)
    n = NS[-1]; base = size[("seq", n)]["wall"]
    modes = ["mpi", "omp", "seq"]; vals = [base / size[(m, n)]["wall"] for m in modes]
    ax.set_xlim(0, max(vals) * 1.25); ax.set_ylim(-0.6, 2.6)
    ax.set_yticks(range(3), [LABEL[m] for m in modes]); ax.grid(axis="y", visible=False)
    for i, (m, v) in enumerate(zip(modes, vals)):
        hbar(ax, i, 0, v, t["mode"][m], t)
        ax.text(v, i, f"  {v:.1f}×   ({size[(m, n)]['wall']:.1f} s)", va="center", color=t["ink"], fontsize=11)
    ax.set_xlabel("Speedup over sequential (higher is better)", color=t["ink2"])
    titles(ax, t, f"Size L ({n} shards, 2.5 M papers): {vals[0]:.1f}× faster on the cluster",
           "Same binary, same answers; only the parallelism changes")
    save(f, "speedup-L", theme)

# ------------------------------------------------------------------ 3. OpenMP thread scaling
def chart_threads(t, theme):
    th = sorted((r for r in B if r["group"] == "threads"), key=lambda r: r["threads"])
    T = [r["threads"] for r in th]; sp = [th[0]["wall"] / r["wall"] for r in th]
    f, ax = fig(t, h=3.8)
    ax.plot(T, T, color=t["muted"], lw=1, zorder=2)
    ax.annotate("ideal (linear)", (T[-1], T[-1]), xytext=(8, 0), textcoords="offset points", va="center", color=t["muted"], fontsize=10)
    ax.axvline(2, color=t["axis"], lw=1, zorder=1)
    ax.text(2.04, 3.7, "2 physical cores →\nthreads 3–4 are hyper-threads", color=t["ink2"], fontsize=9.5, va="top")
    ax.plot(T, sp, color=t["mode"]["omp"], lw=2, zorder=3)
    ax.plot(T, sp, "o", ms=8, color=t["mode"]["omp"], mec=t["surface"], mew=2, zorder=4)
    for x, y in zip(T, sp): ax.annotate(f"{y:.2f}×", (x, y), xytext=(0, -16), textcoords="offset points", ha="center", color=t["ink"], fontsize=10)
    ax.set_xticks(T); ax.set_xlim(0.8, 4.9); ax.set_ylim(0, 4.3)
    ax.set_xlabel("OMP_NUM_THREADS on one m5.xlarge (4 vCPU)", color=t["ink2"]); ax.set_ylabel("Speedup vs 1 thread", color=t["ink2"])
    titles(ax, t, "OpenMP scaling: near-linear on real cores, smaller gains on hyper-threads",
           f"{th[0]['shards']} shards, wall time measured by server.py")
    save(f, "thread-scaling", theme)

# ------------------------------------------------------------------ 4. cluster layouts
def chart_layouts(t, theme):
    """before (OpenMPI default binding, muted) vs after (--bind-to none, cluster colour), one row per P x T layout"""
    before = [r for r in B if r["group"] == "layout"]
    after = {(r["ranks"], r["threads"]): r for r in json.load(open("results/bench-bind-none.json"))["runs"]} \
        if os.path.exists("results/bench-bind-none.json") else {}
    seq = next(r["wall"] for r in B if r["group"] == "size" and r["mode"] == "seq" and r["shards"] == before[0]["shards"])
    f, ax = fig(t, h=4.6)
    ax.set_ylim(-0.7, len(before) - 0.3); ax.set_xlim(0, max(r["wall"] for r in before) * 1.42)
    ax.set_yticks(range(len(before)), [f"P={r['ranks']} × T={r['threads']}  ({r['machines']} node{'s' * (r['machines'] > 1)})" for r in before])
    ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    _, uy = px(ax); off = 8 * abs(uy)                                    # two 14px bars, 2px apart (axis is inverted)
    for i, r in enumerate(before):
        a = after.get((r["ranks"], r["threads"]))
        hbar(ax, i - off if a else i, 0, r["wall"], t["axis"] if a else t["mode"]["mpi"], t, thick=14 if a else 22)
        ax.text(r["wall"], i - off if a else i, f"  {r['wall']:.1f} s", va="center", color=t["ink2"], fontsize=9.5)
        if a:
            hbar(ax, i + off, 0, a["wall"], t["mode"]["mpi"], t, thick=14)
            ax.text(a["wall"], i + off, f"  {a['wall']:.1f} s   {seq / a['wall']:.1f}×", va="center", color=t["ink"], fontsize=10)
    if after:
        ax.plot([], [], "s", ms=9, color=t["axis"], label="OpenMPI default (≤ 2 ranks: each rank pinned to 1 core)")
        ax.plot([], [], "s", ms=9, color=t["mode"]["mpi"], label="with --bind-to none (speedup vs sequential)")
        ax.legend(loc="lower right", frameon=False, labelcolor=t["ink2"], fontsize=9.5)
    ax.set_xlabel("Index build time (s) — shorter is better", color=t["ink2"])
    titles(ax, t, "Cluster layouts: all 3 machines matter more than extra ranks",
           f"{before[0]['shards']} shards; ranks placed round-robin over the nodes (--map-by node)")
    save(f, "layouts", theme)

# ------------------------------------------------------------------ 5. where the time goes (phases, size L)
def chart_phases(t, theme):
    if not PH: return
    modes = [m for m in ("mpi", "omp", "seq") if m in PH]
    f, ax = fig(t, h=2.9)
    total = max(sum(PH[m][k] for k in ("read", "parse", "write")) for m in modes)
    ax.set_xlim(0, total * 1.18); ax.set_ylim(-0.6, len(modes) - 0.4)
    ax.set_yticks(range(len(modes)), [LABEL[m] if m != "mpi" else "Cluster (slowest of 3 ranks)" for m in modes]); ax.grid(axis="y", visible=False)
    ux, _ = px(ax); gap = 2 * ux
    names = [("read", "gunzip"), ("parse", "JSON + tokenize + hash"), ("write", "sort + write buckets")]
    for i, m in enumerate(modes):
        x = 0
        for j, (k, _) in enumerate(names):
            w = PH[m][k]
            hbar(ax, i, x, x + w - (gap if j < 2 else 0), t["phase"][j], t, rounded=(j == 2)); x += w
        ax.text(x, i, f"  {x:.0f} s", va="center", color=t["ink"], fontsize=10.5)
    for j, (_, lab) in enumerate(names):
        ax.plot([], [], "s", ms=9, color=t["phase"][j], label=lab)
    ax.legend(loc="upper left", frameon=False, labelcolor=t["ink2"], fontsize=10, ncol=3, bbox_to_anchor=(0, -0.28), handletextpad=0.3)
    ax.set_xlabel("Seconds inside the engine (from its stderr phase timer)", color=t["ink2"])
    titles(ax, t, "Where the time goes at size L", f"{PH[modes[-1]]['docs']:,} documents; every phase is parallel")
    save(f, "phases-L", theme)

# ------------------------------------------------------------------ 6. memory during the size-L runs
def chart_memory(t, theme):
    if not os.path.exists("results/mem.log"): return
    mem = [tuple(map(float, l.split())) for l in open("results/mem.log") if len(l.split()) == 2]
    L = [r for r in B if r["group"] == "size" and r["shards"] == NS[-1]]
    t0 = L[0]["t_start"] - 5; t1 = L[-1]["t_start"] + L[-1]["wall"] + 15
    pts = [(ts - t0, mb / 1024) for ts, mb in mem if t0 <= ts <= t1]
    f, ax = fig(t, h=3.9)
    for r in L:
        a = r["t_start"] - t0
        ax.axvspan(a, a + r["wall"], color=t["mode"][r["mode"]], alpha=0.10, lw=0, zorder=1)
        ax.text(a + r["wall"] / 2, 12.5, r["mode"], ha="center", color=t["ink2"], fontsize=10.5, fontweight="semibold")
    ax.axhline(15.4, color=t["axis"], lw=1)
    ax.text(t1 - t0, 15.7, "RAM of one m5.xlarge: 15.4 GB", ha="right", color=t["ink2"], fontsize=9.5)
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=t["ink"], lw=2, zorder=3)
    peak = max(pts, key=lambda p: p[1])
    ax.annotate(f"peak {peak[1]:.1f} GB", peak, xytext=(8, 8), textcoords="offset points", color=t["ink"], fontsize=10)
    ax.set_ylim(0, 17.5); ax.set_xlim(0, t1 - t0)
    ax.set_xlabel("Seconds since the first size-L run started", color=t["ink2"]); ax.set_ylabel("Used memory on the master (GB)", color=t["ink2"])
    titles(ax, t, "Memory on the master while indexing 2.5 M papers",
           "Hash-keyed postings in thread-private buckets keep even the 1-node runs at about half the RAM")
    save(f, "memory-L", theme)

# ------------------------------------------------------------------ 7. animated progress race (size L)
def race(t, theme):
    L = {r["mode"]: r for r in B if r["group"] == "size" and r["shards"] == NS[-1]}
    end = max(r["progress"][-1][0] for r in L.values()); fps, secs = 15, 10
    f, ax = fig(t, w=8, h=4)
    f.subplots_adjust(left=0.09, right=0.97, top=0.80, bottom=0.15)    # GIF frames get no tight bbox: fixed margins
    ax.set_xlim(0, end * 1.02); ax.set_ylim(0, 112); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel(f"Seconds (real time, played back {end / secs:.0f}× faster)", color=t["ink2"]); ax.set_ylabel("Progress (%)", color=t["ink2"])
    titles(ax, t, "Indexing 2.5 M papers: the three buttons racing", "Progress numbers exactly as server.py received them")
    lines = {m: ax.plot([], [], color=t["mode"][m], lw=2)[0] for m in L}
    heads = {m: ax.plot([], [], "o", ms=8, color=t["mode"][m], mec=t["surface"], mew=2)[0] for m in L}
    labels = {m: ax.text(0, 0, "", color=t["ink"], fontsize=10, va="center") for m in L}
    clock = ax.text(0.99, 0.04, "", transform=ax.transAxes, ha="right", color=t["ink2"], fontsize=12)
    def frame(i):
        now = end * min(1, i / (fps * secs))
        for m, r in L.items():
            p = [(x, y) for x, y in r["progress"] if x <= now] or [(0, 0)]
            lines[m].set_data([x for x, _ in p], [y for _, y in p]); heads[m].set_data([p[-1][0]], [p[-1][1]])
            done = p[-1][1] >= 100
            labels[m].set_position((p[-1][0] + end * 0.012, p[-1][1]))
            labels[m].set_text(f"{m}  {r['wall']:.0f} s ✓" if done else f"{m}  {p[-1][1]}%")
        clock.set_text(f"t = {now:5.1f} s")
        return []
    anim = FuncAnimation(f, frame, frames=fps * secs + fps * 2, blit=False)
    os.makedirs(OUT, exist_ok=True)
    anim.save(f"{OUT}/race-{theme}.gif", writer=PillowWriter(fps=fps), savefig_kwargs={"facecolor": t["surface"]})
    if os.environ.get("PREVIEW"):
        frame(int(fps * secs * 0.3)); f.savefig(f"{os.environ['PREVIEW']}/race-{theme}.png", facecolor=t["surface"])
    plt.close(f)

for theme, t in THEMES.items():
    for c in (chart_size, chart_speedup, chart_threads, chart_layouts, chart_phases, chart_memory, race): c(t, theme)
print("written:", sorted(os.listdir(OUT)))
