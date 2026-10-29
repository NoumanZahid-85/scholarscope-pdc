// ScholarScope engine: TF-IDF over word + word-pair features, OpenMP threads inside each MPI rank.
//
//   engine index <shard numbers...>    build the index from /data/shards/shard-NNN.jsonl.gz + /data/canary.jsonl
//   engine search "<query>"            print the top 10 "id score" lines from the last index
//
// stdout carries ONLY what server.py parses (progress numbers / "id score" lines). Everything else goes to stderr.
//
// Layout of the index (each rank writes its own part on its own machine):
//   /data/index/rank-R/meta.txt        "<local doc count> <number of ranks>"
//   /data/index/rank-R/maxc.bin        u32 per local doc: its largest feature count (for tf=aug)
//   /data/index/rank-R/idoff.bin + ids.bin   doc ids: u64 offsets + one blob of characters
//   /data/index/rank-R/dict-BBB.bin    sorted {hash, start, df} for features whose hash falls in bucket BBB
//   /data/index/rank-R/post-BBB.bin    postings {local doc, count}, grouped by feature in dict order
#include <mpi.h>
#include <omp.h>
#include <zlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>
#include "json.hpp"

namespace fs = std::filesystem;
using u64 = uint64_t;
using u32 = uint32_t;

static const std::string DATA = "/data";
static const int NB = 256;        // hash buckets = files per rank; the top 8 bits of a feature hash pick its bucket
static const int MAX_ID = 63;     // doc ids are sent between ranks in fixed 64-byte records
static int RANK = 0, NP = 1;
static bool MPI_ON = false;       // started by mpirun? Otherwise we are 1 rank and never touch MPI (see main)

static void barrier() { if (MPI_ON) MPI_Barrier(MPI_COMM_WORLD); }

[[noreturn]] static void die(const std::string& msg) {
    fprintf(stderr, "engine rank %d: %s\n", RANK, msg.c_str());
    fflush(nullptr);
    _exit(1);                     // safe from any thread; mpirun sees the non-zero exit and stops the job
}

// ---------------------------------------------------------------- settings.txt
static struct { char tf, idf; std::string norm, stemming; size_t min_len = 1; std::unordered_set<std::string> stop; } S;

static void load_settings() {
    std::ifstream f(DATA + "/settings.txt");
    if (!f) die("cannot read " + DATA + "/settings.txt");
    std::map<std::string, std::string> kv;
    for (std::string line; std::getline(f, line);) {
        size_t a = line.find_first_not_of(" \t\r"), b = line.find_last_not_of(" \t\r");
        if (a == std::string::npos) continue;
        line = line.substr(a, b - a + 1);
        size_t eq = line.find('=');
        if (eq != std::string::npos) kv[line.substr(0, eq)] = line.substr(eq + 1);
    }
    std::string tf = kv["tf"], idf = kv["idf"];
    if (tf != "raw" && tf != "log" && tf != "aug") die("unknown tf=" + tf);
    if (idf != "plain" && idf != "smooth" && idf != "bm25") die("unknown idf=" + idf);
    S.tf = tf[0]; S.idf = idf[0]; S.norm = kv["norm"]; S.stemming = kv["stemming"];
    // shortcut: norm=l2 needs global df at index time (an all-to-all exchange of every feature); add it only if settings need it
    if (S.norm != "none") die("norm=" + S.norm + " not supported by this engine");
    if (S.stemming != "none" && S.stemming != "plural_s") die("unknown stemming=" + S.stemming);
    S.min_len = std::stoul(kv["min_token_len"]);
    std::string sl = kv["stoplist"];
    for (size_t a = 0; a <= sl.size();) {
        size_t c = sl.find(',', a);
        if (c == std::string::npos) c = sl.size();
        if (c > a) S.stop.insert(sl.substr(a, c - a));
        a = c + 1;
    }
}

// ---------------------------------------------------------------- tokens and features (PDF section 4)
static void keep(std::string& t, std::vector<std::string>& out) {
    if (t.size() >= S.min_len && !S.stop.count(t)) {         // length first, then stop-words, both before stemming
        if (S.stemming == "plural_s" && t.size() > 3 && t.back() == 's' && t[t.size() - 2] != 's') t.pop_back();
        out.push_back(t);
    }
    t.clear();
}

// Lower-case, then maximal runs of [a-z0-9]. Matches Python's str.lower(): the only non-ASCII characters that become
// ASCII letters are U+0130 (-> "i" + a combining dot, so the token ends there) and U+212A Kelvin sign (-> "k").
static void tokenize(const std::string& s, std::vector<std::string>& out) {
    out.clear();
    std::string cur;
    for (size_t i = 0; i < s.size(); i++) {
        unsigned char c = s[i];
        if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) cur += char(c);
        else if (c >= 'A' && c <= 'Z') cur += char(c + 32);
        else if (c == 0xE2 && i + 2 < s.size() && (unsigned char)s[i + 1] == 0x84 && (unsigned char)s[i + 2] == 0xAA) { cur += 'k'; i += 2; }
        else if (c == 0xC4 && i + 1 < s.size() && (unsigned char)s[i + 1] == 0xB0) { cur += 'i'; i += 1; keep(cur, out); }
        else if (!cur.empty()) keep(cur, out);
    }
    if (!cur.empty()) keep(cur, out);
}

// shortcut: features are stored as 64-bit hashes, not strings (saves GBs at size L); two features sharing a hash
// (~1e-4 chance over the whole ~70M vocabulary) would merge. Store the strings if that ever matters.
static inline u64 fnv(const std::string& s, u64 h = 1469598103934665603ULL) {
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ULL; }
    return h;
}
static inline u64 mix(u64 x) {               // splitmix64 finaliser: spreads FNV's bits so the top 8 pick a fair bucket
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL; x ^= x >> 27; x *= 0x94d049bb133111ebULL; return x ^ (x >> 31);
}
static inline u64 uni_hash(const std::string& a) { return mix(fnv(a)); }
static inline u64 bi_hash(const std::string& a, const std::string& b) { return mix(fnv(b, fnv("_", fnv(a)))); }  // = hash of "a_b"

struct Scratch { std::vector<std::string> toks; std::vector<u64> hs; std::vector<std::pair<u64, u32>> fc; };

// Fills w.fc with (feature hash, count) sorted by hash; returns maxc (0 if the text has no features).
static u32 features(const std::string& text, Scratch& w) {
    tokenize(text, w.toks);
    w.hs.clear();
    for (size_t i = 0; i < w.toks.size(); i++) {
        w.hs.push_back(uni_hash(w.toks[i]));
        if (i) w.hs.push_back(bi_hash(w.toks[i - 1], w.toks[i]));
    }
    std::sort(w.hs.begin(), w.hs.end());
    w.fc.clear();
    u32 maxc = 0;
    for (size_t i = 0, j; i < w.hs.size(); i = j) {
        for (j = i; j < w.hs.size() && w.hs[j] == w.hs[i]; j++) {}
        w.fc.push_back({w.hs[i], u32(j - i)});
        maxc = std::max(maxc, u32(j - i));
    }
    return maxc;
}

static double tf_w(u32 c, u32 maxc) {
    if (S.tf == 'r') return c;
    if (S.tf == 'l') return 1 + std::log(double(c));
    return 0.5 + 0.5 * c / maxc;
}
static double idf_w(double N, double df) {
    if (S.idf == 'p') return std::log(N / df);
    if (S.idf == 's') return std::log((1 + N) / (1 + df)) + 1;
    return std::log(1 + (N - df + 0.5) / (df + 0.5));
}

// ---------------------------------------------------------------- small I/O helpers
struct Emit { u64 h; u32 doc, c; };                  // one posting while building (in memory)
struct Dict { u64 h; u32 start, df; };               // on disk: one per feature, sorted by hash
#pragma pack(push, 1)
struct Post { u32 doc; uint16_t c; };                // on disk: 6 bytes per posting
#pragma pack(pop)

static std::string read_gz(const std::string& path) { // gzread also reads plain (uncompressed) files, e.g. canary.jsonl
    gzFile g = gzopen(path.c_str(), "rb");
    if (!g) die("cannot open " + path);
    gzbuffer(g, 1 << 20);
    std::string out;
    std::vector<char> buf(1 << 20);
    int n;
    while ((n = gzread(g, buf.data(), buf.size())) > 0) out.append(buf.data(), n);
    if (n < 0) die("corrupt file " + path);
    gzclose(g);
    return out;
}

template <class T> static void write_vec(const std::string& path, const std::vector<T>& v) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) die("cannot write " + path);
    if (!v.empty() && fwrite(v.data(), sizeof(T), v.size(), f) != v.size()) die("short write " + path);
    if (fclose(f)) die("cannot close " + path);
}

struct Map { const char* p = nullptr; size_t n = 0; };
static Map map_file(const std::string& path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) die("cannot open " + path + " (build the index first)");
    struct stat st;
    fstat(fd, &st);
    Map m;
    m.n = st.st_size;
    if (m.n) {
        void* p = mmap(nullptr, m.n, PROT_READ, MAP_PRIVATE, fd, 0);
        if (p == MAP_FAILED) die("cannot mmap " + path);
        m.p = (const char*)p;
    }
    close(fd);
    return m;
}

static std::string bucket_file(const std::string& dir, const char* kind, int b) {
    char name[32];
    snprintf(name, sizeof name, "/%s-%03d.bin", kind, b);
    return dir + name;
}

// Rank 0 prints whole numbers 0..100, never decreasing, flushed. Only the main thread calls this.
static void progress(int pct) {
    static int last = -1;
    if (RANK != 0 || pct <= last) return;
    last = pct;
    printf("%d\n", pct);
    fflush(stdout);
}

// ---------------------------------------------------------------- engine index
static void cmd_index(const std::vector<int>& shards) {
    progress(0);
    double t0 = omp_get_wtime(), t_read = 0;                  // phase timings, reported on stderr (logs/<job>.err)
    const std::string idx = DATA + "/index", mine = idx + "/rank-" + std::to_string(RANK);

    // [MPI] One rank per machine (rank 0 of the node-local communicator) deletes that machine's old index; all wait.
    int node_rank = 0;
    if (MPI_ON) {
        MPI_Comm node;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
        MPI_Comm_rank(node, &node_rank);
    }
    std::error_code ec;
    if (node_rank == 0) fs::remove_all(idx, ec);
    if (ec) die("cannot delete old index: " + ec.message());
    barrier();
    fs::create_directories(mine, ec);
    if (ec) die("cannot create " + mine + ": " + ec.message());

    // [MPI] Input files = canary + every shard. Rank r takes files r, r+P, r+2P, ... so the ranks split the corpus.
    std::vector<std::string> files = {DATA + "/canary.jsonl"};
    for (int s : shards) {
        char name[64];
        snprintf(name, sizeof name, "/shards/shard-%03d.jsonl.gz", s);
        files.push_back(DATA + name);
    }
    std::vector<std::string> my;
    for (size_t i = RANK; i < files.size(); i += NP) my.push_back(files[i]);

    // [OpenMP] Each thread appends postings to its OWN 256 buckets, so threads never lock or share.
    const int T = omp_get_max_threads();
    std::vector<std::array<std::deque<Emit>, NB>> buck(T);   // deque: grows in small blocks, no 2x vector slack
    std::vector<u32> maxc;
    std::vector<std::string> ids;
    std::string err;

    // Files are handled in batches of T: first the T files are decompressed at the same time (one per thread), then
    // all their documents are parsed by all threads. A batch, not everything at once, keeps memory to T files of text.
    for (size_t fb = 0; fb < my.size(); fb += T) {
        const size_t nb = std::min<size_t>(T, my.size() - fb);
        std::vector<std::string> bufs(nb);
        double tr = omp_get_wtime();
        // [OpenMP] gunzip is sequential inside one file, so the parallelism is across files.
        #pragma omp parallel for schedule(dynamic, 1)
        for (size_t f = 0; f < nb; f++) bufs[f] = read_gz(my[fb + f]);
        t_read += omp_get_wtime() - tr;

        struct Line { u32 f, no; size_t a, n; };               // one non-blank line = one document
        std::vector<Line> lines;
        for (u32 f = 0; f < nb; f++) {
            const std::string& buf = bufs[f];
            u32 no = 0;
            for (size_t a = 0; a < buf.size(); no++) {
                size_t e = buf.find('\n', a);
                if (e == std::string::npos) e = buf.size();
                if (buf.find_first_not_of(" \t\r", a) < e) lines.push_back({f, no, a, e - a});
                a = e + 1;
            }
        }
        const u32 base = ids.size();                           // local doc numbers: fixed by file + line, not by thread
        ids.resize(base + lines.size());
        maxc.resize(base + lines.size());
        std::atomic<size_t> done{0};

        // [OpenMP] The documents of the batch are shared out to the threads in chunks of 64.
        #pragma omp parallel
        {
            const int t = omp_get_thread_num();
            Scratch w;
            #pragma omp for schedule(dynamic, 64)
            for (size_t i = 0; i < lines.size(); i++) {
                const Line& L = lines[i];
                try {
                    auto j = nlohmann::json::parse(bufs[L.f].data() + L.a, bufs[L.f].data() + L.a + L.n);
                    const u32 d = base + i;
                    ids[d] = j.at("id").get<std::string>();
                    if (ids[d].size() > MAX_ID) throw std::runtime_error("doc id longer than 63 characters");
                    maxc[d] = features(j.at("text").get_ref<const std::string&>(), w);
                    for (auto& [h, c] : w.fc) {
                        if (c > 65535) throw std::runtime_error("feature count above 65535");
                        buck[t][h >> 56].push_back({h, d, c});
                    }
                } catch (const std::exception& e) {
                    #pragma omp critical
                    err = my[fb + L.f] + " line " + std::to_string(L.no + 1) + ": " + e.what();
                }
                size_t k = ++done;
                if (t == 0 && (k & 255) == 0) progress(int(80.0 * (fb + nb * double(k) / lines.size()) / my.size()));
            }
        }
        if (!err.empty()) die(err);
        progress(int(80.0 * (fb + nb) / my.size()));
    }

    double t1 = omp_get_wtime();

    // Document table: maxc per doc, ids as offsets + one blob.
    write_vec(mine + "/maxc.bin", maxc);
    std::vector<u64> off = {0};
    std::string blob;
    for (auto& s : ids) { blob += s; off.push_back(blob.size()); }
    write_vec(mine + "/idoff.bin", off);
    write_vec(mine + "/ids.bin", std::vector<char>(blob.begin(), blob.end()));
    std::vector<std::string>().swap(ids);
    double t2 = omp_get_wtime();

    // [OpenMP] Buckets are independent: each thread merges one bucket from all threads, sorts it, writes it.
    std::atomic<int> bdone{0};
    #pragma omp parallel for schedule(dynamic, 1)
    for (int b = 0; b < NB; b++) {
        size_t n = 0;
        for (int t = 0; t < T; t++) n += buck[t][b].size();
        std::vector<Emit> v;
        v.reserve(n);
        for (int t = 0; t < T; t++) {
            v.insert(v.end(), buck[t][b].begin(), buck[t][b].end());
            std::deque<Emit>().swap(buck[t][b]);              // free as we go: peak memory stays ~1x
        }
        // (hash, doc) order makes the files identical whatever the thread count
        std::sort(v.begin(), v.end(), [](const Emit& x, const Emit& y) { return x.h != y.h ? x.h < y.h : x.doc < y.doc; });
        std::vector<Dict> dict;
        std::vector<Post> post(v.size());
        for (size_t i = 0; i < v.size(); i++) {
            if (i == 0 || v[i].h != v[i - 1].h) dict.push_back({v[i].h, u32(i), 0});
            dict.back().df++;
            post[i] = {v[i].doc, uint16_t(v[i].c)};
        }
        write_vec(bucket_file(mine, "dict", b), dict);
        write_vec(bucket_file(mine, "post", b), post);
        int k = ++bdone;
        if (omp_get_thread_num() == 0) progress(80 + 19 * k / NB);
    }

    std::ofstream(mine + "/meta.txt") << maxc.size() << " " << NP << "\n";
    double t3 = omp_get_wtime();
    fprintf(stderr, "engine rank %d: %zu files, %zu docs, %d threads | read %.2fs parse %.2fs docs %.2fs buckets %.2fs\n",
            RANK, my.size(), maxc.size(), T, t_read, t1 - t0 - t_read, t2 - t1, t3 - t2);
    barrier();                                                // 100 only when every rank has finished
    progress(100);
}

// ---------------------------------------------------------------- engine search
struct Rec { char id[MAX_ID + 1]; double score; };            // one candidate, sent from each rank to rank 0

static void cmd_search(const std::string& q) {
    const std::string mine = DATA + "/index/rank-" + std::to_string(RANK);
    u64 nloc;
    int np;
    if (!(std::ifstream(mine + "/meta.txt") >> nloc >> np)) die("no index in " + mine + " (run index first)");
    if (np != NP) die("index was built with " + std::to_string(np) + " ranks, search runs with " + std::to_string(NP));

    // Query features (sorted by hash: the same order on every rank, so every doc's sum is added up identically).
    Scratch w;
    features(q, w);
    const size_t k = w.fc.size();

    // Local df of each query feature from this rank's dictionary; the last slot carries the local doc count.
    std::vector<double> df(k + 1, 0);
    std::vector<Dict> hit(k, Dict{0, 0, 0});
    std::map<int, Map> dicts;
    for (size_t i = 0; i < k; i++) {
        int b = w.fc[i].first >> 56;
        if (!dicts.count(b)) dicts[b] = map_file(bucket_file(mine, "dict", b));
        const Dict* d = (const Dict*)dicts[b].p;
        size_t n = dicts[b].n / sizeof(Dict);
        const Dict* it = std::lower_bound(d, d + n, w.fc[i].first, [](const Dict& x, u64 h) { return x.h < h; });
        if (it != d + n && it->h == w.fc[i].first) { hit[i] = *it; df[i] = it->df; }
    }
    df[k] = nloc;

    // [MPI] Ranks hold disjoint documents, so global df and N are plain sums over the ranks.
    if (MPI_ON) MPI_Allreduce(MPI_IN_PLACE, df.data(), int(k + 1), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    const double N = df[k];

    // Query weight = count x idf (features found nowhere are ignored), then unit length.
    std::vector<double> qw(k, 0);
    double len = 0;
    for (size_t i = 0; i < k; i++)
        if (df[i] > 0) { qw[i] = w.fc[i].second * idf_w(N, df[i]); len += qw[i] * qw[i]; }
    len = std::sqrt(len);

    // Score this rank's documents: score(doc) = sum over query features of (qw/len) x (tf x idf).
    std::vector<Rec> top(10, Rec{{0}, -1});
    if (len > 0 && nloc > 0) {
        const u32* mx = (const u32*)map_file(mine + "/maxc.bin").p;
        const u64* off = (const u64*)map_file(mine + "/idoff.bin").p;
        const char* blob = map_file(mine + "/ids.bin").p;
        std::vector<double> acc(nloc, 0.0);
        std::vector<u32> touched;
        std::map<int, Map> posts;
        for (size_t i = 0; i < k; i++) {
            if (qw[i] <= 0 || hit[i].df == 0) continue;
            const double qn = qw[i] / len, idf = idf_w(N, df[i]);
            int b = w.fc[i].first >> 56;
            if (!posts.count(b)) posts[b] = map_file(bucket_file(mine, "post", b));
            const Post* p = (const Post*)posts[b].p + hit[i].start;
            for (u32 j = 0; j < hit[i].df; j++) {
                if (acc[p[j].doc] == 0) touched.push_back(p[j].doc);   // every term is > 0, so 0 means "not seen yet"
                acc[p[j].doc] += qn * (tf_w(p[j].c, mx[p[j].doc]) * idf);
            }
        }
        auto id = [&](u32 d) { return std::string_view(blob + off[d], off[d + 1] - off[d]); };
        auto better = [&](u32 a, u32 b) { return acc[a] != acc[b] ? acc[a] > acc[b] : id(a) < id(b); };
        size_t m = std::min<size_t>(10, touched.size());
        std::partial_sort(touched.begin(), touched.begin() + m, touched.end(), better);
        for (size_t i = 0; i < m; i++) {
            std::string_view s = id(touched[i]);
            memcpy(top[i].id, s.data(), s.size());
            top[i].score = acc[touched[i]];
        }
    }

    // [MPI] Every rank sends its local top 10 to rank 0; the global top 10 is among them.
    std::vector<Rec> all = top;
    if (MPI_ON) {
        all.resize(RANK == 0 ? 10 * NP : 0);
        MPI_Gather(top.data(), int(10 * sizeof(Rec)), MPI_BYTE, all.data(), int(10 * sizeof(Rec)), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    if (RANK != 0) return;
    all.erase(std::remove_if(all.begin(), all.end(), [](const Rec& r) { return !(r.score > 0); }), all.end());
    std::sort(all.begin(), all.end(), [](const Rec& a, const Rec& b) { return a.score != b.score ? a.score > b.score : strcmp(a.id, b.id) < 0; });
    for (size_t i = 0; i < all.size() && i < 10; i++) printf("%s %.12g\n", all[i].id, all[i].score);
    fflush(stdout);
}

// ---------------------------------------------------------------- main
int main(int argc, char** argv) {
    // Without mpirun we are 1 rank. OpenMPI's MPI_Init would still start helper threads (and an orted process), so the
    // server would count 3 threads for "1 process, 1 thread"; we only initialise MPI when mpirun launched us.
    MPI_ON = getenv("OMPI_COMM_WORLD_SIZE") || getenv("PMIX_RANK");
    if (MPI_ON) {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        MPI_Comm_rank(MPI_COMM_WORLD, &RANK);
        MPI_Comm_size(MPI_COMM_WORLD, &NP);
        if (provided < MPI_THREAD_FUNNELED) die("MPI does not provide MPI_THREAD_FUNNELED");
    }
    load_settings();

    std::string cmd = argc > 1 ? argv[1] : "";
    if (cmd == "index" && argc > 2) {
        std::vector<int> shards;
        for (int i = 2; i < argc; i++) {
            try { shards.push_back(std::stoi(argv[i])); } catch (...) { die(std::string("bad shard number: ") + argv[i]); }
        }
        cmd_index(shards);
    } else if (cmd == "search") {
        cmd_search(argc > 2 ? argv[2] : "");
    } else {
        die("usage: engine index <shard numbers...> | engine search \"<query>\"");
    }
    if (MPI_ON) MPI_Finalize();
    return 0;
}
