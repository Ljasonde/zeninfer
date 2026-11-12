// zeninfer - main.cpp : CLI entry point.
#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "gguf.h"
#include "tokenizer.h"
#include "unicode.h"

namespace {

void print_usage() {
    std::fprintf(stderr,
        "zeninfer - CPU LLM inference engine\n"
        "\n"
        "usage:\n"
        "  zeninfer inspect    <model.gguf> [--json]   architecture metadata + tensor table\n"
        "  zeninfer tokenize   <model.gguf> [--text S | --text-file F | --batch F] [--pieces]\n"
        "  zeninfer spans      <model.gguf> [--text S | --text-file F]   show pre-token spans\n"
        "  zeninfer vocab-dump <model.gguf> <out.vocab>  vocabulary as a flat, diffable file\n"
        "  zeninfer bench-tokenizer <model.gguf> <text-file> [--repeat N]   tokenizer throughput\n"
        "  zeninfer --vocab <file.vocab>               use a vocab-dump instead of a model\n"
        "  zeninfer run        <model.gguf> [opts]     generate text\n"
        "  zeninfer bench      <model.gguf> [opts]     measure prefill/decode throughput\n");
}

std::string human_bytes(uint64_t n) {
    char buf[64];
    if (n >= (1ull << 30)) std::snprintf(buf, sizeof buf, "%.2f GiB", static_cast<double>(n) / (1ull << 30));
    else if (n >= (1ull << 20)) std::snprintf(buf, sizeof buf, "%.2f MiB", static_cast<double>(n) / (1ull << 20));
    else if (n >= (1ull << 10)) std::snprintf(buf, sizeof buf, "%.2f KiB", static_cast<double>(n) / (1ull << 10));
    else std::snprintf(buf, sizeof buf, "%" PRIu64 " B", n);
    return buf;
}

std::string trim_meta(const std::string& s, const std::string& prefix) {
    return s.rfind(prefix, 0) == 0 ? s.substr(prefix.size()) : s;
}

std::string json_escape(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\t': o += "\\t"; break;
            case '\r': o += "\\r"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x", c);
                    o += buf;
                } else {
                    o += static_cast<char>(c);
                }
        }
    }
    return o;
}

// Schema intentionally mirrors tools/dump_gguf.py so `--diff` compares like for like.
int cmd_inspect_json(const std::string& path) {
    zen::GGUFFile g(path);
    const std::string arch = g.get_str("general.architecture", "");
    uint64_t total = 0;
    for (const auto& t : g.tensors()) total += t.nbytes;
    const size_t n_tokens = g.get_str_array("tokenizer.ggml.tokens").size();
    const size_t n_merges = g.get_str_array("tokenizer.ggml.merges").size();

    std::printf("{");
    std::printf("\"path\":\"%s\",", json_escape(path).c_str());
    std::printf("\"version\":%u,", g.version());
    std::printf("\"file_size\":%zu,", g.file().size());
    std::printf("\"alignment\":%llu,", static_cast<unsigned long long>(g.alignment()));
    std::printf("\"tensor_data_offset\":%llu,", static_cast<unsigned long long>(g.tensor_data_offset()));
    std::printf("\"arch\":\"%s\",", json_escape(arch).c_str());
    std::printf("\"name\":\"%s\",", json_escape(g.get_any("general.name", "")).c_str());
    std::printf("\"n_tensors\":%zu,", g.tensors().size());
    std::printf("\"n_tokens\":%zu,", n_tokens);
    std::printf("\"n_merges\":%zu,", n_merges);
    std::printf("\"total_tensor_bytes\":%llu,", static_cast<unsigned long long>(total));
    std::printf("\"tensors\":[");
    bool first = true;
    for (const auto& t : g.tensors()) {
        if (!first) std::printf(",");
        first = false;
        std::printf("{\"name\":\"%s\",\"dtype\":\"%s\",\"dims\":[", json_escape(t.name).c_str(), zen::dtype_name(t.type));
        for (size_t i = 0; i < t.dims.size(); ++i) {
            if (i) std::printf(",");
            std::printf("%llu", static_cast<unsigned long long>(t.dims[i]));
        }
        std::printf("],\"nelems\":%lld,\"offset\":%llu,\"nbytes\":%llu}",
                    static_cast<long long>(t.nelems()),
                    static_cast<unsigned long long>(t.offset),
                    static_cast<unsigned long long>(t.nbytes));
    }
    std::printf("]}\n");
    return 0;
}

int cmd_inspect(const std::string& path) {
    zen::GGUFFile g(path);
    const std::string arch = g.get_any("general.architecture", "<unknown>");

    std::printf("file             : %s\n", path.c_str());
    std::printf("gguf version     : %u\n", g.version());
    std::printf("file size        : %s\n", human_bytes(g.file().size()).c_str());
    std::printf("alignment        : %" PRIu64 "\n", g.alignment());
    std::printf("tensor data at   : %" PRIu64 "\n", g.tensor_data_offset());
    std::printf("architecture     : %s\n", arch.c_str());
    std::printf("name             : %s\n", g.get_any("general.name", "<none>").c_str());
    std::printf("quantization     : %s\n", g.get_any("general.file_type", "<unset>").c_str());
    std::printf("metadata entries : %zu\n", g.metadata().size());
    std::printf("tensors          : %zu\n", g.tensors().size());

    std::printf("\n-- %s.* hyper parameters --\n", arch.c_str());
    const std::string p = arch + ".";
    struct Row { const char* key; const char* label; };
    const Row ints[] = {
        {"block_count", "n_layer"},
        {"embedding_length", "n_embd"},
        {"feed_forward_length", "n_ff"},
        {"attention.head_count", "n_head"},
        {"attention.head_count_kv", "n_head_kv"},
        {"context_length", "n_ctx_train"},
        {"vocab_size", "n_vocab"},
    };
    for (const auto& r : ints) {
        const std::string key = p + r.key;
        if (g.has(key)) std::printf("  %-14s = %u\n", r.label, g.get_u32(key, 0));
    }
    const std::string eps = p + "attention.layer_norm_rms_epsilon";
    if (g.has(eps)) std::printf("  %-14s = %.3e\n", "rms_eps", g.get_f32(eps, 0.f));
    const std::string rope_base = p + "rope.freq_base";
    if (g.has(rope_base)) std::printf("  %-14s = %.1f\n", "rope_base", g.get_f32(rope_base, 0.f));
    const std::string rope_scale = p + "rope.scaling.type";
    if (g.has(rope_scale)) std::printf("  %-14s = %s\n", "rope_scale", g.get_str(rope_scale, "").c_str());
    const std::string tie = p + "tie_word_embeddings";
    if (g.has(tie)) std::printf("  %-14s = %s\n", "tied_embd", g.get(tie).as_u64() ? "true" : "false");

    std::printf("\n-- tokenizer --\n");
    std::printf("  model          = %s\n", g.get_any("tokenizer.ggml.model", "<none>").c_str());
    std::printf("  pre            = %s\n", g.get_any("tokenizer.ggml.pre", "<none>").c_str());
    std::printf("  tokens         = %zu\n", g.get_str_array("tokenizer.ggml.tokens").size());
    std::printf("  merges         = %zu\n", g.get_str_array("tokenizer.ggml.merges").size());
    std::printf("  bos id         = %u\n", g.get_u32("tokenizer.ggml.bos_token_id", 0xFFFFFFFFu));
    std::printf("  eos id         = %u\n", g.get_u32("tokenizer.ggml.eos_token_id", 0xFFFFFFFFu));
    std::printf("  pad id         = %u\n", g.get_u32("tokenizer.ggml.padding_token_id", 0xFFFFFFFFu));
    std::printf("  add_bos        = %s\n", g.has("tokenizer.ggml.add_bos_token")
                                             ? (g.get("tokenizer.ggml.add_bos_token").as_u64() ? "true" : "false")
                                             : "<unset>");

    // per-dtype rollup
    std::printf("\n-- tensor bytes by dtype --\n");
    struct Bucket { const char* name; uint64_t bytes; int count; };
    std::vector<Bucket> buckets;
    for (const auto& t : g.tensors()) {
        const char* nm = zen::dtype_name(t.type);
        bool found = false;
        for (auto& b : buckets) {
            if (std::strcmp(b.name, nm) == 0) { b.bytes += t.nbytes; b.count++; found = true; break; }
        }
        if (!found) buckets.push_back({nm, t.nbytes, 1});
    }
    uint64_t total = 0;
    for (const auto& b : buckets) total += b.bytes;
    for (const auto& b : buckets) {
        std::printf("  %-6s %4d tensors  %12s  %5.1f%%\n", b.name, b.count,
                    human_bytes(b.bytes).c_str(), total ? 100.0 * static_cast<double>(b.bytes) / static_cast<double>(total) : 0.0);
    }
    std::printf("  total  %4zu tensors  %12s\n", g.tensors().size(), human_bytes(total).c_str());

    std::printf("\n-- tensors --\n");
    std::printf("  %-32s %-6s %-22s %12s\n", "name", "dtype", "shape(ggml order)", "bytes");
    for (const auto& t : g.tensors()) {
        std::printf("  %-32s %-6s %-22s %12s\n", trim_meta(t.name, arch + ".").c_str(),
                    zen::dtype_name(t.type), t.shape_str().c_str(), human_bytes(t.nbytes).c_str());
    }
    return 0;
}

// ---------------------------------------------------------------- tokenizer
std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// Escapes bytes for display: control characters and quotes become visible.
std::string repr(std::string_view s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (unsigned char b : s) {
        switch (b) {
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            default:
                if (b < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\x%02x", b);
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(b));
                }
        }
    }
    return out;
}

// The corpus format is one case per line with C-style escapes, so test inputs
// can carry newlines, tabs and arbitrary codepoints without a JSON parser.
std::string unescape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] != '\\' || i + 1 >= s.size()) {
            out.push_back(s[i]);
            continue;
        }
        const char c = s[++i];
        switch (c) {
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            case '0': out.push_back('\0'); break;
            case '\\': out.push_back('\\'); break;
            case 'x': {
                unsigned v = 0;
                if (i + 2 >= s.size() || std::sscanf(s.c_str() + i + 1, "%2x", &v) != 1) {
                    throw std::runtime_error("bad \\x escape");
                }
                out.push_back(static_cast<char>(v));
                i += 2;
                break;
            }
            case 'u':
            case 'U': {
                const int width = (c == 'u') ? 4 : 8;
                unsigned v = 0;
                if (i + width >= s.size() || std::sscanf(s.c_str() + i + 1, width == 4 ? "%4x" : "%8x", &v) != 1) {
                    throw std::runtime_error("bad unicode escape");
                }
                zen::utf8_append(out, v);
                i += static_cast<size_t>(width);
                break;
            }
            default: out.push_back(c); break;
        }
    }
    return out;
}

struct TokOpts {
    std::string model, vocab, text, text_file, batch;
    bool pieces = false, json = false, spans = false, stats = false;
};

zen::Tokenizer load_tokenizer(const TokOpts& o) {
    zen::Tokenizer t;
    if (!o.vocab.empty()) {
        t.load_from_flat(o.vocab);
        return t;
    }
    if (o.model.empty()) throw std::runtime_error("expected a model path or --vocab <file>");
    zen::GGUFFile g(o.model);
    t.load_from_gguf(g);
    return t;
}

void print_tok_stats(const zen::Tokenizer& t) {
    std::printf("vocab            : %zu\n", t.n_vocab());
    std::printf("merges           : %zu\n", t.n_merges());
    std::printf("merges skipped   : %zu\n", t.n_skipped_merges());
    std::printf("merges ambiguous : %zu\n", t.n_ambiguous_merges());
    std::printf("special tokens   : %zu\n", t.n_specials());
}

int cmd_tokenize(const TokOpts& o) {
    const zen::Tokenizer tok = load_tokenizer(o);
    if (!o.batch.empty()) {
        const std::string all = read_file(o.batch);
        size_t pos = 0, count = 0;
        while (pos < all.size()) {
            size_t nl = all.find('\n', pos);
            if (nl == std::string::npos) nl = all.size();
            std::string line = all.substr(pos, nl - pos);
            pos = nl + 1;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;  // blank separator; "" has no tokens anyway
            std::vector<std::string> pieces;
            const std::vector<int32_t> ids = tok.encode(unescape(line), o.pieces ? &pieces : nullptr, true);
            std::printf("{\"ids\":[");
            for (size_t k = 0; k < ids.size(); ++k) std::printf("%s%d", k ? "," : "", ids[k]);
            std::printf("]");
            if (o.pieces) {
                std::printf(",\"pieces\":[");
                for (size_t k = 0; k < pieces.size(); ++k) {
                    std::printf("%s\"%s\"", k ? "," : "", json_escape(pieces[k]).c_str());
                }
                std::printf("]");
            }
            std::printf("}\n");
            ++count;
        }
        std::fprintf(stderr, "encoded %zu case(s)\n", count);
        return 0;
    }

    const std::string text = o.text_file.empty() ? unescape(o.text) : read_file(o.text_file);
    if (o.spans) {
        const auto spans = zen::pretokenize(text);
        std::printf("%zu span(s) for %zu byte(s)\n", spans.size(), text.size());
        for (size_t k = 0; k < spans.size(); ++k) {
            std::printf("[%03zu] %-24s %s\n", k,
                        repr(std::string_view(text).substr(spans[k].first, spans[k].second - spans[k].first)).c_str(),
                        "");
        }
        return 0;
    }

    std::vector<std::string> pieces;
    const std::vector<int32_t> ids = tok.encode(text, &pieces, true);
    print_tok_stats(tok);
    std::printf("\ntext             : \"%s\" (%zu bytes)\n", repr(text).c_str(), text.size());
    std::printf("tokens           : %zu\n", ids.size());
    std::printf("ids              : [");
    for (size_t k = 0; k < ids.size(); ++k) std::printf("%s%d", k ? ", " : "", ids[k]);
    std::printf("]\n");
    if (o.pieces) {
        std::printf("pieces           : ");
        for (size_t k = 0; k < pieces.size(); ++k) std::printf("%s\"%s\"", k ? " | " : "", repr(pieces[k]).c_str());
        std::printf("\n");
    }
    const std::string back = tok.decode(ids);
    std::printf("round trip       : %s\n", back == text ? "exact" : "MISMATCH");
    return back == text ? 0 : 1;
}

int cmd_bench_tokenizer(const std::string& model, const std::string& file, int repeat) {
    zen::Tokenizer tok;
    zen::GGUFFile g(model);
    tok.load_from_gguf(g);

    const std::string text = read_file(file);
    if (text.empty()) throw std::runtime_error("empty input: " + file);

    // One warm-up pass so the first allocation / branch-predictor state does not
    // get charged to the reported number.
    std::vector<int32_t> ids = tok.encode(text, true);
    const size_t n_tokens = ids.size();

    double best_ms = 1e18, total_ms = 0.0;
    for (int r = 0; r < repeat; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        ids = tok.encode(text, true);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (ids.size() != n_tokens) throw std::runtime_error("non-deterministic token count");
        best_ms = std::min(best_ms, ms);
        total_ms += ms;
    }

    const double mb = static_cast<double>(text.size()) / (1024.0 * 1024.0);
    std::printf("input            : %s (%.2f MiB)\n", file.c_str(), mb);
    std::printf("tokens           : %zu (%.3f bytes/token)\n", n_tokens,
                n_tokens ? static_cast<double>(text.size()) / n_tokens : 0.0);
    std::printf("runs             : %d (best of)\n", repeat);
    std::printf("best             : %.2f ms = %.1f MB/s = %.0f tok/s\n", best_ms, mb * 1000.0 / best_ms,
                n_tokens * 1000.0 / best_ms);
    std::printf("mean             : %.2f ms = %.1f MB/s = %.0f tok/s\n", total_ms / repeat,
                mb * 1000.0 / (total_ms / repeat), n_tokens * 1000.0 / (total_ms / repeat));
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { print_usage(); return 2; }
    const std::string cmd = argv[1];
    try {
        if (cmd == "inspect") {
            if (argc < 3) { print_usage(); return 2; }
            const bool json = argc > 3 && std::strcmp(argv[3], "--json") == 0;
            return json ? cmd_inspect_json(argv[2]) : cmd_inspect(argv[2]);
        }
        if (cmd == "tokenize" || cmd == "spans") {
            TokOpts o;
            o.spans = (cmd == "spans");
            for (int i = 2; i < argc; ++i) {
                const std::string a = argv[i];
                auto value = [&]() -> std::string {
                    if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
                    return argv[++i];
                };
                if (a == "--text") o.text = value();
                else if (a == "--text-file") o.text_file = value();
                else if (a == "--batch") o.batch = value();
                else if (a == "--vocab") o.vocab = value();
                else if (a == "--pieces") o.pieces = true;
                else if (a == "--json") o.json = true;
                else if (a.rfind("--", 0) == 0) throw std::runtime_error("unknown flag " + a);
                else o.model = a;
            }
            return cmd_tokenize(o);
        }
        if (cmd == "vocab-dump") {
            if (argc < 4) { print_usage(); return 2; }
            zen::Tokenizer t;
            zen::GGUFFile g(argv[2]);
            t.load_from_gguf(g);
            t.save_flat(argv[3]);
            std::fprintf(stderr, "wrote %s (%zu tokens, %zu merges, %zu specials)\n", argv[3],
                         t.n_vocab(), t.n_merges(), t.n_specials());
            return 0;
        }
        if (cmd == "bench-tokenizer") {
            if (argc < 4) { print_usage(); return 2; }
            int repeat = 5;
            for (int i = 4; i < argc; ++i) {
                if (std::strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) repeat = std::atoi(argv[++i]);
            }
            return cmd_bench_tokenizer(argv[2], argv[3], repeat > 0 ? repeat : 1);
        }
        if (cmd == "-h" || cmd == "--help" || cmd == "help") { print_usage(); return 0; }
        std::fprintf(stderr, "unknown command '%s'\n", cmd.c_str());
        print_usage();
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
