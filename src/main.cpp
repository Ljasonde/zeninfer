// zeninfer - main.cpp : CLI entry point.
#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "gguf.h"
#include "kernels.h"
#include "model.h"
#include "sample.h"
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
        "  zeninfer logits     <model.gguf> [--text S | --text-file F | --ids 1,2,3]\n"
        "                                  [--out F.bin] [--all] [--topk N] [--cache]\n"
        "                                  forward pass, dump logits\n"
        "  zeninfer generate   <model.gguf> [--text S | --text-file F | --ids 1,2,3]\n"
        "                                  [--max-new N] [--temp T] [--top-k K] [--top-p P]\n"
        "                                  [--repeat-penalty R] [--seed S] [--no-cache] [--json]\n"
        "                                  KV-cached incremental decode + sampling\n"
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
    if (o.text_file.empty() && !zen::utf8_valid(text)) {
        std::fprintf(stderr,
                     "warning: --text is not valid UTF-8, so the shell transcoded it "
                     "(Windows PowerShell passes arguments in the OEM code page).\n"
                     "         Use --text-file with a UTF-8 file for non-ASCII text.\n");
    }
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
    if (o.json) {
        // Same shape as the --batch rows, so a caller can parse both the same way.
        std::printf("{\"ids\":[");
        for (size_t k = 0; k < ids.size(); ++k) std::printf("%s%d", k ? "," : "", ids[k]);
        std::printf("],\"pieces\":[");
        for (size_t k = 0; k < pieces.size(); ++k) {
            std::printf("%s\"%s\"", k ? "," : "", json_escape(pieces[k]).c_str());
        }
        std::printf("],\"text\":\"%s\"}\n", json_escape(text).c_str());
        return 0;
    }
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

int cmd_logits(const std::string& model_path, const std::string& text, const std::string& text_file,
               const std::string& ids_arg, const std::string& out, int topk, bool all, int greedy = 0,
               const std::string& text_out_path = "", bool use_cache = false) {
    zen::GGUFFile g(model_path);
    zen::Tokenizer tok;
    tok.load_from_gguf(g);

    std::vector<int32_t> ids;
    if (!ids_arg.empty()) {
        size_t i = 0;
        while (i < ids_arg.size()) {
            size_t j = ids_arg.find(',', i);
            if (j == std::string::npos) j = ids_arg.size();
            const std::string piece = ids_arg.substr(i, j - i);
            if (!piece.empty()) ids.push_back(static_cast<int32_t>(std::strtol(piece.c_str(), nullptr, 10)));
            i = j + 1;
        }
    } else {
        const std::string src = text_file.empty() ? text : read_file(text_file);
        if (src.empty()) throw std::runtime_error("no input: pass --text, --text-file or --ids");
        ids = tok.encode(src, true);
    }
    if (ids.empty()) throw std::runtime_error("empty token sequence");

    const auto t_begin = std::chrono::steady_clock::now();
    zen::Model model(g);
    const auto t_loaded = std::chrono::steady_clock::now();

    std::vector<float> logits;
    if (use_cache) {
        // Same logits, produced one position at a time through the KV cache. This
        // is what makes the two paths comparable row by row: `--all` without the
        // flag is a single full recompute, with it the same rows come out of S
        // single-token steps.
        if (all) {
            const size_t V = static_cast<size_t>(model.cfg().n_vocab);
            model.reserve_cache(static_cast<int>(ids.size()) + 1);
            logits.resize(ids.size() * V);
            for (size_t t = 0; t < ids.size(); ++t) {
                const std::vector<float>& row = model.eval({ids[t]});
                std::memcpy(logits.data() + t * V, row.data(), V * sizeof(float));
            }
        } else {
            model.reserve_cache(static_cast<int>(ids.size()) + 1);
            const std::vector<float>& row = model.eval(ids);
            logits.assign(row.begin(), row.end());
        }
    } else {
        logits = all ? model.forward_all(ids) : model.forward(ids);
    }
    const auto t_done = std::chrono::steady_clock::now();

    const auto& cfg = model.cfg();
    const size_t n_vocab = static_cast<size_t>(cfg.n_vocab);
    const double load_ms = std::chrono::duration<double, std::milli>(t_loaded - t_begin).count();
    const double run_ms = std::chrono::duration<double, std::milli>(t_done - t_loaded).count();

    const float* last = logits.data() + (logits.size() - n_vocab);
    double sum = 0.0, max_l = -1e30, min_l = 1e30;
    for (size_t i = 0; i < n_vocab; ++i) {
        max_l = std::max<double>(max_l, last[i]);
        min_l = std::min<double>(min_l, last[i]);
    }
    for (size_t i = 0; i < n_vocab; ++i) sum += std::exp(static_cast<double>(last[i]) - max_l);

    std::vector<uint32_t> order(n_vocab);
    for (size_t i = 0; i < n_vocab; ++i) order[i] = static_cast<uint32_t>(i);
    const int k = std::max(1, std::min<int>(topk, static_cast<int>(n_vocab)));
    std::partial_sort(order.begin(), order.begin() + k, order.end(),
                      [&](uint32_t a, uint32_t b) { return last[a] > last[b]; });

    std::printf("model            : %s\n", model_path.c_str());
    std::printf("arch             : %s (%d layers, %d embd, %d heads / %d kv)\n", model.arch().c_str(),
                cfg.n_layer, cfg.n_embd, cfg.n_head, cfg.n_head_kv);
    std::printf("weights          : %.2f MiB on disk -> %.2f MiB f32\n",
                model.source_bytes() / 1048576.0, model.resident_bytes() / 1048576.0);
    std::printf("load             : %.1f ms\n", load_ms);
    std::printf("tokens           : %zu\n", ids.size());
    std::printf("forward          : %.1f ms (%.2f ms/token)\n", run_ms, run_ms / static_cast<double>(ids.size()));
    std::printf("logits           : min %.4f  max %.4f  argmax %u\n", min_l, max_l, order[0]);
    std::printf("softmax Z        : %.6f (sum of exp(logit - max))\n", sum);
    std::printf("top-%d\n", k);
    for (int i = 0; i < k; ++i) {
        const uint32_t id = order[static_cast<size_t>(i)];
        const double p = std::exp(static_cast<double>(last[id]) - max_l) / sum;
        std::printf("  %2d. id %6u  logit %8.4f  p %.6f  %s\n", i + 1, id, last[id], p,
                    tok.token_repr(id).c_str());
    }

    if (!out.empty()) {
        std::ofstream f(out, std::ios::binary);
        if (!f) throw std::runtime_error("cannot write " + out);
        f.write(reinterpret_cast<const char*>(logits.data()),
                static_cast<std::streamsize>(logits.size() * sizeof(float)));
        std::fprintf(stderr, "wrote %s (%.2f MiB, %zu floats)\n", out.c_str(),
                     logits.size() * sizeof(float) / 1048576.0, logits.size());
    }

    // Greedy sanity check. Deliberately re-runs the full prefill every step: M2 is
    // about proving the math, and an O(n^2) loop is the version with the fewest ways
    // to be wrong. The KV-cache version replaces this in M3.
    if (greedy > 0) {
        const int32_t eos = static_cast<int32_t>(g.get_u32("tokenizer.ggml.eos_token_id", 151645));
        std::printf("\ngreedy (%d max, eos %d, full recompute per step)\n", greedy, eos);
        std::vector<float> lg = logits;
        std::vector<int32_t> cur = ids;
        std::string text_out;
        int produced = 0;
        const auto t_gen = std::chrono::steady_clock::now();
        for (int step = 0; step < greedy; ++step) {
            const float* row = lg.data() + (lg.size() - n_vocab);
            int32_t best = 0;
            for (size_t i = 1; i < n_vocab; ++i) {
                if (row[i] > row[best]) best = static_cast<int32_t>(i);
            }
            if (best == eos) break;
            const std::string piece = tok.decode({best});
            text_out += piece;
            ++produced;
            cur.push_back(best);
            lg = model.forward(cur);
        }
        const double gen_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_gen).count();
        std::printf("generated        : %d tokens in %.1f ms (%.1f ms/token, no cache)\n", produced, gen_ms,
                    produced ? gen_ms / produced : 0.0);
        std::printf("text             : %s\n", text_out.c_str());
        if (!text_out_path.empty()) {
            std::ofstream tf(text_out_path, std::ios::binary);
            if (!tf) throw std::runtime_error("cannot write " + text_out_path);
            tf.write(text_out.data(), static_cast<std::streamsize>(text_out.size()));
            std::fprintf(stderr, "wrote %s (%zu bytes utf-8)\n", text_out_path.c_str(), text_out.size());
        }
    }
    return 0;
}

double ms_between(const std::chrono::steady_clock::time_point& a,
                  const std::chrono::steady_clock::time_point& b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

std::vector<int32_t> parse_ids(const std::string& s) {
    std::vector<int32_t> ids;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        const std::string piece = s.substr(i, j - i);
        if (!piece.empty()) ids.push_back(static_cast<int32_t>(std::strtol(piece.c_str(), nullptr, 10)));
        i = j + 1;
    }
    return ids;
}

struct GenOpts {
    std::string model, text, text_file, ids_arg, text_out;
    int max_new = 64;
    int top_k = 0;
    int repeat_n = 64;
    float temp = 0.0f, top_p = 1.0f, repeat_penalty = 1.0f;
    uint64_t seed = 0;
    bool no_cache = false, json = false;
};

int cmd_generate(const GenOpts& o) {
    zen::GGUFFile g(o.model);
    zen::Tokenizer tok;
    tok.load_from_gguf(g);

    std::vector<int32_t> prompt;
    if (!o.ids_arg.empty()) {
        prompt = parse_ids(o.ids_arg);
    } else {
        const std::string src = o.text_file.empty() ? o.text : read_file(o.text_file);
        if (src.empty()) throw std::runtime_error("no input: pass --text, --text-file or --ids");
        prompt = tok.encode(src, true);
    }
    if (prompt.empty()) throw std::runtime_error("empty token sequence");

    const auto t_load0 = std::chrono::steady_clock::now();
    zen::Model model(g);
    const auto t_load1 = std::chrono::steady_clock::now();

    const auto& cfg = model.cfg();
    const int32_t eos = static_cast<int32_t>(g.get_u32("tokenizer.ggml.eos_token_id", 151645));
    const int64_t want = static_cast<int64_t>(prompt.size()) + o.max_new + 8;
    const int ctx = static_cast<int>(std::min<int64_t>(cfg.n_ctx, std::max<int64_t>(want, 32)));
    if (!o.no_cache) model.reserve_cache(ctx);

    std::vector<float> logits;
    const auto t_pre0 = std::chrono::steady_clock::now();
    if (o.no_cache) {
        logits = model.forward(prompt);
    } else {
        const std::vector<float>& row = model.eval(prompt);
        logits.assign(row.begin(), row.end());
    }
    const auto t_pre1 = std::chrono::steady_clock::now();

    zen::Rng rng(o.seed);
    zen::SamplingParams sp;
    sp.temp = o.temp;
    sp.top_k = o.top_k;
    sp.top_p = o.top_p;
    sp.repeat_penalty = o.repeat_penalty;
    sp.repeat_last_n = o.repeat_n;

    std::vector<int32_t> gen;
    std::string out;
    double decode_ms = 0.0;
    std::string stop = "max-new";
    for (int step = 0; step < o.max_new; ++step) {
        const int32_t next = zen::sample_token(logits.data(), cfg.n_vocab, sp, gen, rng);
        if (next == eos) {
            stop = "eos";
            break;
        }
        gen.push_back(next);
        out += tok.decode({next});

        const auto d0 = std::chrono::steady_clock::now();
        if (o.no_cache) {
            std::vector<int32_t> cur = prompt;
            cur.insert(cur.end(), gen.begin(), gen.end());
            logits = model.forward(cur);
        } else {
            const std::vector<float>& row = model.eval({next});
            logits.assign(row.begin(), row.end());
        }
        const auto d1 = std::chrono::steady_clock::now();
        decode_ms += ms_between(d0, d1);
    }
    const double prefill_ms = ms_between(t_pre0, t_pre1);
    const double load_ms = ms_between(t_load0, t_load1);

    if (o.json) {
        std::printf("{\"prompt_tokens\":%zu,\"prefill_ms\":%.3f,\"decode_tokens\":%zu,"
                    "\"decode_ms\":%.3f,\"stop\":\"%s\",\"ids\":[",
                    prompt.size(), prefill_ms, gen.size(), decode_ms, stop.c_str());
        for (size_t i = 0; i < gen.size(); ++i) std::printf("%s%d", i ? "," : "", gen[i]);
        std::printf("],\"text\":\"%s\"}\n", json_escape(out).c_str());
        return 0;
    }

    std::printf("model            : %s\n", o.model.c_str());
    std::printf("weights          : %.2f MiB on disk -> %.2f MiB f32 (load %.1f ms)\n",
                model.source_bytes() / 1048576.0, model.resident_bytes() / 1048576.0, load_ms);
    if (o.no_cache) {
        std::printf("kv cache         : disabled (full recompute per token)\n");
    } else {
        std::printf("kv cache         : %d pos x %d layers x %d kv-dim = %.2f MiB\n", ctx, cfg.n_layer,
                    cfg.n_head_kv * cfg.head_dim, model.cache_bytes() / 1048576.0);
    }
    std::printf("prompt           : %zu tokens\n", prompt.size());
    std::printf("prefill          : %.1f ms (%.2f ms/token, %.2f tok/s)\n", prefill_ms,
                prefill_ms / static_cast<double>(prompt.size()),
                1000.0 * static_cast<double>(prompt.size()) / prefill_ms);
    std::printf("decode           : %zu tokens in %.1f ms (%.2f ms/token, %.2f tok/s)\n", gen.size(),
                decode_ms, gen.empty() ? 0.0 : decode_ms / static_cast<double>(gen.size()),
                decode_ms > 0.0 ? 1000.0 * static_cast<double>(gen.size()) / decode_ms : 0.0);
    std::printf("stop             : %s\n", stop.c_str());
    std::printf("text             : %s\n", out.c_str());
    if (!o.text_out.empty()) {
        std::ofstream tf(o.text_out, std::ios::binary);
        if (!tf) throw std::runtime_error("cannot write " + o.text_out);
        tf.write(out.data(), static_cast<std::streamsize>(out.size()));
        std::fprintf(stderr, "wrote %s (%zu bytes utf-8)\n", o.text_out.c_str(), out.size());
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { print_usage(); return 2; }
    const std::string cmd = argv[1];

    // Defaults for the compute path; every command can override with --threads/--impl.
    {
        const char* t = std::getenv("ZEN_THREADS");
        zen::set_threads(t ? std::atoi(t) : 1);
        const char* im = std::getenv("ZEN_IMPL");
        zen::set_impl(zen::cpu_has_avx2() ? zen::Impl::Avx2 : zen::Impl::Scalar);
        if (im && std::strcmp(im, "scalar") == 0) zen::set_impl(zen::Impl::Scalar);
        if (im && std::strcmp(im, "avx2") == 0) zen::set_impl(zen::Impl::Avx2);
    }
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
        if (cmd == "logits") {
            std::string model_path, text, text_file, ids_arg, out, text_out;
            int topk = 8;
            int greedy = 0;
            bool all = false, use_cache = false;
            for (int i = 2; i < argc; ++i) {
                const std::string a = argv[i];
                auto value = [&]() -> std::string {
                    if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
                    return argv[++i];
                };
                if (a == "--text") text = value();
                else if (a == "--text-file") text_file = value();
                else if (a == "--ids") ids_arg = value();
                else if (a == "--out") out = value();
                else if (a == "--topk") topk = std::atoi(value().c_str());
                else if (a == "--greedy") greedy = std::atoi(value().c_str());
                else if (a == "--text-out") text_out = value();
                else if (a == "--all") all = true;
                else if (a == "--cache") use_cache = true;
                else if (a == "--threads") zen::set_threads(std::atoi(value().c_str()));
                else if (a == "--impl") {
                    const std::string v = value();
                    zen::set_impl(v == "scalar" ? zen::Impl::Scalar : zen::Impl::Avx2);
                }
                else if (a.rfind("--", 0) == 0) throw std::runtime_error("unknown flag " + a);
                else model_path = a;
            }
            if (model_path.empty()) throw std::runtime_error("missing model path");
            return cmd_logits(model_path, text, text_file, ids_arg, out, topk, all, greedy, text_out,
                              use_cache);
        }
        if (cmd == "generate") {
            GenOpts o;
            for (int i = 2; i < argc; ++i) {
                const std::string a = argv[i];
                auto value = [&]() -> std::string {
                    if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
                    return argv[++i];
                };
                if (a == "--text") o.text = value();
                else if (a == "--text-file") o.text_file = value();
                else if (a == "--ids") o.ids_arg = value();
                else if (a == "--max-new") o.max_new = std::atoi(value().c_str());
                else if (a == "--temp" || a == "--temperature") o.temp = static_cast<float>(std::atof(value().c_str()));
                else if (a == "--top-k") o.top_k = std::atoi(value().c_str());
                else if (a == "--top-p") o.top_p = static_cast<float>(std::atof(value().c_str()));
                else if (a == "--repeat-penalty") o.repeat_penalty = static_cast<float>(std::atof(value().c_str()));
                else if (a == "--repeat-last-n") o.repeat_n = std::atoi(value().c_str());
                else if (a == "--seed") o.seed = std::strtoull(value().c_str(), nullptr, 10);
                else if (a == "--text-out") o.text_out = value();
                else if (a == "--no-cache") o.no_cache = true;
                else if (a == "--json") o.json = true;
                else if (a == "--threads") zen::set_threads(std::atoi(value().c_str()));
                else if (a == "--impl") {
                    const std::string v = value();
                    zen::set_impl(v == "scalar" ? zen::Impl::Scalar : zen::Impl::Avx2);
                }
                else if (a.rfind("--", 0) == 0) throw std::runtime_error("unknown flag " + a);
                else o.model = a;
            }
            if (o.model.empty()) throw std::runtime_error("missing model path");
            if (o.max_new < 0) throw std::runtime_error("--max-new must be >= 0");
            return cmd_generate(o);
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
