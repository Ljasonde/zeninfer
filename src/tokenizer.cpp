// zeninfer - CPU LLM inference engine
// tokenizer.cpp : qwen2 pre-tokenizer + byte-level BPE.
#include "tokenizer.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "unicode.h"

namespace zen {
namespace {

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error("tokenizer: " + msg); }

inline uint32_t lower_ascii(uint32_t c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }

inline bool is_symbol_cp(uint32_t cp) {
    return !is_space_cp(cp) && !is_letter_cp(cp) && !is_number_cp(cp);
}

inline int hex_val(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

std::string hex_decode(const std::string& h) {
    if (h == "-") return std::string();
    if (h.size() % 2) fail("odd-length hex string");
    std::string out;
    out.reserve(h.size() / 2);
    for (size_t i = 0; i < h.size(); i += 2) {
        const int hi = hex_val(h[i]), lo = hex_val(h[i + 1]);
        if (hi < 0 || lo < 0) fail("bad hex digit");
        out.push_back(static_cast<char>((hi << 4) | lo));
    }
    return out;
}

std::string hex_encode(const std::string& s) {
    if (s.empty()) return "-";
    static const char* kHex = "0123456789abcdef";
    std::string out;
    out.resize(s.size() * 2);
    for (size_t i = 0; i < s.size(); ++i) {
        const unsigned char b = static_cast<unsigned char>(s[i]);
        out[2 * i] = kHex[b >> 4];
        out[2 * i + 1] = kHex[b & 0xF];
    }
    return out;
}

// Qwen2 keeps its vocabulary in GPT-2's byte-level alphabet: the 68 bytes that
// are not printable Latin-1 are moved into U+0100..U+0143 so that every token is
// a valid UTF-8 string. (The gguf copy of the vocabulary keeps that alphabet --
// verifying this is why the space token has no plain " " entry.)
constexpr uint32_t kByteLevelBase = 0x100;

struct ByteLevelTable {
    std::string enc[256];      // byte -> its byte-level UTF-8 spelling
    int16_t dec[kByteLevelBase + 0x44];  // byte-level codepoint -> byte, -1 if none

    ByteLevelTable() {
        for (auto& d : dec) d = -1;
        bool mapped[256] = {false};
        auto identity = [&](int b) {
            // The byte-level character IS codepoint b, so bytes >= 0x80 encode to
            // two UTF-8 bytes -- not to the raw byte.
            std::string s;
            utf8_append(s, static_cast<uint32_t>(b));
            enc[b] = s;
            dec[static_cast<size_t>(b)] = static_cast<int16_t>(b);
            mapped[b] = true;
        };
        for (int b = 0x21; b <= 0x7E; ++b) identity(b);   // ! .. ~
        for (int b = 0xA1; b <= 0xAC; ++b) identity(b);   // ¡ .. ¬
        for (int b = 0xAE; b <= 0xFF; ++b) identity(b);   // ® .. ÿ
        int spare = 0;
        for (int b = 0; b < 256; ++b) {
            if (mapped[b]) continue;
            const uint32_t cp = kByteLevelBase + static_cast<uint32_t>(spare++);
            std::string s;
            utf8_append(s, cp);
            enc[b] = s;
            dec[cp] = static_cast<int16_t>(b);
        }
    }
};

const ByteLevelTable& byte_level() {
    static const ByteLevelTable table;
    return table;
}

// Inverse of the byte-level alphabet: turns a token's bytes back into plain bytes.
std::string byte_level_decode(std::string_view lvl) {
    const ByteLevelTable& bl = byte_level();
    std::string out;
    out.reserve(lvl.size());
    size_t i = 0;
    while (i < lvl.size()) {
        uint32_t cp = 0;
        const size_t adv = utf8_decode(lvl.data(), lvl.size(), i, &cp);
        if (cp < sizeof(bl.dec) / sizeof(bl.dec[0]) && bl.dec[cp] >= 0) {
            out.push_back(static_cast<char>(bl.dec[cp]));
        } else {
            out.append(lvl.data() + i, adv);  // not a byte-level char: keep verbatim
        }
        i += adv;
    }
    return out;
}

}  // namespace

// ------------------------------------------------------------------ pre-tokenizer
// Implemented as an explicit scanner rather than std::regex: the pattern needs
// Unicode properties, and std::regex is both slow and locale-dependent.
// Alternative order matters -- the first alternative that matches at a position
// wins, exactly like the regex engine.
std::vector<std::pair<size_t, size_t>> pretokenize(std::string_view text) {
    std::vector<uint32_t> offsets;
    const std::vector<uint32_t> cps = utf8_decode_all(text, &offsets);
    const size_t n = cps.size();

    std::vector<std::pair<size_t, size_t>> spans;
    spans.reserve(n / 2 + 1);

    size_t i = 0;
    while (i < n) {
        const uint32_t c = cps[i];
        size_t end = i + 1;  // last-resort: one codepoint

        // 1. (?i:'s|'t|'re|'ve|'m|'ll|'d)
        bool matched = false;
        if (c == '\'') {
            auto at = [&](size_t k, char ch) {
                return k < n && lower_ascii(cps[k]) == static_cast<uint32_t>(ch);
            };
            if (at(i + 1, 's')) { end = i + 2; matched = true; }
            else if (at(i + 1, 't')) { end = i + 2; matched = true; }
            else if (at(i + 1, 'r') && at(i + 2, 'e')) { end = i + 3; matched = true; }
            else if (at(i + 1, 'v') && at(i + 2, 'e')) { end = i + 3; matched = true; }
            else if (at(i + 1, 'm')) { end = i + 2; matched = true; }
            else if (at(i + 1, 'l') && at(i + 2, 'l')) { end = i + 3; matched = true; }
            else if (at(i + 1, 'd')) { end = i + 2; matched = true; }
        }

        // 2. [^\r\n\p{L}\p{N}]?\p{L}+
        if (!matched) {
            if (is_letter_cp(c)) {
                size_t j = i + 1;
                while (j < n && is_letter_cp(cps[j])) ++j;
                end = j;
                matched = true;
            } else if (!is_newline_cp(c) && !is_number_cp(c) && i + 1 < n && is_letter_cp(cps[i + 1])) {
                // the optional prefix character: keeps " hello" in one piece
                size_t j = i + 2;
                while (j < n && is_letter_cp(cps[j])) ++j;
                end = j;
                matched = true;
            }
        }

        // 3. \p{N} -- a single digit/number character
        if (!matched && is_number_cp(c)) {
            end = i + 1;
            matched = true;
        }

        // 4.  ?[^\s\p{L}\p{N}]+[\r\n]*
        if (!matched) {
            for (int pass = 0; pass < 2 && !matched; ++pass) {
                size_t p = i;
                if (pass == 0) {
                    if (c != ' ') continue;  // greedy " ?" tries the space first
                    p = i + 1;
                }
                if (p < n && is_symbol_cp(cps[p])) {
                    size_t j = p;
                    while (j < n && is_symbol_cp(cps[j])) ++j;
                    while (j < n && is_newline_cp(cps[j])) ++j;
                    end = j;
                    matched = true;
                }
            }
        }

        // 5. \s*[\r\n]+ -- the run up to and including its LAST newline
        if (!matched && is_space_cp(c)) {
            size_t j = i;
            while (j < n && is_space_cp(cps[j])) ++j;
            size_t last_nl = 0;
            bool found = false;
            for (size_t t = j; t > i; --t) {
                if (is_newline_cp(cps[t - 1])) { last_nl = t - 1; found = true; break; }
            }
            if (found) {
                size_t e = last_nl + 1;
                while (e < n && is_newline_cp(cps[e])) ++e;
                end = e;
                matched = true;
            }
        }

        // 6. \s+(?!\S) -- a space run with no newline: leave one character for
        //    the next chunk so " x" can be taken by alternative 2.
        if (!matched && is_space_cp(c)) {
            size_t j = i;
            while (j < n && is_space_cp(cps[j])) ++j;
            if (j == n) {
                end = j;
                matched = true;
            } else if (j - i >= 2) {
                end = j - 1;
                matched = true;
            }
        }

        // 7. \s+ -- fallback for a single space followed by a non-space
        if (!matched && is_space_cp(c)) {
            size_t j = i;
            while (j < n && is_space_cp(cps[j])) ++j;
            end = j;
            matched = true;
        }

        if (end <= i) end = i + 1;
        spans.emplace_back(offsets[i], offsets[end]);
        i = end;
    }
    return spans;
}

// ------------------------------------------------------------------ Tokenizer
std::string_view Tokenizer::intern(const std::string& s, std::deque<std::string>* pool) {
    pool->push_back(s);
    return std::string_view(pool->back());
}

int32_t Tokenizer::id_of(std::string_view tok) const {
    auto it = token_to_id_.find(tok);
    return it == token_to_id_.end() ? -1 : it->second;
}

const std::string& Tokenizer::token(int32_t id) const {
    static const std::string kEmpty;
    if (id < 0 || static_cast<size_t>(id) >= tokens_.size()) return kEmpty;
    return dict_[static_cast<size_t>(id)];
}

std::string Tokenizer::token_repr(int32_t id) const {
    const std::string s = byte_level_decode(token(id));
    std::string out;
    out.reserve(s.size() + 2);
    const size_t n = s.size();
    for (size_t i = 0; i < n;) {
        const unsigned char b = static_cast<unsigned char>(s[i]);
        if (b < 0x80) {
            switch (b) {
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                case '\\': out += "\\\\"; break;
                default:
                    if (b < 0x20) {
                        char buf[8];
                        std::snprintf(buf, sizeof buf, "\\x%02x", b);
                        out += buf;
                    } else {
                        out.push_back(static_cast<char>(b));
                    }
            }
            ++i;
            continue;
        }
        // A byte-level vocabulary can split one character across several tokens, so a
        // single token may hold an incomplete UTF-8 sequence. Escape anything that does
        // not decode cleanly: CLI output has to stay valid UTF-8 for scripts to read.
        uint32_t cp = 0;
        const size_t adv = zen::utf8_decode(s.data(), n, i, &cp);
        if (adv > 1 && cp != 0xFFFD) {
            out.append(s, i, adv);
            i += adv;
        } else {
            char buf[8];
            std::snprintf(buf, sizeof buf, "\\x%02x", b);
            out += buf;
            ++i;
        }
    }
    return out;
}

void Tokenizer::add_merge(std::string_view left, std::string_view right) {
    std::string key(left);
    key.append(right.data(), right.size());
    if (merge_rank_.count(key)) return;  // keep the lowest rank
    const int32_t rank = static_cast<int32_t>(merges_.size());
    merges_.emplace_back(intern(std::string(left), &merge_pool_),
                         intern(std::string(right), &merge_pool_));
    merge_rank_.emplace(intern(key, &merge_pool_), rank);
}

void Tokenizer::finish() {
    tokens_.clear();
    tokens_.reserve(dict_.size());
    for (size_t i = 0; i < dict_.size(); ++i) tokens_.push_back(std::string_view(dict_[i]));

    token_to_id_.reserve(dict_.size() * 2);
    for (size_t i = 0; i < tokens_.size(); ++i) {
        if (tokens_[i].empty()) continue;  // padding rows in Qwen2's 151936-wide matrix
        token_to_id_.emplace(tokens_[i], static_cast<int32_t>(i));
    }

    // Specials: CONTROL (3) and USER_DEFINED (4) token types.
    special_ids_.clear();
    for (size_t i = 0; i < types_.size() && i < tokens_.size(); ++i) {
        const int32_t t = types_[i];
        if ((t == 3 || t == 4) && !tokens_[i].empty()) special_ids_.push_back(static_cast<int32_t>(i));
    }
    std::stable_sort(special_ids_.begin(), special_ids_.end(),
                     [&](int32_t a, int32_t b) { return tokens_[a].size() > tokens_[b].size(); });
}

void Tokenizer::load_from_gguf(const GGUFFile& f) {
    const std::vector<std::string> toks = f.get_str_array("tokenizer.ggml.tokens");
    if (toks.empty()) fail("no tokenizer.ggml.tokens in gguf");
    const std::vector<uint32_t> ttypes = f.get_u32_array("tokenizer.ggml.token_type");
    const std::vector<std::string> merges = f.get_str_array("tokenizer.ggml.merges");

    dict_.clear();
    merge_pool_.clear();
    merges_.clear();
    merge_rank_.clear();
    token_to_id_.clear();
    n_skipped_merges_ = 0;
    n_ambiguous_merges_ = 0;

    for (const auto& t : toks) dict_.push_back(t);
    types_.assign(toks.size(), 1);
    for (size_t i = 0; i < ttypes.size() && i < types_.size(); ++i) {
        types_[i] = static_cast<int32_t>(ttypes[i]);
    }
    finish();

    // A merge record is "left<SP>right" and both halves are raw bytes, so the
    // separating space is not necessarily the first or last one (" " + "t"
    // serialises as "  t"). Pick the split whose halves are all real tokens.
    for (const auto& m : merges) {
        size_t chosen = std::string::npos;
        size_t alternatives = 0;
        for (size_t p = 0; p < m.size(); ++p) {
            if (m[p] != ' ') continue;
            if (p == 0 || p + 1 >= m.size()) continue;
            const std::string_view left(m.data(), p);
            const std::string_view right(m.data() + p + 1, m.size() - p - 1);
            if (id_of(left) < 0 || id_of(right) < 0) continue;
            std::string joined(m.data(), p);
            joined.append(m.data() + p + 1, m.size() - p - 1);
            if (id_of(joined) < 0) continue;
            if (chosen == std::string::npos) {
                chosen = p;
            } else {
                ++alternatives;
            }
        }
        if (chosen == std::string::npos) {
            ++n_skipped_merges_;
            continue;
        }
        if (alternatives) ++n_ambiguous_merges_;
        add_merge(std::string_view(m).substr(0, chosen), std::string_view(m).substr(chosen + 1));
    }
}

void Tokenizer::load_from_flat(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) fail("cannot open " + path);
    std::string line;
    std::getline(in, line);
    if (line.rfind("zenvocab 1", 0) != 0) fail("not a zenvocab file: " + path);

    dict_.clear();
    merge_pool_.clear();
    merges_.clear();
    merge_rank_.clear();
    token_to_id_.clear();
    n_skipped_merges_ = 0;
    n_ambiguous_merges_ = 0;

    size_t n_tokens = 0, n_merges = 0;
    std::vector<std::pair<std::string, std::string>> merges;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        if (line.rfind("tokens ", 0) == 0) {
            n_tokens = std::stoull(line.substr(7));
            types_.reserve(n_tokens);
            for (size_t k = 0; k < n_tokens; ++k) {
                if (!std::getline(in, line)) fail("truncated token list");
                if (!line.empty() && line.back() == '\r') line.pop_back();
                const size_t sp = line.rfind(' ');
                if (sp == std::string::npos) fail("bad token line");
                dict_.push_back(hex_decode(line.substr(0, sp)));
                types_.push_back(std::stoi(line.substr(sp + 1)));
            }
        } else if (line.rfind("merges ", 0) == 0) {
            n_merges = std::stoull(line.substr(7));
            merges.reserve(n_merges);
            for (size_t k = 0; k < n_merges; ++k) {
                if (!std::getline(in, line)) fail("truncated merge list");
                if (!line.empty() && line.back() == '\r') line.pop_back();
                const size_t sp = line.find(' ');
                if (sp == std::string::npos) fail("bad merge line");
                merges.emplace_back(hex_decode(line.substr(0, sp)), hex_decode(line.substr(sp + 1)));
            }
        }
    }
    finish();
    for (const auto& m : merges) add_merge(m.first, m.second);
}

void Tokenizer::save_flat(const std::string& path) const {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) fail("cannot write " + path);
    std::fprintf(f, "zenvocab 1\n");
    std::fprintf(f, "tokens %zu\n", dict_.size());
    for (size_t i = 0; i < dict_.size(); ++i) {
        const std::string hex = hex_encode(dict_[i]);
        std::fprintf(f, "%s %d\n", hex.c_str(), i < types_.size() ? types_[i] : 1);
    }
    std::fprintf(f, "merges %zu\n", merges_.size());
    for (const auto& m : merges_) {
        const std::string l = hex_encode(std::string(m.first));
        const std::string r = hex_encode(std::string(m.second));
        std::fprintf(f, "%s %s\n", l.c_str(), r.c_str());
    }
    std::fclose(f);
}

int32_t Tokenizer::match_special(std::string_view text, size_t pos) const {
    for (int32_t id : special_ids_) {
        const std::string& s = token(id);
        if (pos + s.size() <= text.size() && std::memcmp(text.data() + pos, s.data(), s.size()) == 0) {
            return id;
        }
    }
    return -1;
}

std::vector<int32_t> Tokenizer::encode(std::string_view text, bool parse_special) const {
    return encode(text, nullptr, parse_special);
}

std::vector<int32_t> Tokenizer::encode(std::string_view text, std::vector<std::string>* pieces,
                                       bool parse_special) const {
    std::vector<int32_t> ids;
    ids.reserve(text.size() / 2 + 1);
    if (pieces) pieces->clear();

    // Pre-tokenize, then BPE each span independently: merges never cross a span
    // boundary, which is the whole point of the split.
    auto emit = [&](size_t begin, size_t end) {
        if (end <= begin) return;
        const auto spans = pretokenize(text.substr(begin, end - begin));
        for (const auto& sp : spans) {
            encode_span(text, begin + sp.first, begin + sp.second, &ids, pieces);
        }
    };

    size_t segment = 0;
    size_t i = 0;
    while (i < text.size()) {
        int32_t sp = -1;
        if (parse_special && text[i] == '<') sp = match_special(text, i);
        if (sp >= 0) {
            emit(segment, i);
            ids.push_back(sp);
            if (pieces) pieces->push_back(byte_level_decode(token(sp)));
            i += token(sp).size();
            segment = i;
        } else {
            ++i;
        }
    }
    emit(segment, text.size());
    return ids;
}

void Tokenizer::encode_span(std::string_view text, size_t begin, size_t end,
                            std::vector<int32_t>* out, std::vector<std::string>* pieces) const {
    struct Sym {
        uint32_t start, len;
    };
    const ByteLevelTable& bl = byte_level();

    // Translate the raw bytes into the byte-level alphabet first: the merges and
    // the vocabulary are both written in it, so BPE runs over that string.
    std::string lvl;
    lvl.reserve((end - begin) * 2);
    std::vector<Sym> syms;
    syms.reserve(end - begin);
    for (size_t p = begin; p < end; ++p) {
        const std::string& e = bl.enc[static_cast<unsigned char>(text[p])];
        syms.push_back({static_cast<uint32_t>(lvl.size()), static_cast<uint32_t>(e.size())});
        lvl.append(e);
    }
    const char* base = lvl.data();

    // Repeatedly apply the lowest-ranked adjacent merge (leftmost on ties),
    // which is what the HF BPE implementation does.
    while (syms.size() >= 2) {
        int best = -1;
        int32_t best_rank = 0x7FFFFFFF;
        for (size_t k = 0; k + 1 < syms.size(); ++k) {
            const std::string_view key(base + syms[k].start, syms[k].len + syms[k + 1].len);
            auto it = merge_rank_.find(key);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best = static_cast<int>(k);
            }
        }
        if (best < 0) break;
        syms[static_cast<size_t>(best)].len += syms[static_cast<size_t>(best) + 1].len;
        syms.erase(syms.begin() + best + 1);
    }

    for (const auto& s : syms) {
        const std::string_view piece(base + s.start, s.len);
        const int32_t id = id_of(piece);
        if (id >= 0) {
            out->push_back(id);
            if (pieces) pieces->push_back(byte_level_decode(piece));
            continue;
        }
        // Byte fallback: Qwen keeps a token for every single byte value.
        for (uint32_t k = 0; k < s.len;) {
            uint32_t cp = 0;
            const size_t adv = utf8_decode(piece.data(), piece.size(), k, &cp);
            const std::string_view one(piece.data() + k, adv);
            out->push_back(id_of(one));  // -1 surfaces a genuine vocabulary hole
            if (pieces) pieces->push_back(byte_level_decode(one));
            k += static_cast<uint32_t>(adv);
        }
    }
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
    std::string out;
    for (int32_t id : ids) {
        if (id < 0 || static_cast<size_t>(id) >= tokens_.size()) continue;
        out += byte_level_decode(dict_[static_cast<size_t>(id)]);
    }
    return out;
}

}  // namespace zen
