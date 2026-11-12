// zeninfer - CPU LLM inference engine
// tokenizer.h : byte-level BPE with the qwen2 pre-tokenizer.
//
// The vocabulary is stored as raw bytes (llama.cpp converts Qwen's byte-level
// alphabet back to plain bytes when writing the gguf), so a token is simply a
// byte string and BPE operates on bytes. That makes the merge lookup a plain
// string compare instead of a per-byte remap table.
#pragma once

#include <cstdint>
#include <deque>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gguf.h"

namespace zen {

// Splits `text` into the spans that the qwen2 pre-tokenizer regex produces:
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|
//   ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
// Returns [begin,end) byte ranges that cover the input exactly.
std::vector<std::pair<size_t, size_t>> pretokenize(std::string_view text);

class Tokenizer {
  public:
    // Build from a model file: tokenizer.ggml.{tokens,merges,token_type}.
    void load_from_gguf(const GGUFFile& f);
    // Build from the flat dump written by tools/dump_vocab.py ("zenvocab").
    // Same information without needing the 600 MB model, so tokenizer parity
    // can be tested on its own.
    void load_from_flat(const std::string& path);

    bool empty() const { return tokens_.empty(); }
    size_t n_vocab() const { return tokens_.size(); }
    size_t n_merges() const { return merge_rank_.size(); }
    size_t n_skipped_merges() const { return n_skipped_merges_; }
    size_t n_ambiguous_merges() const { return n_ambiguous_merges_; }
    size_t n_specials() const { return special_ids_.size(); }

    int32_t id_of(std::string_view tok) const;
    // Byte contents of a token; empty string for out-of-range ids.
    const std::string& token(int32_t id) const;
    // Token rendered for humans: raw control characters escaped.
    std::string token_repr(int32_t id) const;
    // Writes the vocabulary in the flat "zenvocab" format, so the gguf copy of
    // the tokenizer can be diffed against tokenizer.json without a model file.
    void save_flat(const std::string& path) const;

    std::vector<int32_t> encode(std::string_view text, bool parse_special = true) const;
    // `pieces`, when non-null, receives the byte string of each produced token.
    std::vector<int32_t> encode(std::string_view text, std::vector<std::string>* pieces,
                                bool parse_special = true) const;
    std::string decode(const std::vector<int32_t>& ids) const;

  private:
    // Interns `s`, returning a view with a stable address (deque never
    // invalidates references on push_back).
    std::string_view intern(const std::string& s, std::deque<std::string>* pool);
    void add_merge(std::string_view left, std::string_view right);
    void finish();
    int32_t match_special(std::string_view text, size_t pos) const;
    void encode_span(std::string_view text, size_t begin, size_t end,
                     std::vector<int32_t>* out, std::vector<std::string>* pieces) const;

    std::deque<std::string> dict_;        // owns token byte strings
    std::deque<std::string> merge_pool_;  // owns merge keys and halves
    std::vector<std::string_view> tokens_;
    std::vector<int32_t> types_;
    std::vector<int32_t> special_ids_;
    std::vector<std::pair<std::string_view, std::string_view>> merges_;  // rank order
    std::unordered_map<std::string_view, int32_t> token_to_id_;
    std::unordered_map<std::string_view, int32_t> merge_rank_;
    size_t n_skipped_merges_ = 0;
    size_t n_ambiguous_merges_ = 0;
};

}  // namespace zen
