// zeninfer - CPU LLM inference engine
// unicode.h : UTF-8 codec + the three character predicates the qwen2
//             pre-tokenizer regex needs. Tables come from unicode_tables.h,
//             which tools/gen_unicode_tables.py regenerates.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "unicode_tables.h"

namespace zen {

inline bool in_ranges(const CodeRange* r, size_t n, uint32_t cp) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        if (r[mid].hi < cp) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo < n && r[lo].lo <= cp;
}

inline bool is_letter_cp(uint32_t cp) { return in_ranges(kLetterRanges, kLetterRanges_len, cp); }
inline bool is_number_cp(uint32_t cp) { return in_ranges(kNumberRanges, kNumberRanges_len, cp); }
// ECMAScript \s: an explicit list, not a Unicode property.
inline bool is_space_cp(uint32_t cp) {
    return in_ranges(kWhitespaceRanges, kWhitespaceRanges_len, cp);
}
inline bool is_newline_cp(uint32_t cp) { return cp == '\r' || cp == '\n'; }

// Decodes the sequence starting at s[i]. Returns the number of bytes consumed
// (>= 1). Ill-formed input decodes to U+FFFD and consumes a single byte, which
// keeps the caller's byte offsets exact.
inline size_t utf8_decode(const char* s, size_t n, size_t i, uint32_t* out) {
    const unsigned char b0 = static_cast<unsigned char>(s[i]);
    auto bad = [&](size_t adv) {
        *out = 0xFFFD;
        return adv;
    };
    if (b0 < 0x80) {
        *out = b0;
        return 1;
    }
    size_t need;
    uint32_t cp;
    if ((b0 & 0xE0) == 0xC0) {
        need = 1;
        cp = b0 & 0x1Fu;
    } else if ((b0 & 0xF0) == 0xE0) {
        need = 2;
        cp = b0 & 0x0Fu;
    } else if ((b0 & 0xF8) == 0xF0) {
        need = 3;
        cp = b0 & 0x07u;
    } else {
        return bad(1);
    }
    if (i + need >= n) return bad(1);   // truncated sequence
    for (size_t k = 1; k <= need; ++k) {
        const unsigned char bk = static_cast<unsigned char>(s[i + k]);
        if ((bk & 0xC0) != 0x80) return bad(1);
        cp = (cp << 6) | (bk & 0x3Fu);
    }
    // reject overlong encodings, surrogates and out-of-range values
    static const uint32_t kMin[4] = {0, 0x80, 0x800, 0x10000};
    if (cp < kMin[need] || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) return bad(1);
    *out = cp;
    return need + 1;
}

inline void utf8_append(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

// Decodes all of `s`. `offsets` (may be null) receives byte_end[i] for every
// codepoint, with a final entry equal to s.size(), so byte boundaries are
// recoverable: cp i spans [offsets[i], offsets[i+1]).
inline std::vector<uint32_t> utf8_decode_all(std::string_view s, std::vector<uint32_t>* offsets) {
    std::vector<uint32_t> cps;
    cps.reserve(s.size());
    if (offsets) {
        offsets->clear();
        offsets->reserve(s.size() + 1);
    }
    size_t i = 0;
    while (i < s.size()) {
        uint32_t cp = 0;
        const size_t adv = utf8_decode(s.data(), s.size(), i, &cp);
        i += adv;
        cps.push_back(cp);
        if (offsets) offsets->push_back(static_cast<uint32_t>(i));
    }
    if (offsets) {
        // offsets currently holds end positions; prepend 0 so [i] is the start.
        offsets->insert(offsets->begin(), 0);
    }
    return cps;
}

}  // namespace zen
