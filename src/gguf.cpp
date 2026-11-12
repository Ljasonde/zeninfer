// zeninfer - gguf.cpp : GGUF v3 parsing + mmap + dequantization kernels.
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace zen {

namespace {

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error("gguf: " + msg); }

// Reads forward through a byte range, bounds-checked.
struct Cursor {
    const uint8_t* p;
    const uint8_t* end;

    void need(size_t n) const {
        if (p + n > end) fail("unexpected end of metadata");
    }
    template <typename T>
    T num() {
        need(sizeof(T));
        T v = read_le<T>(p);
        p += sizeof(T);
        return v;
    }
    std::string str() {
        uint64_t n = num<uint64_t>();
        need(static_cast<size_t>(n));
        std::string s(reinterpret_cast<const char*>(p), static_cast<size_t>(n));
        p += n;
        return s;
    }
};

Value read_value(Cursor& c, uint32_t kind, int depth);

Value read_value(Cursor& c, uint32_t kind, int depth) {
    if (depth > 8) fail("metadata nesting too deep");
    Value v;
    switch (kind) {
        case 0: v.kind = Value::Kind::U8;   v.u = c.num<uint8_t>();  break;
        case 1: v.kind = Value::Kind::I8;   v.u = static_cast<uint64_t>(static_cast<int64_t>(c.num<int8_t>())); break;
        case 2: v.kind = Value::Kind::U16;  v.u = c.num<uint16_t>(); break;
        case 3: v.kind = Value::Kind::I16;  v.u = static_cast<uint64_t>(static_cast<int64_t>(c.num<int16_t>())); break;
        case 4: v.kind = Value::Kind::U32;  v.u = c.num<uint32_t>(); break;
        case 5: v.kind = Value::Kind::I32;  v.u = static_cast<uint64_t>(static_cast<int64_t>(c.num<int32_t>())); break;
        case 6: v.kind = Value::Kind::F32;  v.f = c.num<float>();    break;
        case 7: v.kind = Value::Kind::Bool; v.u = c.num<uint8_t>() ? 1 : 0; break;
        case 8: v.kind = Value::Kind::Str;  v.s = c.str();           break;
        case 9: {
            v.kind = Value::Kind::Arr;
            uint32_t atype = c.num<uint32_t>();
            uint64_t alen = c.num<uint64_t>();
            if (alen > (1u << 24)) fail("array too long");
            v.arr.reserve(static_cast<size_t>(alen));
            for (uint64_t i = 0; i < alen; ++i) v.arr.push_back(read_value(c, atype, depth + 1));
            break;
        }
        case 10: v.kind = Value::Kind::U64; v.u = c.num<uint64_t>(); break;
        case 11: v.kind = Value::Kind::I64; v.u = static_cast<uint64_t>(c.num<int64_t>()); break;
        case 12: v.kind = Value::Kind::F64; v.f = c.num<double>();   break;
        default: fail("unknown metadata value type " + std::to_string(kind));
    }
    return v;
}

}  // namespace

// ---------------------------------------------------------------- Value accessors
uint64_t Value::as_u64() const {
    switch (kind) {
        case Kind::F32:
        case Kind::F64: return static_cast<uint64_t>(f);
        case Kind::Str:
        case Kind::Arr: fail("value is not numeric");
        default: return u;
    }
}
int64_t Value::as_i64() const {
    switch (kind) {
        case Kind::F32:
        case Kind::F64: return static_cast<int64_t>(f);
        case Kind::Str:
        case Kind::Arr: fail("value is not numeric");
        default: return static_cast<int64_t>(u);
    }
}
double Value::as_f64() const {
    switch (kind) {
        case Kind::F32:
        case Kind::F64: return f;
        case Kind::Str:
        case Kind::Arr: fail("value is not numeric");
        default: return static_cast<double>(static_cast<int64_t>(u));
    }
}
const std::string& Value::as_str() const {
    if (kind != Kind::Str) fail("value is not a string");
    return s;
}
const std::vector<Value>& Value::as_arr() const {
    if (kind != Kind::Arr) fail("value is not an array");
    return arr;
}

// ---------------------------------------------------------------- dtypes
const char* dtype_name(DType t) {
    switch (t) {
        case DType::F32:  return "F32";
        case DType::F16:  return "F16";
        case DType::Q4_0: return "Q4_0";
        case DType::Q4_1: return "Q4_1";
        case DType::Q5_0: return "Q5_0";
        case DType::Q5_1: return "Q5_1";
        case DType::Q8_0: return "Q8_0";
        case DType::Q8_1: return "Q8_1";
        case DType::Q2_K: return "Q2_K";
        case DType::Q3_K: return "Q3_K";
        case DType::Q4_K: return "Q4_K";
        case DType::Q5_K: return "Q5_K";
        case DType::Q6_K: return "Q6_K";
    }
    return "?";
}

int64_t dtype_block_elems(DType t) {
    switch (t) {
        case DType::F32:
        case DType::F16: return 1;
        case DType::Q2_K:
        case DType::Q3_K:
        case DType::Q4_K:
        case DType::Q5_K:
        case DType::Q6_K: return 256;
        default: return 32;  // Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1
    }
}

int64_t dtype_block_bytes(DType t) {
    switch (t) {
        case DType::F32:  return 4;
        case DType::F16:  return 2;
        case DType::Q4_0: return 2 + 16;   // d(fp16) + 16 packed nibbles
        case DType::Q4_1: return 2 + 2 + 16;
        case DType::Q5_0: return 2 + 4 + 16;
        case DType::Q5_1: return 2 + 2 + 4 + 16;
        case DType::Q8_0: return 2 + 32;   // d(fp16) + 32 int8
        case DType::Q8_1: return 2 + 2 + 32;
        case DType::Q2_K: return 256 / 16 + 256 / 4 + 2 + 2;   // 84
        case DType::Q3_K: return 256 / 8 + 256 / 4 + 12 + 2;   // 110
        case DType::Q4_K: return 2 + 2 + 12 + 256 / 2;         // 144
        case DType::Q5_K: return 2 + 2 + 12 + 256 / 8 + 256 / 2;  // 176
        case DType::Q6_K: return 256 / 2 + 256 / 4 + 256 / 16 + 2;  // 210
    }
    return 0;
}

static inline float f16_to_f32(uint16_t h) {
    // Ported from the branchless half->float trick; exact for normals, subnormals,
    // inf and NaN.
    uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            // subnormal: normalize
            int e = -1;
            uint32_t m = mant;
            do { ++e; m <<= 1; } while ((m & 0x400u) == 0);
            bits = sign | static_cast<uint32_t>(127 - 15 - e) << 23 | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &bits, 4);
    return out;
}

void dequantize(DType t, const uint8_t* src, float* dst, int64_t n) {
    const int64_t qk = dtype_block_elems(t);
    const int64_t bytes = dtype_block_bytes(t);
    const int64_t nblocks = (n + qk - 1) / qk;
    switch (t) {
        case DType::F32:
            std::memcpy(dst, src, static_cast<size_t>(n) * 4);
            return;
        case DType::F16:
            for (int64_t i = 0; i < n; ++i) dst[i] = f16_to_f32(read_le<uint16_t>(src + i * 2));
            return;
        case DType::Q8_0: {
            for (int64_t b = 0; b < nblocks; ++b) {
                const uint8_t* blk = src + b * bytes;
                const float d = f16_to_f32(read_le<uint16_t>(blk));
                const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
                for (int64_t i = 0; i < qk; ++i) {
                    const int64_t idx = b * qk + i;
                    if (idx >= n) break;
                    dst[idx] = d * static_cast<float>(qs[i]);
                }
            }
            return;
        }
        case DType::Q4_0: {
            for (int64_t b = 0; b < nblocks; ++b) {
                const uint8_t* blk = src + b * bytes;
                const float d = f16_to_f32(read_le<uint16_t>(blk));
                const uint8_t* qs = blk + 2;
                // nibble i holds element i (low) and element i+16 (high)
                for (int64_t i = 0; i < qk / 2; ++i) {
                    const int v0 = static_cast<int>(qs[i] & 0x0F) - 8;
                    const int v1 = static_cast<int>(qs[i] >> 4) - 8;
                    const int64_t i0 = b * qk + i;
                    const int64_t i1 = i0 + qk / 2;
                    if (i0 < n) dst[i0] = d * static_cast<float>(v0);
                    if (i1 < n) dst[i1] = d * static_cast<float>(v1);
                }
            }
            return;
        }
        default:
            fail(std::string("dequantize: unsupported dtype ") + dtype_name(t));
    }
}

// ---------------------------------------------------------------- TensorInfo
int64_t TensorInfo::nelems() const {
    int64_t n = 1;
    for (auto d : dims) n *= static_cast<int64_t>(d);
    return n;
}

std::string TensorInfo::shape_str() const {
    std::string s = "[";
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i) s += ", ";
        s += std::to_string(dims[i]);
    }
    return s + "]";
}

// ---------------------------------------------------------------- MappedFile
#ifdef _WIN32
MappedFile::MappedFile(const std::string& path) : path_(path) {
    int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::wstring wpath(static_cast<size_t>(wlen), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wlen);

    HANDLE h = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, nullptr);
    if (h == INVALID_HANDLE_VALUE) fail("cannot open " + path);
    handle_ = h;

    LARGE_INTEGER sz;
    if (!GetFileSizeEx(h, &sz)) { CloseHandle(h); handle_ = nullptr; fail("cannot stat " + path); }
    size_ = static_cast<size_t>(sz.QuadPart);
    if (size_ == 0) fail("empty file " + path);

    HANDLE m = CreateFileMappingW(h, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!m) { CloseHandle(h); handle_ = nullptr; fail("CreateFileMapping failed for " + path); }
    void* p = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(m);
    if (!p) { CloseHandle(h); handle_ = nullptr; fail("MapViewOfFile failed for " + path); }
    map_ = p;
    data_ = static_cast<const uint8_t*>(p);
}

MappedFile::~MappedFile() {
    if (map_) UnmapViewOfFile(map_);
    if (handle_) CloseHandle(static_cast<HANDLE>(handle_));
}
#else
MappedFile::MappedFile(const std::string& path) : path_(path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) fail("cannot open " + path);
    struct stat st {};
    if (fstat(fd, &st) != 0) { ::close(fd); fail("cannot stat " + path); }
    size_ = static_cast<size_t>(st.st_size);
    if (size_ == 0) { ::close(fd); fail("empty file " + path); }
    void* p = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (p == MAP_FAILED) fail("mmap failed for " + path);
    map_ = p;
    data_ = static_cast<const uint8_t*>(p);
}

MappedFile::~MappedFile() {
    if (map_) munmap(map_, size_);
}
#endif

// ---------------------------------------------------------------- GGUFFile
GGUFFile::GGUFFile(const std::string& path) : path_(path), file_(path) { parse(); }

bool GGUFFile::has(const std::string& key) const { return kv_index_.count(key) != 0; }

const Value& GGUFFile::get(const std::string& key) const {
    auto it = kv_index_.find(key);
    if (it == kv_index_.end()) fail("missing metadata key '" + key + "'");
    return kv_[it->second].second;
}

uint32_t GGUFFile::get_u32(const std::string& key, uint32_t fallback) const {
    return has(key) ? static_cast<uint32_t>(get(key).as_u64()) : fallback;
}
int32_t GGUFFile::get_i32(const std::string& key, int32_t fallback) const {
    return has(key) ? static_cast<int32_t>(get(key).as_i64()) : fallback;
}
uint64_t GGUFFile::get_u64(const std::string& key, uint64_t fallback) const {
    return has(key) ? get(key).as_u64() : fallback;
}
float GGUFFile::get_f32(const std::string& key, float fallback) const {
    return has(key) ? static_cast<float>(get(key).as_f64()) : fallback;
}
std::string GGUFFile::get_str(const std::string& key, const std::string& fallback) const {
    return has(key) ? get(key).as_str() : fallback;
}

std::string GGUFFile::get_any(const std::string& key, const std::string& fallback) const {
    if (!has(key)) return fallback;
    const Value& v = get(key);
    char buf[64];
    switch (v.kind) {
        case Value::Kind::Str: return v.s;
        case Value::Kind::Arr: {
            std::snprintf(buf, sizeof buf, "<array of %zu>", v.arr.size());
            return buf;
        }
        case Value::Kind::F32:
        case Value::Kind::F64:
            std::snprintf(buf, sizeof buf, "%g", v.f);
            return buf;
        case Value::Kind::Bool: return v.u ? "true" : "false";
        case Value::Kind::I8:
        case Value::Kind::I16:
        case Value::Kind::I32:
        case Value::Kind::I64:
            std::snprintf(buf, sizeof buf, "%lld", static_cast<long long>(v.as_i64()));
            return buf;
        default:
            std::snprintf(buf, sizeof buf, "%llu", static_cast<unsigned long long>(v.u));
            return buf;
    }
}

std::vector<float> GGUFFile::get_f32_array(const std::string& key) const {
    std::vector<float> out;
    if (!has(key)) return out;
    const Value& v = get(key);
    if (v.kind != Value::Kind::Arr) return out;
    out.reserve(v.arr.size());
    for (const auto& e : v.arr) out.push_back(static_cast<float>(e.as_f64()));
    return out;
}

std::vector<uint32_t> GGUFFile::get_u32_array(const std::string& key) const {
    std::vector<uint32_t> out;
    if (!has(key)) return out;
    const Value& v = get(key);
    if (v.kind != Value::Kind::Arr) return out;
    out.reserve(v.arr.size());
    for (const auto& e : v.arr) out.push_back(static_cast<uint32_t>(e.as_u64()));
    return out;
}

std::vector<std::string> GGUFFile::get_str_array(const std::string& key) const {
    std::vector<std::string> out;
    if (!has(key)) return out;
    const Value& v = get(key);
    if (v.kind != Value::Kind::Arr) return out;
    out.reserve(v.arr.size());
    for (const auto& e : v.arr) out.push_back(e.as_str());
    return out;
}

bool GGUFFile::has_tensor(const std::string& name) const { return tensor_index_.count(name) != 0; }

const TensorInfo& GGUFFile::tensor(const std::string& name) const {
    auto it = tensor_index_.find(name);
    if (it == tensor_index_.end()) fail("missing tensor '" + name + "'");
    return tensors_[it->second];
}

void GGUFFile::parse() {
    const uint8_t* base = file_.data();
    Cursor c{base, base + file_.size()};

    uint32_t magic = c.num<uint32_t>();
    if (magic != 0x46554747u) {  // "GGUF" little-endian
        fail("bad magic, not a gguf file");
    }
    version_ = c.num<uint32_t>();
    if (version_ < 2 || version_ > 3) {
        fail("unsupported gguf version " + std::to_string(version_));
    }
    const uint64_t n_tensors = c.num<uint64_t>();
    const uint64_t n_kv = c.num<uint64_t>();
    if (n_kv > (1u << 22) || n_tensors > (1u << 22)) fail("implausible header counts");

    kv_.reserve(static_cast<size_t>(n_kv));
    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key = c.str();
        uint32_t kind = c.num<uint32_t>();
        Value v = read_value(c, kind, 0);
        if (key == "general.alignment") alignment_ = v.as_u64();
        kv_index_[key] = kv_.size();
        kv_.emplace_back(std::move(key), std::move(v));
    }
    if (alignment_ == 0) alignment_ = 32;

    struct RawInfo {
        std::string name;
        DType type;
        std::vector<uint64_t> dims;
        uint64_t offset;
    };
    std::vector<RawInfo> raw;
    raw.reserve(static_cast<size_t>(n_tensors));
    for (uint64_t i = 0; i < n_tensors; ++i) {
        RawInfo r;
        r.name = c.str();
        uint32_t n_dims = c.num<uint32_t>();
        if (n_dims > 8) fail("tensor '" + r.name + "' has too many dims");
        r.dims.reserve(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d) r.dims.push_back(c.num<uint64_t>());
        r.type = static_cast<DType>(c.num<uint32_t>());
        r.offset = c.num<uint64_t>();
        raw.push_back(std::move(r));
    }

    // tensor payloads start at the next `alignment` boundary
    uint64_t pos = static_cast<uint64_t>(c.p - base);
    uint64_t aligned = (pos + alignment_ - 1) / alignment_ * alignment_;
    if (aligned > file_.size()) fail("tensor data offset past end of file");
    tensor_data_offset_ = aligned;

    tensors_.reserve(raw.size());
    for (auto& r : raw) {
        TensorInfo ti;
        ti.name = std::move(r.name);
        ti.type = r.type;
        ti.dims = std::move(r.dims);
        ti.offset = r.offset;
        int64_t n = ti.nelems();
        const int64_t qk = dtype_block_elems(ti.type);
        ti.nbytes = static_cast<uint64_t>((n + qk - 1) / qk) * static_cast<uint64_t>(dtype_block_bytes(ti.type));
        if (tensor_data_offset_ + ti.offset + ti.nbytes > file_.size()) {
            fail("tensor '" + ti.name + "' payload runs past end of file");
        }
        ti.data = base + tensor_data_offset_ + ti.offset;
        tensor_index_[ti.name] = tensors_.size();
        tensors_.push_back(std::move(ti));
    }
}

}  // namespace zen
