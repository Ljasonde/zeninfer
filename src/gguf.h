// zeninfer - CPU LLM inference engine
// gguf.h : minimal, allocation-light GGUF v3 reader with mmap-backed weights.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace zen {

// ---------------------------------------------------------------- ggml tensor types
enum class DType : uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
};

const char* dtype_name(DType t);
// values per block (32 for the legacy formats, 256 for K-quants, 1 for F16/F32)
int64_t dtype_block_elems(DType t);
// bytes per block
int64_t dtype_block_bytes(DType t);
// dequantize `n` elements starting at `src` into `dst` (f32). n must be a multiple
// of the block size; it is rounded up internally and the tail is clipped.
void dequantize(DType t, const uint8_t* src, float* dst, int64_t n);

// ---------------------------------------------------------------- metadata values
struct Value {
    enum class Kind : uint32_t {
        U8 = 0, I8 = 1, U16 = 2, I16 = 3, U32 = 4, I32 = 5, F32 = 6, Bool = 7,
        Str = 8, Arr = 9, U64 = 10, I64 = 11, F64 = 12
    };
    Kind kind = Kind::U32;
    uint64_t u = 0;   // integer-ish payload (sign-extended for signed kinds)
    double f = 0.0;   // F32 / F64
    std::string s;    // Str
    std::vector<Value> arr;

    bool is_num() const { return kind != Kind::Str && kind != Kind::Arr; }
    uint64_t as_u64() const;
    int64_t as_i64() const;
    double as_f64() const;
    const std::string& as_str() const;
    const std::vector<Value>& as_arr() const;
};

// ---------------------------------------------------------------- tensor directory
struct TensorInfo {
    std::string name;
    DType type = DType::F32;
    // dims in ggml order: dims[0] is the fastest-varying axis.
    std::vector<uint64_t> dims;
    uint64_t offset = 0;   // relative to the tensor-data section
    uint64_t nbytes = 0;   // byte size of the payload
    const uint8_t* data = nullptr;

    int64_t nelems() const;
    std::string shape_str() const;
};

// ---------------------------------------------------------------- reader
class MappedFile {
  public:
    explicit MappedFile(const std::string& path);
    ~MappedFile();
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    const uint8_t* data() const { return data_; }
    size_t size() const { return size_; }

  private:
    void* map_ = nullptr;
    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
#ifdef _WIN32
    void* handle_ = nullptr;
#endif
    std::string path_;
};

class GGUFFile {
  public:
    explicit GGUFFile(const std::string& path);

    uint32_t version() const { return version_; }
    const std::string& path() const { return path_; }
    const MappedFile& file() const { return file_; }
    uint64_t alignment() const { return alignment_; }
    uint64_t tensor_data_offset() const { return tensor_data_offset_; }

    bool has(const std::string& key) const;
    const Value& get(const std::string& key) const;

    uint32_t get_u32(const std::string& key, uint32_t fallback) const;
    int32_t get_i32(const std::string& key, int32_t fallback) const;
    uint64_t get_u64(const std::string& key, uint64_t fallback) const;
    float get_f32(const std::string& key, float fallback) const;
    std::string get_str(const std::string& key, const std::string& fallback) const;
    // Type-tolerant accessor: renders any scalar/string metadata value as text.
    // Used by the inspection CLI, where a key's type is whatever the converter
    // happened to write (general.file_type is a u32, not a string).
    std::string get_any(const std::string& key, const std::string& fallback) const;
    // array accessors: empty vector when the key is missing or not an array
    std::vector<float> get_f32_array(const std::string& key) const;
    std::vector<uint32_t> get_u32_array(const std::string& key) const;
    std::vector<std::string> get_str_array(const std::string& key) const;

    const std::vector<std::pair<std::string, Value>>& metadata() const { return kv_; }
    const std::vector<TensorInfo>& tensors() const { return tensors_; }
    const TensorInfo& tensor(const std::string& name) const;
    bool has_tensor(const std::string& name) const;

  private:
    void parse();

    std::string path_;
    MappedFile file_;
    uint32_t version_ = 0;
    uint64_t alignment_ = 32;
    uint64_t tensor_data_offset_ = 0;
    std::vector<std::pair<std::string, Value>> kv_;
    std::unordered_map<std::string, size_t> kv_index_;
    std::vector<TensorInfo> tensors_;
    std::unordered_map<std::string, size_t> tensor_index_;
};

// Little-endian scalar read from an arbitrary offset (no alignment assumptions).
template <typename T>
inline T read_le(const uint8_t* p) {
    T v;
    std::memcpy(&v, p, sizeof(T));
    return v;
}

}  // namespace zen
