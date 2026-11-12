"""Independent (pure Python, no deps) GGUF v3 reader used to cross-check the C++ parser.

    python tools/dump_gguf.py models/xxx.gguf --json   > out/python.json
    build/zeninfer.exe inspect models/xxx.gguf --json  > out/cpp.json
    python tools/dump_gguf.py --diff out/python.json out/cpp.json

The two implementations share no code, so agreement is real evidence.
"""
import argparse
import json
import struct
import sys

GGUF_MAGIC = 0x46554747  # "GGUF"

SCALARS = {
    0: ("<B", 1),   # uint8
    1: ("<b", 1),   # int8
    2: ("<H", 2),   # uint16
    3: ("<h", 2),   # int16
    4: ("<I", 4),   # uint32
    5: ("<i", 4),   # int32
    6: ("<f", 4),   # float32
    7: ("<B", 1),   # bool
    10: ("<Q", 8),  # uint64
    11: ("<q", 8),  # int64
    12: ("<d", 8),  # float64
}
T_STRING = 8
T_ARRAY = 9

# ggml type id -> (name, elems per block, bytes per block)
DTYPES = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
}


class Reader:
    def __init__(self, buf):
        self.buf = buf
        self.pos = 0

    def take(self, n):
        end = self.pos + n
        if end > len(self.buf):
            raise ValueError(f"unexpected EOF at {self.pos} (+{n})")
        chunk = self.buf[self.pos:end]
        self.pos = end
        return chunk

    def scalar(self, kind):
        fmt, size = SCALARS[kind]
        return struct.unpack(fmt, self.take(size))[0]

    def string(self):
        n = struct.unpack("<Q", self.take(8))[0]
        return self.take(n).decode("utf-8", errors="replace")

    def value(self, kind, depth=0):
        if depth > 8:
            raise ValueError("nesting too deep")
        if kind == T_STRING:
            return self.string()
        if kind == T_ARRAY:
            ekind = struct.unpack("<I", self.take(4))[0]
            n = struct.unpack("<Q", self.take(8))[0]
            return [self.value(ekind, depth + 1) for _ in range(n)]
        if kind in SCALARS:
            return self.scalar(kind)
        raise ValueError(f"unknown value type {kind}")


def parse(path):
    with open(path, "rb") as f:
        buf = f.read()
    r = Reader(buf)
    magic = struct.unpack("<I", r.take(4))[0]
    if magic != GGUF_MAGIC:
        raise ValueError(f"bad magic 0x{magic:08x}")
    version, n_tensors, n_kv = struct.unpack("<IQQ", r.take(20))
    kv = {}
    for _ in range(n_kv):
        key = r.string()
        kind = struct.unpack("<I", r.take(4))[0]
        kv[key] = r.value(kind)
    alignment = int(kv.get("general.alignment", 32)) or 32

    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        n_dims = struct.unpack("<I", r.take(4))[0]
        dims = list(struct.unpack("<" + "Q" * n_dims, r.take(8 * n_dims)))
        ttype = struct.unpack("<I", r.take(4))[0]
        offset = struct.unpack("<Q", r.take(8))[0]
        tensors.append({"name": name, "dims": dims, "type": ttype, "offset": offset})

    header_end = r.pos
    data_start = (header_end + alignment - 1) // alignment * alignment

    out_t = []
    total = 0
    for t in tensors:
        name, qk, blk_bytes = DTYPES.get(t["type"], (f"UNKNOWN{t['type']}", 1, 0))
        n = 1
        for d in t["dims"]:
            n *= d
        nbytes = (n + qk - 1) // qk * blk_bytes
        total += nbytes
        out_t.append({
            "name": t["name"],
            "dtype": name,
            "dims": t["dims"],
            "nelems": n,
            "offset": t["offset"],
            "nbytes": nbytes,
        })

    arch = kv.get("general.architecture", "")
    hp = {k: v for k, v in kv.items() if k.startswith(arch + ".")}
    tok = {k: (len(v) if isinstance(v, list) else v) for k, v in kv.items() if k.startswith("tokenizer.") and k != "tokenizer.ggml.tokens" and k != "tokenizer.ggml.merges"}

    # every payload must live inside the file, and the extents must not overlap
    ordered = sorted(out_t, key=lambda x: x["offset"])
    for i, t in enumerate(ordered):
        if data_start + t["offset"] + t["nbytes"] > len(buf):
            raise ValueError(f"tensor {t['name']} past EOF")
        if i + 1 < len(ordered):
            nxt = ordered[i + 1]
            if t["offset"] + t["nbytes"] > nxt["offset"]:
                raise ValueError(f"tensor {t['name']} overlaps {nxt['name']}")

    return {
        "path": path,
        "version": version,
        "file_size": len(buf),
        "alignment": alignment,
        "tensor_data_offset": data_start,
        "arch": arch,
        "name": kv.get("general.name", ""),
        "n_kv": n_kv,
        "n_tensors": n_tensors,
        "hyper": hp,
        "tokenizer": tok,
        "n_tokens": len(kv.get("tokenizer.ggml.tokens", [])),
        "n_merges": len(kv.get("tokenizer.ggml.merges", [])),
        "total_tensor_bytes": total,
        "tensors": out_t,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--diff", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()

    if args.diff:
        a = json.load(open(args.diff[0], encoding="utf-8"))
        b = json.load(open(args.diff[1], encoding="utf-8"))
        problems = []
        for key in ("version", "file_size", "alignment", "tensor_data_offset", "arch",
                    "n_tensors", "n_tokens", "n_merges", "total_tensor_bytes"):
            if a.get(key) != b.get(key):
                problems.append(f"{key}: {a.get(key)!r} != {b.get(key)!r}")
        ta = {t["name"]: (t["dtype"], t["dims"], t["nbytes"], t["offset"]) for t in a["tensors"]}
        tb = {t["name"]: (t["dtype"], t["dims"], t["nbytes"], t["offset"]) for t in b["tensors"]}
        if set(ta) != set(tb):
            problems.append(f"tensor names differ: only-A={sorted(set(ta)-set(tb))[:5]} only-B={sorted(set(tb)-set(ta))[:5]}")
        for name in sorted(set(ta) & set(tb)):
            if ta[name] != tb[name]:
                problems.append(f"{name}: {ta[name]} != {tb[name]}")
        if problems:
            print("MISMATCH")
            for p in problems[:40]:
                print("  " + p)
            return 1
        print(f"OK: {len(ta)} tensors identical, {a['total_tensor_bytes']} bytes, arch={a['arch']}")
        print(f"    tokens={a['n_tokens']} merges={a['n_merges']} alignment={a['alignment']}")
        return 0

    if not args.path:
        ap.error("path required")
    info = parse(args.path)
    if args.json:
        json.dump(info, sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
        return 0

    print(f"file             : {info['path']}")
    print(f"gguf version     : {info['version']}")
    print(f"arch             : {info['arch']}")
    print(f"file size        : {info['file_size']:,}")
    print(f"tensor data at   : {info['tensor_data_offset']:,} (alignment {info['alignment']})")
    print(f"tensors          : {info['n_tensors']}  total {info['total_tensor_bytes']:,} bytes")
    print(f"tokens/merges    : {info['n_tokens']} / {info['n_merges']}")
    print("-- hyper --")
    for k, v in sorted(info["hyper"].items()):
        print(f"  {k} = {v}")
    print("-- tensors --")
    for t in info["tensors"]:
        print(f"  {t['name']:<32} {t['dtype']:<6} {t['dims']} {t['nbytes']:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
