import sys
sys.path.insert(0, r"D:\zeninfer\tools")
import dump_gguf
g = dump_gguf.parse(r"D:\zeninfer\models\qwen2.5-0.5b-instruct-q8_0.gguf")
print(type(g), list(g.keys())[:10] if isinstance(g, dict) else "")
print({k: (v if not isinstance(v, list) else f"list[{len(v)}]") for k, v in g.items() if k != "tensors"})
t = g["tensors"][0]
print(type(t), t if not isinstance(t, dict) else {k: v for k, v in t.items()})
