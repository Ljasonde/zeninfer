# zeninfer - CPU LLM inference engine
#
#   make                              build build/zeninfer (native arch)
#   make ARCH="-march=x86-64-v3"      portable AVX2/FMA build (used by CI)
#   make clean
#
# Requires only a C++17 compiler and make -- no cmake, no external libraries.

CXX      ?= g++
ARCH     ?= -march=native
OPT      ?= -O3
CXXFLAGS := -std=c++17 $(OPT) $(ARCH) -Wall -Wextra -Wno-unused-parameter -Isrc
LDFLAGS  :=
LDLIBS   := -lm

ifeq ($(OS),Windows_NT)
  EXE := .exe
else
  EXE :=
endif

SRC_DIR := src
BUILD   := build
BIN     := $(BUILD)/zeninfer$(EXE)

SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD)/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

.PHONY: all clean test check check-forward check-hf

PYTHON ?= python

all: $(BIN)

$(BIN): $(OBJS)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(OBJS) $(LDLIBS)

$(BUILD)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -MMD -MP -c -o $@ $<

clean:
	rm -rf $(BUILD)

test: $(BIN)
	./$(BIN) inspect models/qwen2.5-0.5b-instruct-q8_0.gguf

# Forward-pass parity: every logit of three prompts against the NumPy reference.
# Slow on purpose (the reference is float64 NumPy), so it is not part of `make check`.
check-forward: $(BIN)
	@$(PYTHON) -c "import numpy" 2>/dev/null || { echo "make check-forward needs numpy"; exit 1; }
	$(PYTHON) tools/check_forward.py --exe ./$(BIN) --model models/qwen2.5-0.5b-instruct-q8_0.gguf

# Three-way parity: zeninfer vs NumPy (float64) vs HuggingFace transformers.
# The HF checkpoint is built once from the gguf (~2.4 GiB of f32 weights).
HF_DIR := $(BUILD)/hf_f32
check-hf: $(BIN)
	@$(PYTHON) -c "import numpy, torch, transformers, safetensors" 2>/dev/null || \
		{ echo "make check-hf needs numpy + torch (CPU) + transformers + safetensors"; \
		  echo "  pip install transformers safetensors -i https://pypi.tuna.tsinghua.edu.cn/simple"; \
		  echo "  pip install torch --index-url https://download.pytorch.org/whl/cpu"; exit 1; }
	@test -f $(HF_DIR)/model.safetensors || \
		$(PYTHON) tools/gguf_to_hf.py models/qwen2.5-0.5b-instruct-q8_0.gguf $(HF_DIR)
	$(PYTHON) tools/check_forward.py --exe ./$(BIN) --model models/qwen2.5-0.5b-instruct-q8_0.gguf --hf $(HF_DIR)

# Cross-checks the two halves of the front end against independent
# implementations. Requires: pip install tokenizers
check: $(BIN)
	@$(PYTHON) -c "import tokenizers" 2>/dev/null || \
		{ echo "make check needs the HuggingFace tokenizers package:"; \
		  echo "  pip install tokenizers -i https://pypi.tuna.tsinghua.edu.cn/simple"; \
		  echo "  make check PYTHON=/path/to/that/python"; exit 1; }
	@mkdir -p $(BUILD)
	./$(BIN) vocab-dump models/qwen2.5-0.5b-instruct-q8_0.gguf $(BUILD)/q8_0.vocab
	$(PYTHON) tools/dump_gguf.py models/qwen2.5-0.5b-instruct-q8_0.gguf --json > $(BUILD)/py.json
	./$(BIN) inspect models/qwen2.5-0.5b-instruct-q8_0.gguf --json > $(BUILD)/cpp.json
	$(PYTHON) tools/dump_gguf.py --diff $(BUILD)/py.json $(BUILD)/cpp.json
	$(PYTHON) tools/check_vocab.py --vocab $(BUILD)/q8_0.vocab --tokenizer models/tokenizer.json
	$(PYTHON) tools/check_tokenizer.py --tokenizer models/tokenizer.json --exe ./$(BIN) \
		--gguf models/qwen2.5-0.5b-instruct-q8_0.gguf --fuzz 2000

-include $(DEPS)
