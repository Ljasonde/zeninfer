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

.PHONY: all clean test check

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
