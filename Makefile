PYTHON ?= python3
CXX ?= clang++

PY_INCLUDES := $(shell $(PYTHON)-config --includes)
PY_EXT_SUFFIX := $(shell $(PYTHON)-config --extension-suffix)

NB_DIR := $(shell $(PYTHON) -m nanobind --cmake_dir)
NB_ROOT := $(abspath $(NB_DIR)/..)
NB_INCLUDE := $(NB_ROOT)/include
NB_EXT_INCLUDE := $(NB_ROOT)/ext/robin_map/include
NB_SRC := $(NB_ROOT)/src/nb_combined.cpp

COMMON_FLAGS := -std=c++20 -x c++ -g -O0 -fPIC
USER_WARNINGS := -Wall -Wextra -Wpedantic -Wconversion
THIRD_PARTY_WARNINGS := -Wall -Wextra -Wpedantic -Wno-sign-conversion -Wno-conversion

USER_INCLUDES := $(PY_INCLUDES) -isystem $(NB_INCLUDE) -isystem $(NB_EXT_INCLUDE)
NB_INCLUDES := $(PY_INCLUDES) -isystem $(NB_INCLUDE) -isystem $(NB_EXT_INCLUDE)

USER_CXXFLAGS := $(COMMON_FLAGS) $(USER_WARNINGS) $(USER_INCLUDES)
NB_CXXFLAGS := $(COMMON_FLAGS) $(THIRD_PARTY_WARNINGS) $(NB_INCLUDES)

LDFLAGS := -shared -undefined dynamic_lookup

SRCS := core/tensor.cpp core/bindings.cpp
OBJS := $(patsubst core/%.cpp,build/%.o,$(SRCS))
NB_OBJ := build/nb_combined.o
OUT := studious/_core$(PY_EXT_SUFFIX)

.PHONY: build clean debug check build_dir

build: check compile_flags.txt $(OUT)

build_dir:
	mkdir -p build

compile_flags.txt: Makefile
	@printf "%s\n" $(USER_CXXFLAGS) > $@
	@echo "wrote $@"

$(OBJS): build/%.o: core/%.cpp compile_flags.txt | build_dir
	$(CXX) $(USER_CXXFLAGS) -Icore -c $< -o $@

$(NB_OBJ): $(NB_SRC) compile_flags.txt | build_dir
	$(CXX) $(NB_CXXFLAGS) -c $(NB_SRC) -o $(NB_OBJ)

$(OUT): $(OBJS) $(NB_OBJ) studious/__init__.py
	$(CXX) $(OBJS) $(NB_OBJ) -o $(OUT) $(LDFLAGS)

check:
	@for src in $(SRCS); do test -f "$$src" || (echo "missing source: $$src" && exit 1); done
	@test -f "core/tensor.h" || (echo "missing header: core/tensor.h" && exit 1)
	@test -f "$(NB_SRC)" || (echo "missing nanobind source: $(NB_SRC)" && exit 1)
	@test -d "$(NB_INCLUDE)" || (echo "missing nanobind include dir: $(NB_INCLUDE)" && exit 1)
	@test -d "$(NB_EXT_INCLUDE)" || (echo "missing robin_map include dir: $(NB_EXT_INCLUDE)" && exit 1)
	@test -f "studious/__init__.py" || (echo "missing studious/__init__.py" && exit 1)
	@echo "ok"

debug:
	@echo "PYTHON = $(PYTHON)"
	@echo "CXX = $(CXX)"
	@echo "PY_INCLUDES = $(PY_INCLUDES)"
	@echo "PY_EXT_SUFFIX = $(PY_EXT_SUFFIX)"
	@echo "NB_DIR = $(NB_DIR)"
	@echo "NB_ROOT = $(NB_ROOT)"
	@echo "NB_INCLUDE = $(NB_INCLUDE)"
	@echo "NB_EXT_INCLUDE = $(NB_EXT_INCLUDE)"
	@echo "NB_SRC = $(NB_SRC)"
	@echo "SRCS = $(SRCS)"
	@echo "OBJS = $(OBJS)"
	@echo "NB_OBJ = $(NB_OBJ)"
	@echo "OUT = $(OUT)"
	@echo
	@echo "USER_CXXFLAGS:"
	@printf "  %s\n" $(USER_CXXFLAGS)
	@echo
	@echo "NB_CXXFLAGS:"
	@printf "  %