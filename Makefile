# asm-ols — top-level Makefile
# Targets: all | harness | test | verify | lint | bench | bench-report | repro | clean
# Variants: BUILD=debug|release|harness  (asm-project-delivery §5)
NASM      ?= nasm
NASMFLAGS  = -f elf64 -Wall
NASMFLAGS += $(if $(filter release,$(BUILD)),,-g -F dwarf)
CC        ?= gcc
CFLAGS     = -O2 -std=c11 -Wall -Wextra -Werror -Iinclude
PYTHON    ?= python3

CORE_SRCS  = src/core/cpu.asm src/core/dispatch.asm \
             src/core/sums.asm src/core/simple.asm \
             src/core/dot.asm src/core/axpy.asm
PROBE_SRC  = tests/abi/abi_probe.asm
ADAPTERS   = tests/abi/abi_adapters.c
TEST_BINS  = build/test_abi build/test_guard build/test_guard_full

.PHONY: all harness test verify lint bench bench-report repro clean
.DEFAULT_GOAL := all

all: harness

build/libols.so: $(CORE_SRCS) $(PROBE_SRC) $(ADAPTERS) src/abi.inc src/macros.inc include/ols.h
	@mkdir -p build
	@for f in $(CORE_SRCS) $(PROBE_SRC); do \
	  $(NASM) $(NASMFLAGS) -I. -o build/$$(basename $$f .asm).o $$f || exit 1; \
	done
	$(CC) -O2 -std=c11 -fPIC -shared -o $@ build/cpu.o build/dispatch.o \
	    build/dot.o build/axpy.o build/sums.o build/simple.o build/abi_probe.o \
	    $(ADAPTERS) -Iinclude

build/test_abi: tests/abi/test_abi.c build/libols.so
	@mkdir -p build
	$(CC) $(CFLAGS) -Iinclude -o $@ $< -Lbuild -lols -Wl,-rpath,$(CURDIR)/build

build/test_guard: tests/guard/test_guard.c build/libols.so
	@mkdir -p build
	$(CC) $(CFLAGS) -Iinclude -Itests/guard -o $@ $< -Lbuild -lols -Wl,-rpath,$(CURDIR)/build

build/test_guard_full: tests/guard/test_guard_full.c build/libols.so
	@mkdir -p build
	$(CC) $(CFLAGS) -Iinclude -Itests/guard -o $@ $< -Lbuild -lols -Wl,-rpath,$(CURDIR)/build

harness: build/libols.so $(TEST_BINS)

test: harness
	./build/test_abi && ./build/test_guard && ./build/test_guard_full
	$(PYTHON) tests/py/run_all.py

# make verify == CI (asm-verification §7)
verify: lint test
	$(PYTHON) tools/lint_contracts.py --check-src

lint:
	$(PYTHON) tools/lint_contracts.py

bench: harness
	$(PYTHON) bench/machine/characterize.py

bench-report:
	$(PYTHON) tools/update_perflog.py

repro: verify bench bench-report
	@echo "repro: results under results/ regenerated from source"

clean:
	rm -rf build
