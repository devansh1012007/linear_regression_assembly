# asm-ols — top-level Makefile
# Targets: all | harness | test | verify | lint | bench | bench-report | repro | clean
# Variants: BUILD=debug|release|harness  (asm-project-delivery §5)
# Layout: build/lib (libols.so + libols.a, PRODUCTION objects only),
#         build/tests (test binaries + the ABI probe object),
#         build/cli (command-line tools).
# The test-only objects (abi_probe, adapters) NEVER link into the production
# library (review finding: libols.so must not contain test functionality).
NASM      ?= nasm
NASMFLAGS  = -f elf64 -Wall
NASMFLAGS += $(if $(filter release,$(BUILD)),,-g -F dwarf)
CC        ?= gcc
CFLAGS     = -O2 -std=c11 -Wall -Wextra -Werror -Iinclude
PYTHON    ?= python3

CORE_SRCS  = src/core/cpu.asm src/core/dispatch.asm src/core/api.asm \
             src/core/sums.asm src/core/simple.asm \
             src/core/dot.asm src/core/axpy.asm \
             src/core/gram.asm src/core/fit.asm src/core/chol.asm \
             src/core/stats.asm src/core/qr.asm src/core/special.asm
CORE_OBJS  = $(addprefix build/lib/,$(notdir $(CORE_SRCS:.asm=.o)))

PROBE_SRC  = tests/abi/abi_probe.asm
ADAPTERS   = tests/abi/abi_adapters.c
TEST_BINS  = build/tests/test_abi build/tests/test_guard \
             build/tests/test_guard_full build/tests/test_guard_p2 \
             build/tests/test_upper_half

.PHONY: all harness test verify lint bench bench-report plots repro clean
.DEFAULT_GOAL := all

all: harness

# ---- production library: ONLY core objects -----------------------------------
build/lib:
	@mkdir -p build/lib build/tests build/cli

$(CORE_OBJS): build/lib/%.o: src/core/%.asm src/abi.inc src/macros.inc include/ols.h | build/lib
	$(NASM) $(NASMFLAGS) -I. -o $@ $<

build/lib/ols_mt.o: src/thread/ols_mt.c | build/lib
	$(CC) $(CFLAGS) -fPIC -pthread -Iinclude -c -o $@ $<

build/lib/libols.so: $(CORE_OBJS) build/lib/ols_mt.o
	$(CC) -O2 -std=c11 -fPIC -shared -o $@ $(CORE_OBJS) build/lib/ols_mt.o -lpthread

build/lib/libols.a: $(CORE_OBJS) build/lib/ols_mt.o
	ar rcs $@ $(CORE_OBJS) build/lib/ols_mt.o

# ---- tests (own objects; link the production lib) ----------------------------
build/tests/abi_probe.o: $(PROBE_SRC) src/abi.inc | build/lib
	$(NASM) $(NASMFLAGS) -I. -o $@ $(PROBE_SRC)

build/tests/test_abi: tests/abi/test_abi.c build/lib/libols.so build/tests/abi_probe.o $(ADAPTERS)
	$(CC) $(CFLAGS) -Iinclude -o $@ tests/abi/test_abi.c $(ADAPTERS) \
	    build/tests/abi_probe.o -Lbuild/lib -lols -Wl,-rpath,$(CURDIR)/build/lib

build/tests/test_guard: tests/guard/test_guard.c build/lib/libols.so build/tests/abi_probe.o | build/lib
	$(CC) $(CFLAGS) -Iinclude -Itests/guard -o $@ $< build/tests/abi_probe.o -Lbuild/lib -lols -Wl,-rpath,$(CURDIR)/build/lib

build/tests/test_guard_full: tests/guard/test_guard_full.c build/lib/libols.so | build/lib
	$(CC) $(CFLAGS) -Iinclude -Itests/guard -o $@ $< build/tests/abi_probe.o -Lbuild/lib -lols -Wl,-rpath,$(CURDIR)/build/lib

build/tests/test_guard_p2: tests/guard/test_guard_p2.c build/lib/libols.so | build/lib
	$(CC) $(CFLAGS) -Iinclude -Itests/guard -o $@ $< build/tests/abi_probe.o -Lbuild/lib -lols -Wl,-rpath,$(CURDIR)/build/lib

# runtime upper-half discipline: the SSE sink must be legacy-encoded
# (-mno-avx) so a dirty upper half from the library shows as a transition
# penalty in the cold-vs-warm ratio (tests/guard/test_upper_half.c)
build/tests/test_upper_half: tests/guard/test_upper_half.c build/lib/libols.so | build/lib
	$(CC) $(CFLAGS) -mno-avx -Iinclude -Itests/guard -o $@ $< -Lbuild/lib -lols -Wl,-rpath,$(CURDIR)/build/lib

build/cli/asm-ols: cli/asm_ols.c build/lib/libols.so | build/lib
	$(CC) $(CFLAGS) -Iinclude -o $@ cli/asm_ols.c -Lbuild/lib -lols -lm -Wl,-rpath,$(CURDIR)/build/lib

# freestanding CLI (G4 clause 4, MoSCoW MUST "libc-free"): raw syscalls,
# no libc, no dynamic section, NX stack. Links the PRODUCTION objects.
build/cli/asm-ols-fs: cli/freestanding.c $(CORE_OBJS) | build/cli
	$(CC) -O2 -std=c11 -Wall -Wextra -Werror -nostdlib -static -fno-stack-protector -fno-builtin \
	  -Wl,--build-id=none -Wl,-z,noexecstack -Wl,-e,_start \
	  -o $@ cli/freestanding.c $(CORE_OBJS) -lgcc
	@readelf -d $@ | grep -q 'There is no dynamic section' \
	  || (echo 'FAIL: dynamic section present in freestanding build' && exit 1)
	@readelf -W -l $@ | grep GNU_STACK | grep -qv 'RWE' \
	  || (echo 'FAIL: executable stack in freestanding build' && exit 1)
	@echo 'freestanding OK: no dynamic section, NX stack ('$$(stat -c%s $@)' bytes)'

harness: build/lib/libols.so build/lib/libols.a build/tests/libolstest.so build/cli/asm-ols build/cli/asm-ols-fs $(TEST_BINS)

test: harness
	./build/tests/test_abi && ./build/tests/test_guard && ./build/tests/test_guard_full && ./build/tests/test_guard_p2 && ./build/tests/test_upper_half
	$(PYTHON) tests/py/run_all.py
	$(PYTHON) tests/py/test_gram.py && $(PYTHON) tests/py/test_fit.py && \
	$(PYTHON) tests/py/test_qr.py && $(PYTHON) tests/py/test_phase2.py && $(PYTHON) tests/py/test_mt.py && $(PYTHON) tests/py/test_special.py

# make verify == CI (asm-verification §7)
verify: lint test
	$(PYTHON) tools/lint_contracts.py --check-src

lint:
	$(PYTHON) tools/lint_contracts.py

bench: harness
	$(PYTHON) bench/machine/characterize.py

bench-report:
	$(PYTHON) tools/update_perflog.py

# DoD-3 evidence plots from committed raw results (no re-measurement)
plots:
	$(PYTHON) tools/make_plots.py

repro: verify bench bench-report plots
	@echo "repro: results under results/ regenerated from source"

clean:
	rm -rf build

# test-support shared lib: production core + ABI probe + adapters. Loaded by
# tests/py for abi_probe_call; the PRODUCTION libols.so stays test-free.
build/tests/libolstest.so: $(CORE_OBJS) build/lib/ols_mt.o build/tests/abi_probe.o $(ADAPTERS)
	$(CC) -O2 -std=c11 -fPIC -shared -o $@ $(CORE_OBJS) build/lib/ols_mt.o build/tests/abi_probe.o $(ADAPTERS) -Iinclude -lpthread
