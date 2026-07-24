# Arch-aware baseline harness for the ISPC reference benchmarks.
# One binary pair per kernel:
#   bench_<stem>_a : ISPC narrow  + serial compiled -O3 (auto-vectorized)
#   bench_<stem>_b : ISPC wide    + serial compiled -fno-vectorize (true scalar)
# Targets: build-all (default), refs (dump ref-out/*.bin), clean.

UNAME_M := $(shell uname -m)

ISPC       ?= ispc
CXX        := $(shell command -v clang++ 2>/dev/null || command -v clang++-22 2>/dev/null)

ifeq ($(filter arm64 aarch64,$(UNAME_M)),)
  # x86_64
  ISPCFLAGS ?= -O3 --pic
  T_NARROW  := avx2-i32x8
  T_WIDE    := avx512skx-i32x16
else
  ISPCFLAGS ?= -O3 --arch=aarch64 --pic
  T_NARROW  := neon-i32x4
  T_WIDE    := neon-i32x8
endif

CXXFLAGS       ?= -O3 -std=c++17
CXXFLAGS_NOVEC := $(CXXFLAGS) -fno-vectorize -fno-slp-vectorize
BENCHDIR       := $(CURDIR)/../ispc-bench
REFOUT         := ref-out

STEMS := mandel options stencil volume ao rt
BINS  := $(foreach s,$(STEMS),bench_$(s)_a bench_$(s)_b)

.PHONY: build-all refs clean
build-all: $(BINS)

$(REFOUT):
	mkdir -p $(REFOUT)

# ---- mandelbrot (serial inline in driver) ----
mandelbrot_narrow.o mandelbrot_ispc.h: mandelbrot.ispc
	$(ISPC) $(ISPCFLAGS) --target=$(T_NARROW) -o mandelbrot_narrow.o -h mandelbrot_ispc.h mandelbrot.ispc
mandelbrot_wide.o: mandelbrot.ispc
	$(ISPC) $(ISPCFLAGS) --target=$(T_WIDE) -o mandelbrot_wide.o mandelbrot.ispc
bench_mandel_a: bench_mandel.cpp mandelbrot_narrow.o mandelbrot_ispc.h
	$(CXX) $(CXXFLAGS) bench_mandel.cpp mandelbrot_narrow.o -o $@
bench_mandel_b: bench_mandel.cpp mandelbrot_wide.o mandelbrot_ispc.h
	$(CXX) $(CXXFLAGS_NOVEC) bench_mandel.cpp mandelbrot_wide.o -o $@

# ---- options (serial inline; tasksys_shim resolves task symbols) ----
options_narrow.o options_ispc.h: options.ispc options_defs.h
	$(ISPC) $(ISPCFLAGS) --target=$(T_NARROW) -o options_narrow.o -h options_ispc.h options.ispc
options_wide.o: options.ispc options_defs.h
	$(ISPC) $(ISPCFLAGS) --target=$(T_WIDE) -o options_wide.o options.ispc
bench_options_a: bench_options.cpp tasksys_shim.cpp options_narrow.o options_ispc.h
	$(CXX) $(CXXFLAGS) bench_options.cpp tasksys_shim.cpp options_narrow.o -o $@
bench_options_b: bench_options.cpp tasksys_shim.cpp options_wide.o options_ispc.h
	$(CXX) $(CXXFLAGS_NOVEC) bench_options.cpp tasksys_shim.cpp options_wide.o -o $@

# ---- stencil (serial inline in driver) ----
stencil_narrow.o stencil_ispc.h: stencil.ispc
	$(ISPC) $(ISPCFLAGS) --target=$(T_NARROW) -o stencil_narrow.o -h stencil_ispc.h stencil.ispc
stencil_wide.o: stencil.ispc
	$(ISPC) $(ISPCFLAGS) --target=$(T_WIDE) -o stencil_wide.o stencil.ispc
bench_stencil_a: bench_stencil.cpp stencil_narrow.o stencil_ispc.h
	$(CXX) $(CXXFLAGS) bench_stencil.cpp stencil_narrow.o -o $@
bench_stencil_b: bench_stencil.cpp stencil_wide.o stencil_ispc.h
	$(CXX) $(CXXFLAGS_NOVEC) bench_stencil.cpp stencil_wide.o -o $@

# ---- volume (serial in a separate TU: compile it twice) ----
volume_narrow.o volume_ispc.h: volume.ispc
	$(ISPC) $(ISPCFLAGS) --target=$(T_NARROW) -o volume_narrow.o -h volume_ispc.h volume.ispc
volume_wide.o: volume.ispc
	$(ISPC) $(ISPCFLAGS) --target=$(T_WIDE) -o volume_wide.o volume.ispc
volume_serial_autovec.o: volume_serial.cpp
	$(CXX) $(CXXFLAGS) -c volume_serial.cpp -o $@
volume_serial_novec.o: volume_serial.cpp
	$(CXX) $(CXXFLAGS_NOVEC) -c volume_serial.cpp -o $@
bench_volume_a: bench_volume.cpp volume_serial_autovec.o volume_narrow.o volume_ispc.h
	$(CXX) $(CXXFLAGS) bench_volume.cpp volume_serial_autovec.o volume_narrow.o -o $@
bench_volume_b: bench_volume.cpp volume_serial_novec.o volume_wide.o volume_ispc.h
	$(CXX) $(CXXFLAGS) bench_volume.cpp volume_serial_novec.o volume_wide.o -o $@

# ---- ao (serial in ispc-bench/ao_serial.cpp: compile it twice) ----
ao_narrow.o ao_ispc.h: ao.ispc
	$(ISPC) $(ISPCFLAGS) --target=$(T_NARROW) -o ao_narrow.o -h ao_ispc.h ao.ispc
ao_wide.o: ao.ispc
	$(ISPC) $(ISPCFLAGS) --target=$(T_WIDE) -o ao_wide.o ao.ispc
ao_serial_autovec.o: $(BENCHDIR)/ao_serial.cpp
	$(CXX) $(CXXFLAGS) -c $(BENCHDIR)/ao_serial.cpp -o $@
ao_serial_novec.o: $(BENCHDIR)/ao_serial.cpp
	$(CXX) $(CXXFLAGS_NOVEC) -c $(BENCHDIR)/ao_serial.cpp -o $@
bench_ao_a: bench_ao.cpp ao_narrow.o ao_serial_autovec.o ao_ispc.h
	$(CXX) $(CXXFLAGS) bench_ao.cpp ao_narrow.o ao_serial_autovec.o -o $@
bench_ao_b: bench_ao.cpp ao_wide.o ao_serial_novec.o ao_ispc.h
	$(CXX) $(CXXFLAGS) bench_ao.cpp ao_wide.o ao_serial_novec.o -o $@

# ---- rt (serial inline in driver; tasksys_shim resolves unused task symbols) ----
rt_narrow.o rt_ispc.h: rt.ispc
	$(ISPC) $(ISPCFLAGS) --target=$(T_NARROW) -o rt_narrow.o -h rt_ispc.h rt.ispc
rt_wide.o: rt.ispc
	$(ISPC) $(ISPCFLAGS) --target=$(T_WIDE) -o rt_wide.o rt.ispc
bench_rt_a: bench_rt.cpp tasksys_shim.cpp rt_narrow.o rt_ispc.h
	$(CXX) $(CXXFLAGS) bench_rt.cpp tasksys_shim.cpp rt_narrow.o -o $@
bench_rt_b: bench_rt.cpp tasksys_shim.cpp rt_wide.o rt_ispc.h
	$(CXX) $(CXXFLAGS_NOVEC) bench_rt.cpp tasksys_shim.cpp rt_wide.o -o $@

# ---- reference-output dump (the _a binaries write the canonical ISPC output) ----
refs: build-all | $(REFOUT)
	./bench_mandel_a   $(REFOUT)/mandelbrot.bin
	./bench_options_a  $(REFOUT)/options
	./bench_stencil_a  $(REFOUT)/stencil.bin
	./bench_volume_a   $(REFOUT)/volume.bin
	./bench_ao_a       $(REFOUT)/ao_ispc.bin
	./bench_rt_a       $(REFOUT)/rt.bin

clean:
	rm -f $(BINS) *.o *_ispc.h
