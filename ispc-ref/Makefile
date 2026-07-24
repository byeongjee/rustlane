# Unified baseline harness for the ISPC reference benchmarks (Apple Silicon,
# aarch64/NEON). Builds one binary per (benchmark, NEON gang-width) pair plus an
# inlined/linked serial reference, and runs each at whole-benchmark granularity
# (3 warmup + min-of-15 timed reps). Every _x4 run also dumps the canonical
# reference output into ref-out/ for the Rust side to compare against.
#
# Targets:
#   make build-all   compile all kernels + drivers (default)
#   make run-all     run every benchmark (x4 dumps references, x8 does not)
#   make clean       remove generated objects, headers, and binaries
# Per-benchmark: make {mandelbrot,options,stencil,volume,ao,rt}
#                make run-{mandelbrot,options,stencil,volume,ao,rt}

# CXX is pinned (not ?=) so a makefile assignment wins over an inherited CXX
# environment variable, keeping the reference-output bytes stable across runs.
ISPC       ?= ispc
CXX        := clang++
ISPCFLAGS  ?= -O3 --arch=aarch64 --pic
CXXFLAGS   ?= -O3 -std=c++17
T4         := neon-i32x4
T8         := neon-i32x8
BENCHDIR   := /Users/byeongjee/side/rust-ispc/ispc-bench
REFOUT     := ref-out

BINS := bench_mandel_x4 bench_mandel_x8 \
        bench_options_x4 bench_options_x8 \
        bench_stencil_x4 bench_stencil_x8 \
        bench_volume_x4 bench_volume_x8 \
        bench_ao_x4 bench_ao_x8 \
        bench_rt_x4 bench_rt_x8

OBJS := mandelbrot_i32x4.o mandelbrot_i32x8.o \
        options_i32x4.o options_i32x8.o \
        stencil_i32x4.o stencil_i32x8.o \
        volume_ispc_i32x4.o volume_ispc_i32x8.o \
        ao_i32x4.o ao_i32x8.o ao_serial.o \
        rt_i32x4.o rt_i32x8.o

HDRS := mandelbrot_ispc.h options_ispc.h stencil_ispc.h \
        volume_ispc.h ao_ispc.h rt_ispc.h rt_ispc_x8.h

.PHONY: build-all run-all clean \
        mandelbrot options stencil volume ao rt \
        run-mandelbrot run-options run-stencil run-volume run-ao run-rt

build-all: $(BINS)

run-all: run-mandelbrot run-options run-stencil run-volume run-ao run-rt

$(REFOUT):
	mkdir -p $(REFOUT)

# ---- mandelbrot (no tasks in kernel; serial inlined in driver) ----
mandelbrot: bench_mandel_x4 bench_mandel_x8
mandelbrot_i32x4.o mandelbrot_ispc.h: mandelbrot.ispc
	$(ISPC) $(ISPCFLAGS) --target=$(T4) -o mandelbrot_i32x4.o -h mandelbrot_ispc.h mandelbrot.ispc
mandelbrot_i32x8.o: mandelbrot.ispc
	$(ISPC) $(ISPCFLAGS) --target=$(T8) -o mandelbrot_i32x8.o -h mandelbrot_ispc.h mandelbrot.ispc
bench_mandel_x4: bench_mandel.cpp mandelbrot_i32x4.o
	$(CXX) $(CXXFLAGS) bench_mandel.cpp mandelbrot_i32x4.o -o $@
bench_mandel_x8: bench_mandel.cpp mandelbrot_i32x8.o
	$(CXX) $(CXXFLAGS) bench_mandel.cpp mandelbrot_i32x8.o -o $@
run-mandelbrot: bench_mandel_x4 bench_mandel_x8 | $(REFOUT)
	./bench_mandel_x4 $(REFOUT)/mandelbrot.bin
	./bench_mandel_x8

# ---- options (tasksys_shim.cpp resolves unused *_ispc_tasks symbols) ----
options: bench_options_x4 bench_options_x8
options_i32x4.o options_ispc.h: options.ispc options_defs.h
	$(ISPC) $(ISPCFLAGS) --target=$(T4) -o options_i32x4.o -h options_ispc.h options.ispc
options_i32x8.o: options.ispc options_defs.h
	$(ISPC) $(ISPCFLAGS) --target=$(T8) -o options_i32x8.o -h options_ispc.h options.ispc
bench_options_x4: bench_options.cpp tasksys_shim.cpp options_i32x4.o
	$(CXX) $(CXXFLAGS) bench_options.cpp tasksys_shim.cpp options_i32x4.o -o $@
bench_options_x8: bench_options.cpp tasksys_shim.cpp options_i32x8.o
	$(CXX) $(CXXFLAGS) bench_options.cpp tasksys_shim.cpp options_i32x8.o -o $@
run-options: bench_options_x4 bench_options_x8 | $(REFOUT)
	./bench_options_x4 $(REFOUT)/options
	./bench_options_x8

# ---- stencil (never-reached abort() task stubs in the driver) ----
stencil: bench_stencil_x4 bench_stencil_x8
stencil_i32x4.o stencil_ispc.h: stencil.ispc
	$(ISPC) $(ISPCFLAGS) --target=$(T4) -o stencil_i32x4.o -h stencil_ispc.h stencil.ispc
stencil_i32x8.o: stencil.ispc
	$(ISPC) $(ISPCFLAGS) --target=$(T8) -o stencil_i32x8.o -h stencil_ispc.h stencil.ispc
bench_stencil_x4: bench_stencil.cpp stencil_i32x4.o
	$(CXX) $(CXXFLAGS) bench_stencil.cpp stencil_i32x4.o -o $@
bench_stencil_x8: bench_stencil.cpp stencil_i32x8.o
	$(CXX) $(CXXFLAGS) bench_stencil.cpp stencil_i32x8.o -o $@
run-stencil: bench_stencil_x4 bench_stencil_x8 | $(REFOUT)
	./bench_stencil_x4 $(REFOUT)/stencil.bin
	./bench_stencil_x8

# ---- volume (serial in a separate TU; x8 object built without -h) ----
volume: bench_volume_x4 bench_volume_x8
volume_ispc_i32x4.o volume_ispc.h: volume.ispc
	$(ISPC) $(ISPCFLAGS) --target=$(T4) -o volume_ispc_i32x4.o -h volume_ispc.h volume.ispc
volume_ispc_i32x8.o: volume.ispc
	$(ISPC) $(ISPCFLAGS) --target=$(T8) -o volume_ispc_i32x8.o volume.ispc
bench_volume_x4: bench_volume.cpp volume_serial.cpp volume_ispc_i32x4.o volume_ispc.h
	$(CXX) $(CXXFLAGS) bench_volume.cpp volume_serial.cpp volume_ispc_i32x4.o -o $@
bench_volume_x8: bench_volume.cpp volume_serial.cpp volume_ispc_i32x8.o volume_ispc.h
	$(CXX) $(CXXFLAGS) bench_volume.cpp volume_serial.cpp volume_ispc_i32x8.o -o $@
run-volume: bench_volume_x4 bench_volume_x8 | $(REFOUT)
	./bench_volume_x4 $(REFOUT)/volume.bin
	./bench_volume_x8

# ---- ao (serial compiled from ispc-bench/ao_serial.cpp; RNG streams differ) ----
ao: bench_ao_x4 bench_ao_x8
ao_i32x4.o ao_ispc.h: ao.ispc
	$(ISPC) $(ISPCFLAGS) --target=$(T4) -o ao_i32x4.o -h ao_ispc.h ao.ispc
ao_i32x8.o: ao.ispc
	$(ISPC) $(ISPCFLAGS) --target=$(T8) -o ao_i32x8.o -h ao_ispc.h ao.ispc
ao_serial.o: $(BENCHDIR)/ao_serial.cpp
	$(CXX) $(CXXFLAGS) -c $(BENCHDIR)/ao_serial.cpp -o ao_serial.o
bench_ao_x4: bench_ao.cpp ao_i32x4.o ao_serial.o
	$(CXX) $(CXXFLAGS) bench_ao.cpp ao_i32x4.o ao_serial.o -o $@
bench_ao_x8: bench_ao.cpp ao_i32x8.o ao_serial.o
	$(CXX) $(CXXFLAGS) bench_ao.cpp ao_i32x8.o ao_serial.o -o $@
run-ao: bench_ao_x4 bench_ao_x8 | $(REFOUT)
	./bench_ao_x4 $(REFOUT)/ao_ispc.bin
	./bench_ao_x8

# ---- rt (-Wl,-dead_strip drops the unused task entry + its tasksys refs) ----
rt: bench_rt_x4 bench_rt_x8
rt_i32x4.o rt_ispc.h: rt.ispc
	$(ISPC) $(ISPCFLAGS) --target=$(T4) -o rt_i32x4.o -h rt_ispc.h rt.ispc
rt_i32x8.o rt_ispc_x8.h: rt.ispc
	$(ISPC) $(ISPCFLAGS) --target=$(T8) -o rt_i32x8.o -h rt_ispc_x8.h rt.ispc
bench_rt_x4: bench_rt.cpp rt_i32x4.o rt_ispc.h
	$(CXX) $(CXXFLAGS) -Wl,-dead_strip bench_rt.cpp rt_i32x4.o -o $@
bench_rt_x8: bench_rt.cpp rt_i32x8.o rt_ispc.h
	$(CXX) $(CXXFLAGS) -Wl,-dead_strip bench_rt.cpp rt_i32x8.o -o $@
run-rt: bench_rt_x4 bench_rt_x8 | $(REFOUT)
	./bench_rt_x4 $(REFOUT)/rt.bin
	./bench_rt_x8

clean:
	rm -f $(BINS) $(OBJS) $(HDRS)
