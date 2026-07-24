# Serial-only reference binaries for the ISPC benchmark suite.
#
# The former vegen-based build was removed (vegen is defunct). Each benchmark is
# now built as a plain serial binary by linking its <name>_serial.cpp against the
# matching test-<name>.cpp driver with clang++ -O3. The test drivers already
# #if-0 out their ISPC/tasks paths, so only the serial symbol is needed at link.
#
# Targets:
#   make            build all serial binaries (default)
#   make run        build and run all of them
#   make run-<name> run a single benchmark
#   make clean      remove binaries and generated image/debug outputs
# The ISPC-vs-serial speedup harness lives separately under ../ispc-ref/.

CXX      ?= clang++
CXXFLAGS ?= -O3

BENCHES := ao rt options stencil mandelbrot volume

.PHONY: all run clean \
        run-ao run-rt run-options run-stencil run-mandelbrot run-volume

all: $(BENCHES)

# Each binary = serial implementation + its test driver (serial path only).
$(BENCHES): %: %_serial.cpp test-%.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

run: run-ao run-rt run-options run-stencil run-mandelbrot run-volume

run-ao: ao
	./ao 500 500
run-rt: rt
	./rt sponza
run-options: options
	./options
run-stencil: stencil
	./stencil
run-mandelbrot: mandelbrot
	./mandelbrot
run-volume: volume
	./volume camera.dat density_lowres.vol

clean:
	rm -f $(BENCHES) *.ppm debug.txt
