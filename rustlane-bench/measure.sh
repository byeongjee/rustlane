#!/bin/bash
# Final single-tenant measurement: 5 interleaved rounds over every benchmark
# variant (rustlane / ISPC neon-i32x4 / ISPC neon-i32x8; serial times are printed
# by the ISPC drivers). Each binary does its own warmup + internal min-of-N;
# the parser takes the minimum across rounds. Run from anywhere.
set -u
cd "$(dirname "$0")/.."
LOG=rustlane-bench/measure-log.txt
: > "$LOG"

run() { # round, workdir, binary
  echo "### round=$1 bin=$3" >> "$LOG"
  (cd "$2" && "./$3") >> "$LOG" 2>&1
  echo "### exit=$?" >> "$LOG"
  sleep 2
}

for r in 1 2 3 4 5; do
  echo "### ---- ROUND $r ----" >> "$LOG"
  run $r . target/release/mandelbrot
  run $r ispc-ref bench_mandel_x4
  run $r ispc-ref bench_mandel_x8
  run $r . target/release/options
  run $r ispc-ref bench_options_x4
  run $r ispc-ref bench_options_x8
  run $r . target/release/stencil
  run $r ispc-ref bench_stencil_x4
  run $r ispc-ref bench_stencil_x8
  run $r . target/release/volume
  run $r ispc-ref bench_volume_x4
  run $r ispc-ref bench_volume_x8
  run $r . target/release/ao
  run $r ispc-ref bench_ao_x4
  run $r ispc-ref bench_ao_x8
  run $r . target/release/rt
  run $r ispc-ref bench_rt_x4
  run $r ispc-ref bench_rt_x8
done
echo "### ALL DONE" >> "$LOG"
