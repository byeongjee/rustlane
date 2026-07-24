#!/bin/bash
# Arch-aware single-tenant measurement. Generates reference outputs, then runs
# 5 interleaved rounds over every variant. Each binary does its own warmup +
# internal min-of-N; the parser takes the min across rounds. Run from anywhere.
set -u
cd "$(dirname "$0")/.."

ARCH=$(uname -m); case "$ARCH" in arm64) ARCH=aarch64;; esac
LOG="rustlane-bench/measure-log.$ARCH.txt"
: > "$LOG"

# rustlane build dirs: one on aarch64, two (v3 + native) on x86.
if [ "$ARCH" = x86_64 ]; then
  RL_DIRS="target-v3/release target-native/release"
else
  RL_DIRS="target/release"
fi
STEMS="mandel options stencil volume ao rt"
RLBINS="mandelbrot options stencil volume ao rt"

# --- reference outputs (rustlane bins read ispc-ref/ref-out/*.bin) ---
( cd ispc-ref && make refs ) >/dev/null 2>&1

run() { # round, workdir, binary
  echo "### round=$1 bin=$3" >> "$LOG"
  ( cd "$2" && "./$3" ) >> "$LOG" 2>&1
  echo "### exit=$?" >> "$LOG"
  sleep 2
}

for r in 1 2 3 4 5; do
  echo "### ---- ROUND $r ----" >> "$LOG"
  for d in $RL_DIRS; do
    for k in $RLBINS; do run $r . "$d/$k"; done
  done
  for s in $STEMS; do
    run $r ispc-ref "bench_${s}_a"
    run $r ispc-ref "bench_${s}_b"
  done
done
echo "### ALL DONE" >> "$LOG"
echo "wrote $LOG"
