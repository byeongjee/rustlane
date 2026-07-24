# Performance: spmd vs ISPC

Final measurements of the `spmd` library — an ISPC-style SPMD-on-SIMD
programming model for Rust, implemented entirely at the library level
(proc macros + nightly `std::simd`, no compiler modification) — against
Intel ISPC 1.30 on the six kernels of ISPC's example/benchmark suite.

## Headline

**Geometric mean of spmd/ISPC-best runtime across the 7 kernels: 0.784**
(spmd is ~22% faster on average). Six of seven kernels run at parity or
faster than the best ISPC NEON target; one (volume) is 12.8% slower.

| Kernel | Serial C++ | ISPC neon-i32x4 | ISPC neon-i32x8 | spmd (Rust) | spmd / ISPC-best | spmd vs serial |
|---|---:|---:|---:|---:|---:|---:|
| mandelbrot | 77.9 ms | 26.3 ms | 14.7 ms | **12.3 ms** | **0.84** | 6.3× |
| options: black_scholes | 1.31 ms | 0.70 ms | 0.72 ms | **0.52 ms** | **0.74** | 2.5× |
| options: binomial_put | 126.7 ms | 43.2 ms | 26.4 ms | **26.9 ms** | **1.02** | 4.7× |
| stencil | 109.6 ms | 108.2 ms | 102.4 ms | **93.4 ms** | **0.91** | 1.2× |
| volume | 3116 ms | 2277 ms | 1954 ms | 2204 ms | 1.13 | 1.4× |
| ao | 893 ms | 452 ms | 379 ms | **192 ms** | **0.51** | 4.6× |
| rt | 308.6 ms | 103.1 ms | 92.5 ms | **50.8 ms** | **0.55** | 6.1× |

## Environment & methodology

- MacBook Pro (Mac14,10), Apple M2 Pro (8P+4E), 32 GB, macOS 26.5.2 —
  aarch64 NEON. rustc 1.92.0-nightly (2025-10-14); ISPC 1.30.0 (LLVM
  22.1.0); clang++ (Homebrew LLVM 22), all at `-O3`.
- 5 interleaved rounds over every binary in a fixed order with 2 s
  cool-downs; each binary performs 3 warm-up + internal min-of-15 reps
  (mandelbrot: 20); the reported value is the minimum across rounds.
  Runner: `spmd-bench/measure.sh`; parser: `spmd-bench/parse_measurements.py`;
  raw log: `spmd-bench/measure-log.txt`; machine-readable results:
  `spmd-bench/RESULTS.final.json`.
- Workloads are ISPC's example defaults, identical on every
  implementation (same inputs, same data files, same work per timed rep).

## Correctness validation (enforced every timed round)

| Kernel | Criterion vs reference |
|---|---|
| black_scholes | bit-exact vs ISPC output (0/131072 mismatches) |
| binomial_put | bit-exact vs ISPC output (0/131072) |
| rt | bit-exact vs serial ground truth (0/810000) |
| volume | 0/1060864 mismatches at 1e-3; checksum rel 2.7e-8 |
| stencil | checksum rel 1.1e-10; max abs 4.4e-5 (reference is fma-contracted; port proven byte-identical to a non-fma scalar recomputation) |
| mandelbrot | self-checksum; matches ISPC modulo fma boundary pixels (0.16%) |
| ao | statistical: 0.022% checksum deviation (RNG streams legitimately differ per gang width; ISPC's own x4/x8/serial spread is 0.04%) |

Additionally, the library itself carries 140 workspace tests: an N=1
(scalar) vs N=8 differential suite over 30 kernels including adversarial
mask-stack stressors, a 48-case compile-fail suite for the static
rejection rules, and hand-expanded lowering-contract tests.

## Why it is fast

1. **Zero-overhead masking by construction.** Uniform (scalar) control
   flow monomorphizes to plain branches and stores via zero-sized
   execution-context types; mask blends exist only where control flow is
   actually divergent. Verified at the assembly level: the macro-lowered
   kernels are instruction-equivalent to hand-written `std::simd`.
2. **No unconditional coherence guards.** Emitting ISPC's `cif`-style
   any-lane checks on every varying `if` measured a 3.0× slowdown on
   NEON (the guard becomes a horizontal reduce on the loop-carried mask
   chain); the macro emits them only for loop exits and opt-in `cif!`.
3. **Contiguity in the type system.** `foreach` indices carry a
   `LinearIndex` type; index arithmetic with uniform offsets stays
   contiguous, so stencil's 19 taps are plain vector loads — where ISPC
   pattern-matches in the compiler, spmd encodes it in types.
4. **Bounds checking that vanishes.** Safe gathers/scatters prove
   whole-vector bounds with one scalar reduction and fall back to a
   masked path only on failure — memory safety at ~1 instruction of cost.

## Honest caveats

- **ao and rt overstate pure codegen wins.** Both are ~2× faster than
  ISPC, but part of that comes from different (legitimate) lane
  mappings: the library's v1 `foreach` forms are 1-D/2-D, so ao maps
  lanes to adjacent columns rather than ISPC's 4-D tile of pixel
  subsamples, which improves ray coherence. The comparison is fair as
  "what each system's natural program achieves," not as isolated
  code-generation quality.
- **volume is the one loss (1.13×).** It is gather-bound (8 gathers per
  trilinear sample); the safe-gather fast path narrowed the gap from
  21% to ~13%, and the remainder is the price of memory-safe gathers vs
  ISPC's unchecked ones.
- **binomial_put needed source-level help twice**: explicit `mul_add` to
  match ISPC's fp-contraction (rustc never contracts; this also made the
  output bit-exact), and a manually rotated register carry because LLVM
  would not forward a loaded value across the loop back-edge past an
  intervening store (ISPC's own emission rotates registers). With both,
  the loop is instruction-isomorphic to ISPC's and within 2%.
- Measured on one machine (Apple Silicon NEON). The x86-64 multi-target
  dispatch path (SSE2/SSE4.1/AVX2/AVX-512 shims + cpuid dispatch) is
  verified by disassembly and a Rosetta functional run, but not
  benchmarked on x86 hardware.

## Reproducing

```sh
make -C ispc-ref build-all      # ISPC + serial baselines (needs ispc, clang++)
cargo build --workspace --release
./spmd-bench/measure.sh         # ~25 min, writes spmd-bench/measure-log.txt
python3 spmd-bench/parse_measurements.py
```
