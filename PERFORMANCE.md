# Performance: rustlane vs ISPC — Apple Silicon (NEON)

Measurements of the `rustlane` library — an ISPC-style SPMD-on-SIMD
programming model for Rust, implemented entirely at the library level
(proc macros + nightly `std::simd`, no compiler modification) — against
Intel ISPC 1.30 on seven kernels from six programs of ISPC's
example/benchmark suite (`options` contributes two). x86 (AMD Zen4) numbers
are in [PERFORMANCE-x86.md](PERFORMANCE-x86.md).

## Two honest baselines, not one "serial"

The C++ reference is compiled from ordinary scalar source. At `-O3` the
compiler *auto-vectorizes* those scalar loops, so a single "serial" column
conflates two very different things. We report both:

- **scalar (no-vec)** — `clang++ -O3 -fno-vectorize -fno-slp-vectorize`: the
  true one-lane floor.
- **C++ auto-vec** — `clang++ -O3`: the same source with auto-vectorization on.
  This is what earlier versions of this report labelled "Serial C++".

The gap between them is "free" compiler SIMD; the gap from auto-vec to
ISPC/rustlane is what the explicit SPMD model adds.

| Kernel | scalar (no-vec) | C++ auto-vec | ratio | what auto-vec did |
|---|---:|---:|---:|---|
| mandelbrot | 77.4 ms | 77.3 ms | 1.00× | nothing (divergent control flow) |
| black_scholes | 1.57 ms | 1.31 ms | 1.19× | modest (transcendentals) |
| binomial_put | 159.7 ms | 126.5 ms | 1.26× | modest |
| stencil | 339.9 ms | 108.9 ms | **3.12×** | large (clean data-parallel loop) |
| volume | 3205 ms | 3112 ms | 1.03× | little (gather-bound) |
| ao | 888.2 ms | 891.9 ms | 1.00× | nothing (control flow + RNG) |
| rt | 306.5 ms | 306.7 ms | 1.00× | nothing (BVH traversal) |

So for four of seven kernels the compiler cannot auto-vectorize the scalar
source at all (the old "serial" number was already ~scalar); for **stencil**
the old "serial" understated the true one-lane cost by 3.1×.

## Headline

**Geometric mean of rustlane / ISPC-best runtime across the 7 kernels: 0.82**
(rustlane ~18% faster on average). Six of seven kernels run at parity or
faster than the best ISPC NEON target; one (volume) is ~13% slower.

| Kernel | scalar | C++ auto-vec | ISPC neon-i32x4 | ISPC neon-i32x8 | rustlane | rustlane / ISPC-best |
|---|---:|---:|---:|---:|---:|---:|
| mandelbrot | 77.4 ms | 77.3 ms | 26.1 ms | 14.7 ms | **12.3 ms** | **0.84** |
| options: black_scholes | 1.57 ms | 1.31 ms | 0.58 ms | 0.52 ms | **0.52 ms** | **0.99** |
| options: binomial_put | 159.7 ms | 126.5 ms | 43.0 ms | 26.3 ms | 27.0 ms | 1.03 |
| stencil | 339.9 ms | 108.9 ms | 109.9 ms | 101.7 ms | **94.7 ms** | **0.93** |
| volume | 3205 ms | 3112 ms | 2267 ms | 1947 ms | 2198 ms | 1.13 |
| ao | 888.2 ms | 891.9 ms | 450.0 ms | 377.9 ms | **192.2 ms** | **0.51** |
| rt | 306.5 ms | 306.7 ms | 102.9 ms | 91.7 ms | **50.4 ms** | **0.55** |

`rustlane / ISPC-best` uses the faster of the two ISPC NEON targets (neon-i32x8
in every row). Against the *true scalar* floor rustlane is 1.5× (volume) to
6.3× (mandelbrot) faster.

## Environment & methodology

- MacBook Pro (Mac14,10), Apple M2 Pro (8P+4E), 32 GB, macOS 26.5.2 —
  aarch64 NEON. rustc 1.92.0-nightly (2025-10-14); ISPC 1.30.0 (LLVM
  22.1.0); clang++ 22.1.8 (Homebrew LLVM), all at `-O3`. rustlane built with
  the aarch64 default (NEON is the baseline ISA), 8 lanes.
- 5 interleaved rounds over every binary in a fixed order with 2 s
  cool-downs; each binary performs 3 warm-up + internal min-of-15 reps
  (mandelbrot: 20); the reported value is the minimum across rounds.
  Runner: `rustlane-bench/measure.sh`; parser:
  `rustlane-bench/parse_measurements.py aarch64`; raw log:
  `rustlane-bench/measure-log.aarch64.txt`; machine-readable results:
  `rustlane-bench/RESULTS.aarch64.json`.
- The scalar and auto-vec baselines are the *same* C++ source; only the
  `-fno-vectorize -fno-slp-vectorize` flags differ, so the pair isolates the
  compiler's auto-vectorizer.
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

Additionally, the library itself carries 142 workspace tests: an N=1
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
   pattern-matches in the compiler, rustlane encodes it in types.
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
  the loop is instruction-isomorphic to ISPC's and within a few percent.
- rustlane is 8 lanes wide on both platforms. See
  [PERFORMANCE-x86.md](PERFORMANCE-x86.md) for the x86 (Zen4) run, including
  a same-ISA AVX2 comparison and the effect of AVX-512VL masking.

## Reproducing

```sh
make -C ispc-ref build-all       # ISPC + scalar/auto-vec C++ baselines (needs ispc, clang++)
cargo build --workspace --release
./rustlane-bench/measure.sh          # ~25 min, writes rustlane-bench/measure-log.aarch64.txt
python3 rustlane-bench/parse_measurements.py aarch64
```
