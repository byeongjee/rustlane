# Performance: rustlane vs ISPC

Measurements of `rustlane` — an ISPC-style SPMD-on-SIMD programming model
implemented entirely at the library level (proc macros + nightly `std::simd`,
no compiler modification) — against Intel ISPC on seven kernels from six
programs of ISPC's example/benchmark suite (`options` contributes two), on two
machines: Apple M2 Pro (aarch64/NEON) and AMD Ryzen 9 7900X (Zen4, AVX2 /
AVX-512).

rustlane is fixed at 8 lanes on both platforms. Compared against ISPC at that
same width:

| Platform | ISPC target | geomean rustlane / ISPC |
|---|---|---:|
| aarch64 / NEON | `neon-i32x8` | **0.82** |
| x86-64 / AVX2 | `avx2-i32x8` | **0.99** |

The C++ reference appears as two columns throughout: `scalar`
(`-fno-vectorize`, the one-lane floor) and `C++ auto-vec` (`-O3`, the same
source with auto-vectorization on) — see [Baselines](#baselines).

## Apple Silicon (M2 Pro, NEON)

Six of seven kernels run at parity or faster than the best ISPC NEON target;
volume is ~13% slower.

| Kernel | scalar | C++ auto-vec | ISPC neon-i32x4 | ISPC neon-i32x8 | rustlane | rustlane / ISPC-best |
|---|---:|---:|---:|---:|---:|---:|
| mandelbrot | 77.4 ms | 77.3 ms | 26.1 ms | 14.7 ms | **12.3 ms** | **0.84** |
| options: black_scholes | 1.57 ms | 1.31 ms | 0.58 ms | 0.52 ms | **0.52 ms** | **0.99** |
| options: binomial_put | 159.7 ms | 126.5 ms | 43.0 ms | 26.3 ms | 27.0 ms | 1.03 |
| stencil | 339.9 ms | 108.9 ms | 109.9 ms | 101.7 ms | **94.7 ms** | **0.93** |
| volume | 3205 ms | 3112 ms | 2267 ms | 1947 ms | 2198 ms | 1.13 |
| ao | 888.2 ms | 891.9 ms | 450.0 ms | 377.9 ms | **192.2 ms** | **0.51** |
| rt | 306.5 ms | 306.7 ms | 102.9 ms | 91.7 ms | **50.4 ms** | **0.55** |

`rustlane / ISPC-best` uses the faster of the two ISPC NEON targets
(`neon-i32x8` in every row). Against the scalar floor rustlane is 1.5×
(volume) to 6.3× (mandelbrot) faster.

## x86-64 (AMD Ryzen 9 7900X, Zen4)

At the same-ISA, same-width axis — rustlane 8-wide vs ISPC `avx2-i32x8`, both
256-bit AVX2 — rustlane is at parity on Zen4: geomean 0.99.

| Kernel | scalar | C++ auto-vec | ISPC avx2-i32x8 | ISPC avx512skx-i32x16 | rustlane v3 (AVX2) | rustlane native (AVX-512VL) | rustlane(v3) / ISPC-avx2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| mandelbrot | 47.0 ms | 47.0 ms | 8.22 ms | 5.97 ms | **6.88 ms** | 7.71 ms | **0.84** |
| options: black_scholes | 2.39 ms | 2.11 ms | 0.34 ms | 0.23 ms | 0.45 ms | 0.45 ms | 1.33 |
| options: binomial_put | 173.4 ms | 94.4 ms | 28.4 ms | 24.5 ms | **27.6 ms** | 27.7 ms | **0.97** |
| stencil | 252.9 ms | 110.6 ms | 86.6 ms | 83.9 ms | 97.1 ms | 96.7 ms | 1.12 |
| volume | 3739 ms | 3894 ms | 1518 ms | 1208 ms | 2070 ms | 2122 ms | 1.36 |
| ao | 957.7 ms | 961.5 ms | 164.5 ms | 129.3 ms | **88.2 ms** | 91.3 ms | **0.54** |
| rt | 184.2 ms | 151.0 ms | 34.4 ms | 25.7 ms | 36.8 ms | 36.9 ms | 1.07 |

### AVX-512VL buys nothing at 8 lanes

`rustlane v3` is built `-C target-cpu=x86-64-v3` (AVX2, no AVX-512); `rustlane
native` is `-C target-cpu=native` (Zen4, AVX-512VL enabled). rustlane is fixed
at 8 lanes, so `native` uses AVX-512VL *encodings* (k-mask ops, `vpternlog`)
on 256-bit `ymm` — it does not get wider.

| Kernel | native / v3 |
|---|---:|
| mandelbrot | 1.12 (native **slower**) |
| black_scholes | 1.00 |
| binomial_put | 1.00 |
| stencil | 1.00 |
| volume | 1.03 |
| ao | 1.04 |
| rt | 1.00 |
| **geomean** | **1.03** (native ~3% slower) |

The two builds are within noise on six kernels, and on mandelbrot the AVX-512
build is 12% slower — AVX-512 frequency behavior and wider encodings that buy
nothing for this masked-loop shape. Ship `-C target-cpu=x86-64-v3` on x86;
`native` adds no benefit and can regress.

### The 16-wide ISPC gang

ISPC can target a 16-wide gang (`avx512skx-i32x16`) that rustlane cannot match
at its fixed 8 lanes. Against that wider target rustlane is a geomean **1.26×
slower**, the expected width gap — partly offset because Zen4 double-pumps
512-bit ops, so the wider gang gains only 1.03×–1.48× over `avx2-i32x8` rather
than a full 2×. The column is shown for reference, not as a same-width
comparison. A 16-wide rustlane is future work.

## What carries over between platforms

Same-width ratio `rustlane / ISPC-8wide` on each platform:

| Kernel | aarch64 (rl / neon-i32x8) | x86 (rl-v3 / avx2-i32x8) | |
|---|---:|---:|---|
| mandelbrot | 0.84 | 0.84 | identical — advantage fully portable |
| ao | 0.51 | 0.54 | ~2× win holds (1-D lane mapping helps on both) |
| binomial_put | 1.03 | 0.97 | ~parity on both |
| stencil | 0.93 | 1.12 | NEON edge → small x86 loss |
| black_scholes | 0.99 | 1.33 | parity → x86 loss |
| volume | 1.13 | 1.36 | loss on both, larger on x86 |
| rt | 0.55 | 1.07 | biggest change: 1.8× win → parity |
| **geomean** | **0.82** | **0.99** | |

The kernels where rustlane's own codegen and lane mapping are the
differentiator (mandelbrot, ao) port their advantage intact. The kernels where
rustlane was at parity-or-ahead on NEON (rt, stencil, black_scholes) lose that
edge on x86, where ISPC's AVX2 backend is relatively stronger. Net: an 18%
NEON lead becomes parity.

## Why it is fast

1. **Zero-overhead masking by construction.** Uniform (scalar) control flow
   monomorphizes to plain branches and stores via zero-sized execution-context
   types; mask blends exist only where control flow is actually divergent.
   Verified at the assembly level: the macro-lowered kernels are
   instruction-equivalent to hand-written `std::simd`.
2. **No unconditional coherence guards.** Emitting ISPC's `cif`-style any-lane
   checks on every varying `if` measured a 3.0× slowdown on NEON (the guard
   becomes a horizontal reduce on the loop-carried mask chain); the macro emits
   them only for loop exits and opt-in `cif!`.
3. **Contiguity in the type system.** `foreach` indices carry a `LinearIndex`
   type; index arithmetic with uniform offsets stays contiguous, so stencil's
   19 taps are plain vector loads — where ISPC pattern-matches in the compiler,
   rustlane encodes it in types.
4. **Bounds checking that vanishes.** Safe gathers/scatters prove whole-vector
   bounds with one scalar reduction and fall back to a masked path only on
   failure — memory safety at ~1 instruction of cost.

## Caveats

- **ao and rt overstate pure codegen wins.** Part of the gap comes from a
  different (legitimate) lane mapping: rustlane's `foreach` forms are 1-D/2-D,
  so ao maps lanes to adjacent columns rather than ISPC's 4-D tile of pixel
  subsamples, which improves ray coherence. The comparison holds as "what each
  system's natural program achieves", not as isolated code-generation quality.
- **volume is the largest loss on both platforms** (1.13× NEON, 1.36× x86). It
  is gather-bound (8 gathers per trilinear sample); the safe-gather fast path
  narrowed the NEON gap from 21% to ~13%, and the remainder is the price of
  memory-safe gathers vs ISPC's unchecked ones.
- **binomial_put needed source-level help twice**: explicit `mul_add` to match
  ISPC's fp-contraction (rustc never contracts; this also made the output
  bit-exact), and a manually rotated register carry because LLVM would not
  forward a loaded value across the loop back-edge past an intervening store
  (ISPC's own emission rotates registers). With both, the loop is
  instruction-isomorphic to ISPC's and within a few percent.
- **The two platforms are not directly comparable to each other** — different
  CPUs, ISPC versions (1.30.0 vs 1.31.0), and clang builds. Only the
  rustlane-vs-ISPC ratio within a platform is meaningful.

## Baselines

The C++ reference is compiled from ordinary scalar source. At `-O3` the
compiler auto-vectorizes those loops, so a single "serial" column would
conflate two different things. Both are reported: `scalar` is
`clang++ -O3 -fno-vectorize -fno-slp-vectorize`, `C++ auto-vec` is
`clang++ -O3` on the *same* source, so the pair isolates the auto-vectorizer.
The gap between them is free compiler SIMD; the gap from auto-vec to
ISPC/rustlane is what the explicit SPMD model adds.

| Kernel | aarch64 scalar → auto-vec | x86 scalar → auto-vec |
|---|---:|---:|
| mandelbrot | 1.00× (divergent control flow) | 1.00× |
| black_scholes | 1.19× | 1.13× |
| binomial_put | 1.26× | 1.84× |
| stencil | **3.12×** (clean data-parallel loop) | **2.29×** |
| volume | 1.03× (gather-bound) | 0.96× (slight pessimization) |
| ao | 1.00× (control flow + RNG) | 1.00× |
| rt | 1.00× (BVH traversal) | 1.22× |
| **geomean** | **1.14×** | **1.28×** |

Same qualitative shape on both: stencil vectorizes heavily, mandelbrot/ao do
not. Per-kernel magnitudes differ by compiler backend — clang's x86
auto-vectorizer touches `rt` (1.22×) where its NEON one did not.

## Correctness validation

Enforced every timed round, on both platforms:

| Kernel | Criterion vs reference |
|---|---|
| black_scholes | bit-exact vs ISPC output (0/131072 mismatches) |
| binomial_put | bit-exact vs ISPC output (0/131072) |
| rt | bit-exact vs serial ground truth on aarch64 (0/810000); statistical on x86 (below) |
| volume | 0/1060864 mismatches at 1e-3; checksum rel 2.7e-8 |
| stencil | checksum rel 1.1e-10; max abs 4.4e-5 (reference is fma-contracted; port proven byte-identical to a non-fma scalar recomputation) |
| mandelbrot | self-checksum; matches ISPC modulo fma boundary pixels (0.16%) |
| ao | statistical: 0.022% checksum deviation (RNG streams legitimately differ per gang width; ISPC's own x4/x8/serial spread is 0.04%) |

One cross-toolchain difference on x86: **rt is not bit-exact there** (it is on
the matched-toolchain aarch64 box). The clang-22 serial ground truth and the
rustc SPMD kernel contract fma differently, flipping ~0.33% of silhouette-edge
rays (2675/810000 pixels, max per-pixel 1.5%); the global image checksum still
matches to 3.9e-8 — the same order as ISPC's *own* serial-vs-SPMD divergence
there (4.3e-8). rt is therefore validated statistically on x86, like
mandelbrot's fma-boundary pixels and ao's per-width RNG spread. mandelbrot's
escape-count checksum is identical across platforms (27304085).

The library itself carries 142 workspace tests: an N=1 (scalar) vs N=8
differential suite over 30 kernels including adversarial mask-stack stressors,
a 48-case compile-fail suite for the static rejection rules, and hand-expanded
lowering-contract tests.

## Environment & methodology

Both platforms: 5 interleaved rounds over every binary in a fixed order with
2 s cool-downs; each binary does 3 warm-up + internal min-of-15 reps
(mandelbrot 20); the reported value is the minimum across rounds. Workloads are
ISPC's example defaults, identical on every implementation (same inputs, same
data files, same work per timed rep). Runner: `rustlane-bench/measure.sh`;
parser: `rustlane-bench/parse_measurements.py <arch>`; raw logs and
machine-readable results: `rustlane-bench/measure-log.<arch>.txt` and
`rustlane-bench/RESULTS.<arch>.json`.

- **aarch64** — MacBook Pro (Mac14,10), Apple M2 Pro (8P+4E), 32 GB,
  macOS 26.5.2. rustc 1.92.0-nightly (2025-10-14); ISPC 1.30.0 (LLVM 22.1.0);
  clang++ 22.1.8 (Homebrew LLVM), all at `-O3`. rustlane built with the
  aarch64 default — NEON is the baseline ISA.
- **x86-64** — AMD Ryzen 9 7900X (Zen4), 12C/24T, Ubuntu x86_64 (kernel 7.0),
  single-tenant. rustc 1.92.0-nightly (2025-10-14); ISPC 1.31.0 (LLVM 23,
  `--pic`); clang++-22 22.1.2 at `-O3`. Two rustlane builds:
  `-C target-cpu=x86-64-v3` (→ `target-v3/`) and `-C target-cpu=native`
  (→ `target-native/`). Feature sets confirmed with `rustc --print cfg`: v3
  tops out at AVX2/FMA/BMI; native adds AVX-512{F,VL,BW,DQ,VBMI,…}.

## Reproducing

```sh
# aarch64
make -C ispc-ref build-all       # ISPC + scalar/auto-vec C++ baselines (needs ispc, clang++)
cargo build --workspace --release
./rustlane-bench/measure.sh      # ~25 min
python3 rustlane-bench/parse_measurements.py aarch64
```

```sh
# x86-64
make -C ispc-ref build-all CXX=clang++-22
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release --target-dir target-v3
RUSTFLAGS="-C target-cpu=native"    cargo build --release --target-dir target-native
./rustlane-bench/measure.sh
python3 rustlane-bench/parse_measurements.py x86_64
```
