# Performance: rustlane vs ISPC — x86-64 (AMD Zen4, AVX2 / AVX-512)

First measurement of the `rustlane` library on real x86 hardware, closing the
caveat in [PERFORMANCE.md](PERFORMANCE.md) that the x86 path was only
disassembly- and Rosetta-verified. Same seven kernels, same workloads, same
taxonomy as the Apple Silicon report — run on an AMD Ryzen 9 7900X (Zen4).

## Headline

At the **fair same-ISA, same-width axis** (rustlane 8-wide vs ISPC
`avx2-i32x8`, both 256-bit AVX2), rustlane is **at parity with ISPC on Zen4:
geomean 0.99**. This is the honest contrast with NEON, where rustlane is ~18%
faster than ISPC at the same width ([PERFORMANCE.md](PERFORMANCE.md)). rustlane's
NEON codegen advantage does not carry to x86 in aggregate — though it carries
per-kernel for mandelbrot and ao (see below).

| Kernel | scalar | C++ auto-vec | ISPC avx2-i32x8 | ISPC avx512skx-i32x16 | rustlane v3 (AVX2) | rustlane native (AVX-512VL) | rustlane(v3) / ISPC-avx2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| mandelbrot | 47.0 ms | 47.0 ms | 8.22 ms | 5.97 ms | **6.88 ms** | 7.71 ms | **0.84** |
| options: black_scholes | 2.39 ms | 2.11 ms | 0.34 ms | 0.23 ms | 0.45 ms | 0.45 ms | 1.33 |
| options: binomial_put | 173.4 ms | 94.4 ms | 28.4 ms | 24.5 ms | **27.6 ms** | 27.7 ms | **0.97** |
| stencil | 252.9 ms | 110.6 ms | 86.6 ms | 83.9 ms | 97.1 ms | 96.7 ms | 1.12 |
| volume | 3739 ms | 3894 ms | 1518 ms | 1208 ms | 2070 ms | 2122 ms | 1.36 |
| ao | 957.7 ms | 961.5 ms | 164.5 ms | 129.3 ms | **88.2 ms** | 91.3 ms | **0.54** |
| rt | 184.2 ms | 151.0 ms | 34.4 ms | 25.7 ms | 36.8 ms | 36.9 ms | 1.07 |

## What carries over from NEON, and what doesn't

Comparing the same-width ratio `rustlane / ISPC-8wide` on each platform:

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

The kernels where rustlane's own codegen/lane-mapping is the differentiator
(mandelbrot, ao) port their advantage intact. The kernels where rustlane was at
parity-or-ahead on NEON (rt, stencil, black_scholes) lose that edge on x86,
where ISPC's AVX2 backend is relatively stronger. Net: an 18% NEON lead becomes
parity.

## AVX-512VL buys rustlane nothing at 8 lanes

`rustlane v3` is built `-C target-cpu=x86-64-v3` (AVX2, no AVX-512); `rustlane
native` is `-C target-cpu=native` (Zen4, AVX-512VL enabled). rustlane is fixed
at 8 lanes, so `native` uses AVX-512VL *encodings* (k-mask ops, `vpternlog`) on
256-bit `ymm` — it does not get wider.

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

The masking-primitive win we hypothesized does not materialize at 8 lanes on
Zen4: the two builds are within noise on six kernels, and on mandelbrot the
AVX-512 build is 12% slower (AVX-512 frequency behavior / the wider encodings
buying nothing for this masked-loop shape). **Takeaway: for rustlane's current
8-wide v1, `-C target-cpu=x86-64-v3` is the build to ship on x86; `native` adds
no benefit and can regress.**

## The wider ISPC gang (16-wide AVX-512)

ISPC can target a 16-wide gang (`avx512skx-i32x16`) that rustlane v1 cannot
match — it is fixed at 8 lanes. Against that wider ISPC target rustlane is a
geomean **1.26× slower**, which is the expected width gap (partly offset because
Zen4 double-pumps 512-bit ops, so the wider gang gains only 1.03×–1.48× over
`avx2-i32x8` rather than a full 2×). This column is shown for reference, not as a
same-width comparison. A 16-wide rustlane is future work.

## Two honest baselines (scalar vs auto-vectorized C++)

As on NEON, the C++ reference is split into a true `scalar (no-vec)` floor and
`-O3` auto-vectorized C++ (the "serial" of older reports). On x86 the
auto-vectorizer is a bit more effective on average (geomean scalar/auto-vec
1.28× vs 1.14× on NEON):

| Kernel | scalar → auto-vec |
|---|---:|
| mandelbrot | 1.00× (no auto-vec — divergent control flow) |
| black_scholes | 1.13× |
| binomial_put | 1.84× |
| stencil | 2.29× (large) |
| volume | 0.96× (none; slight pessimization) |
| ao | 1.00× (none) |
| rt | 1.22× |

Same qualitative shape as NEON (stencil vectorizes heavily; mandelbrot/ao/volume
do not), with per-kernel magnitudes differing by compiler backend — e.g. clang's
x86 auto-vectorizer touches `rt` (1.22×) where its NEON one did not (1.00×).

## Environment & methodology

- AMD Ryzen 9 7900X (Zen4), 12C/24T, Ubuntu x86_64 (kernel 7.0), single-tenant.
  ISPC 1.31.0 (LLVM 23); clang++-22 22.1.2; rustc 1.92.0-nightly (2025-10-14).
  ISPC `--pic`; C++ at `-O3`; rustlane at 8 lanes.
- Two rustlane builds: `RUSTFLAGS="-C target-cpu=x86-64-v3"` (→ `target-v3/`)
  and `-C target-cpu=native` (→ `target-native/`). Feature sets confirmed with
  `rustc --print cfg`: v3 tops out at AVX2/FMA/BMI; native adds AVX-512{F,VL,BW,
  DQ,VBMI,…}.
- 5 interleaved rounds, fixed order, 2 s cool-downs; each binary does 3 warm-up
  + internal min-of-15 (mandelbrot 20); reported value = min across rounds.
  Runner `rustlane-bench/measure.sh`; parser
  `rustlane-bench/parse_measurements.py x86_64`; log
  `rustlane-bench/measure-log.x86_64.txt`; results
  `rustlane-bench/RESULTS.x86_64.json`.
- Same C++ source drives the scalar and auto-vec columns; only
  `-fno-vectorize -fno-slp-vectorize` differs.

## Correctness on x86

All seven kernels validate every round on both builds against x86-native ISPC
reference output, with one documented cross-toolchain difference:

- **rt is not bit-exact on x86** (it is on the matched-toolchain aarch64 box).
  The clang-22 serial ground truth and the rustc SPMD kernel contract fma
  differently, flipping ~0.33% of silhouette-edge rays (2675/810000 pixels,
  max per-pixel 1.5%); the global image checksum still matches to 3.9e-8 — the
  same order as ISPC's *own* serial-vs-SPMD divergence there (4.3e-8). rt is
  therefore validated statistically (checksum + bounded edge fraction), like
  mandelbrot's fma-boundary pixels and ao's per-width RNG spread, rather than by
  strict per-pixel bit-exactness.
- mandelbrot's escape-count checksum is identical to the aarch64 value
  (27304085); black_scholes/binomial/stencil/volume pass at their documented
  tolerances; ao passes statistically (0.02% checksum, per-gang-width RNG).

## Reproducing

```sh
# on the x86 box, from a checkout at ~/rust-ispc
make -C ispc-ref build-all CXX=clang++-22
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo +nightly-2025-10-14 build --release --target-dir target-v3
RUSTFLAGS="-C target-cpu=native"   cargo +nightly-2025-10-14 build --release --target-dir target-native
./rustlane-bench/measure.sh                          # writes measure-log.x86_64.txt
python3 rustlane-bench/parse_measurements.py x86_64
```
