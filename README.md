# rustlane

**rustlane brings ISPC's SPMD-on-SIMD programming model to Rust as a pure
library** — you write natural scalar-looking control flow (`if`, `while`,
`for`, `break`, early `return`) over *varying* values, and proc macros lower it
to masked SIMD over nightly `std::simd`. No compiler fork, no external
toolchain: just `#[kernel]`, `#[export]`, and `foreach!`.

> The name ISPC (Intel SPMD Program Compiler) is used here only to describe the
> programming model rustlane adopts. rustlane is an independent project, not
> affiliated with, endorsed by, or derived from Intel.

## Status

**Experimental. Nightly-only. The API is unstable and will change without
notice.** This is a v1 research implementation; it is not yet published to
crates.io. Do not depend on it for production work.

## What a kernel looks like

The SPMD idea: write one program as if it runs on a single lane, and the
library runs it across a whole SIMD vector of lanes at once. Control flow that
diverges between lanes is handled for you. Here is the Mandelbrot escape-time
kernel — plain `for` / `if` / `break` over *varying* values:

```rust
#![feature(portable_simd)]
use rustlane::prelude::*;
use rustlane::kernel;

#[kernel]
fn mandel(c_re: Varying<f32>, c_im: Varying<f32>, count: i32) -> Varying<i32> {
    let mut z_re = c_re;
    let mut z_im = c_im;
    let mut ret = Varying::splat(0);
    for i in 0..count {
        if z_re * z_re + z_im * z_im > 4.0 {
            break;
        }
        let new_re = z_re * z_re - z_im * z_im;
        let new_im = 2.0 * z_re * z_im;
        unmasked! {
            z_re = c_re + new_re;
            z_im = c_im + new_im;
        }
        ret = i + 1;
    }
    ret
}
```

The macro rewrites this into masked `std::simd`: the varying `if`/`break`
become per-lane mask updates and blends (`ret = i + 1` stores only into lanes
still iterating), and the `for` loop keeps running until every lane's mask is
off, so a lane that has already escaped stops contributing without stopping the
others.

## Performance

Geometric mean of rustlane / best-ISPC runtime across 7 kernels of ISPC's own
example suite: **0.82** (rustlane ~18% faster on average on Apple M2 Pro /
NEON). Six of seven kernels reach parity or beat the best ISPC NEON target; one
(volume) is slower. x86 (AMD Zen4) numbers are in
[PERFORMANCE-x86.md](PERFORMANCE-x86.md).

The C++ baseline is split into a **true scalar** floor (`-fno-vectorize`) and
**auto-vectorized** C++ (`-O3`) — the old single "Serial C++" column was really
the auto-vectorized one. (For stencil the compiler's auto-vectorizer alone is
3.1×; for mandelbrot/ao/rt it does essentially nothing.)

| Kernel | scalar | C++ auto-vec | ISPC neon-i32x4 | ISPC neon-i32x8 | rustlane | rustlane / ISPC-best |
|---|---:|---:|---:|---:|---:|---:|
| mandelbrot | 77.4 ms | 77.3 ms | 26.1 ms | 14.7 ms | **12.3 ms** | **0.84** |
| options: black_scholes | 1.57 ms | 1.31 ms | 0.58 ms | 0.52 ms | **0.52 ms** | **0.99** |
| options: binomial_put | 159.7 ms | 126.5 ms | 43.0 ms | 26.3 ms | 27.0 ms | 1.03 |
| stencil | 339.9 ms | 108.9 ms | 109.9 ms | 101.7 ms | **94.7 ms** | **0.93** |
| volume | 3205 ms | 3112 ms | 2267 ms | 1947 ms | 2198 ms | 1.13 |
| ao | 888.2 ms | 891.9 ms | 450.0 ms | 377.9 ms | **192.2 ms** | **0.51** |
| rt | 306.5 ms | 306.7 ms | 102.9 ms | 91.7 ms | **50.4 ms** | **0.55** |

> Honest caveat: `ao` and `rt` gain partly from a legitimately different
> lane-to-work mapping (rustlane's v1 `foreach` forms are 1-D/2-D, not ISPC's
> 4-D tile), not from codegen alone, and `volume` is a genuine ~13% loss to the
> cost of memory-safe gathers — see [PERFORMANCE.md](PERFORMANCE.md) for full
> methodology, environment, and correctness validation.

## Quick start

rustlane requires a nightly toolchain (for `#![feature(portable_simd)]`). Pin it
with a `rust-toolchain.toml` next to your `Cargo.toml`:

```toml
[toolchain]
channel = "nightly"
```

On **x86-64**, build with AVX2 enabled — otherwise rustlane's 8-lane vectors
lower to paired 128-bit SSE2 (the x86-64 baseline ISA) and run well below the
benchmark numbers. Add a `.cargo/config.toml` next to your `Cargo.toml`:

```toml
[target.'cfg(target_arch = "x86_64")']
rustflags = ["-C", "target-cpu=x86-64-v3"]   # AVX2 + FMA
```

`target-cpu=native` is not worth it here: AVX-512VL gives rustlane's fixed
8-wide code no benefit — and can slightly regress it — on current hardware (see
[PERFORMANCE-x86.md](PERFORMANCE-x86.md)). aarch64/NEON needs no flag; NEON is
the baseline ISA there.

**Not yet on crates.io** (publishing is pending). For now, depend on it by
path from a checkout of this repository:

```toml
# Cargo.toml
[dependencies]
rustlane = { path = "../rustlane" }
```

A complete program — define a `#[kernel]`, drive it lane-by-lane from an
all-uniform `#[export]` entry point with `foreach!`, and call that entry point
from ordinary Rust:

```rust
#![feature(portable_simd)]
use rustlane::prelude::*;
use rustlane::{export, kernel};

// The per-lane program.
#[kernel]
fn scale(x: Varying<f32>, factor: f32) -> Varying<f32> {
    x * factor
}

// The all-uniform entry point. `foreach!` hands each chunk of `input`
// to the kernel as a `Varying`; `output[i] = ..` is a contiguous store.
#[export]
fn scale_all(input: &[f32], output: &mut [f32], factor: f32) {
    foreach!(i in 0..input.len() {
        output[i] = scale(input[i], factor);
    });
}

fn main() {
    let input: Vec<f32> = (0..1024).map(|v| v as f32).collect();
    let mut output = vec![0.0f32; input.len()];
    scale_all(&input, &mut output, 2.0);
    assert_eq!(output[10], 20.0);
    println!("scaled {} elements", output.len());
}
```

```sh
cargo +nightly build          # or just `cargo build` with the toolchain file
cargo +nightly run
cargo +nightly test --workspace
```

`#[export]` compiles the kernel tree once per SIMD target behind
`#[target_feature]` shims and picks the widest available at runtime (see
[How it works](#how-it-works)); the `scale_all` you call is a safe, ordinary
Rust function.

## Feature matrix

Source of truth: [`rustlane-core/LOWERING.md`](rustlane-core/LOWERING.md) §14.

| Supported in v1 | v1 limitations |
|---|---|
| `if` / `else` with divergent (varying) conditions | No `match` on varying values (rewrite as `if`/`else` chains) |
| `while` / `for` / bare `loop`, with `break` / `continue` / early `return` under divergence | Structs vectorize to `VaryingS<N>` (a generated SoA type), **not** `Varying<S>` |
| `unmasked! { .. }` (all-lanes block for loop-carried locals) | No `foreach_3d!` / `foreach_4d!` (only 1-D, 2-D, and tiled) |
| `cif!` / `cwhile!` (opt-in *coherent* control flow, ISPC-style `any()` guards) | Kernel calls must be single-segment lowercase paths; qualified-path calls are not exec-threaded |
| `foreach!` / `foreach_2d!` / `foreach_tiled!` inline iteration (no closures) | Uniform `while` conditions are unusable inside kernels (use `for` or `loop { if !c { break; } }`) |
| `#[derive(SpmdValue)]` structs with `#[spmd(uniform)]` fields | Nightly-only (`#![feature(portable_simd)]`); no stable-Rust path |
| Multi-target x86 dispatch (SSE2 / SSE4.1 / AVX2 / AVX-512) + aarch64 NEON | User type/const generics on kernels are rejected (the macro owns `const N`) |
| `math` stdlib incl. ISPC-ported transcendentals (`exp` / `log` / `pow` / `sin` / `cos` / `rsqrt` / `rcp`) | |
| `rng`: LFSR113 combined-Tausworthe varying RNG | |
| `reduce`: horizontal reductions + cross-lane ops (broadcast / rotate / shift / scan / pack) | |

## Static safety

The dangerous SPMD mistakes are compile errors, not silent races. Assigning to
a *uniform* (scalar) variable while lanes are diverging would make every lane
race on one location, so there is simply no such trait impl — the write fails
to type-check with a domain-specific diagnostic:

```text
error[E0277]: cannot assign to a value of type `i32` under execution context `VMask<N>`
  --> tests/ui/uniform_assign.rs:14:7
   |
14 |     s = 5;
   |       ^ this assignment target cannot be written under the current control-flow mask
   |
   = note: assigning to a uniform (scalar) variable under VARYING control flow is
     not supported: every lane would race on one location. Make the variable a
     `Varying`, hoist the assignment out of the varying branch, or wrap it in
     `unmasked!` if all-lanes semantics are intended
```

Two safety properties hold by construction:

- **Inactive lanes never touch memory.** Every masked gather/scatter carries
  the active mask as its hardware enable, so an out-of-bounds index on a
  masked-off lane neither faults nor affects results.
- **48-case compile-fail diagnostics suite.** The static rejection rules for
  `#[kernel]` and `#[export]` (LOWERING.md §14.7) are pinned by 48 checked-in
  `trybuild` snapshots — each asserts the error span points at the offending
  user token — plus an additional case covering the `SpmdValue` derive.

## How it works

Each execution context (all-on, a uniform branch bool, a varying mask, a
uniform branch under a varying mask) is a distinct Rust *type* that the macro
threads through the kernel as a hidden first parameter; every operator,
assignment, and control-flow construct resolves against that type by trait, so
the macro itself never inspects a value or a type. Because the context is a
type and the "is this uniform?" test is a `const`, monomorphization deletes the
mask machinery on every path that turns out uniform: a kernel entered all-on
compiles down to plain scalar branches and contiguous vector loads, with blends
emitted *only* where control flow is genuinely divergent. There are no
unconditional coherence guards — the one `any()` reduction per loop is the loop
exit check, not a per-`if` tax. The whole kernel tree is `#[inline(always)]`, so
`#[export]` can stamp it out once per SIMD target inside a `#[target_feature]`
shim and pick the widest at runtime with a single cached indirect call.

See [`rustlane-core/LOWERING.md`](rustlane-core/LOWERING.md) for the full
lowering contract and [PERFORMANCE.md](PERFORMANCE.md) for measured results and
methodology.

## License

Licensed under either of

- MIT license ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option. Unless you explicitly state otherwise, any contribution
intentionally submitted for inclusion in the work by you, as defined in the
Apache-2.0 license, shall be dual-licensed as above, without any additional
terms or conditions.

Portions of the benchmark kernels and math routines are ported from Intel's
ISPC examples under their original BSD-3-Clause terms; see
[THIRD-PARTY.md](THIRD-PARTY.md) for the full attributions.
