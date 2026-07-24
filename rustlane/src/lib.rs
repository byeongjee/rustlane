//! `rustlane` — the SPMD-on-SIMD facade crate.
//!
//! This is the single crate users depend on. It re-exports, in one place, the
//! two halves of the library so downstream code needs exactly one dependency:
//!
//! * the **procedural macros** from [`rustlane_macros`] — the [`macro@kernel`] and
//!   [`macro@export`] attributes, the [`macro@SpmdValue`] derive, and the
//!   kernel-world function-like macros ([`foreach!`](macro@foreach),
//!   [`foreach_2d!`](macro@foreach_2d), [`foreach_tiled!`](macro@foreach_tiled),
//!   [`unmasked!`](macro@unmasked), [`cif!`](macro@cif),
//!   [`cwhile!`](macro@cwhile));
//! * the **runtime** from [`rustlane_core`] — the [`Varying`] value type, the
//!   execution-context types and mask-stack machinery, the condition/memory
//!   traits, and the [`prelude`] a kernel body pulls in with
//!   `use rustlane::prelude::*`.
//!
//! Typical usage:
//!
//! ```ignore
//! use rustlane::prelude::*;
//! use rustlane::{kernel, export};
//!
//! #[kernel]
//! fn square(x: Varying<f32>) -> Varying<f32> { x * x }
//! ```
//!
//! # Building on x86-64
//!
//! Kernels operate on 8-lane vectors. The x86-64 baseline ISA is only SSE2
//! (128-bit), so a *default* build lowers each 8-lane operation to a pair of
//! SSE instructions and runs well below par. Enable AVX2 — e.g. with a
//! `.cargo/config.toml`:
//!
//! ```toml
//! [target.'cfg(target_arch = "x86_64")']
//! rustflags = ["-C", "target-cpu=x86-64-v3"]  # AVX2 + FMA
//! ```
//!
//! `target-cpu=native` is not worth it: AVX-512VL buys the fixed 8-wide code
//! nothing — and can slightly regress it — on current hardware. aarch64/NEON
//! needs no flag; NEON is the baseline ISA there.
#![feature(portable_simd)]

pub use rustlane_macros::{
    cif, cwhile, export, foreach, foreach_2d, foreach_tiled, kernel, unmasked, SpmdValue,
};

pub use rustlane_core::*;
