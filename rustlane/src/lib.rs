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
#![feature(portable_simd)]

pub use rustlane_macros::{
    cif, cwhile, export, foreach, foreach_2d, foreach_tiled, kernel, unmasked, SpmdValue,
};

pub use rustlane_core::*;
