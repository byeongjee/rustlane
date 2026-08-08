//! `rustlane` — an ISPC-style SPMD programming model over portable SIMD.
//!
//! This is the single crate users depend on. It provides the runtime types and
//! traits that kernel bodies lower onto, and re-exports the procedural macros
//! implemented by the companion `rustlane-macros` crate:
//!
//! * the [`Varying`] value type, execution-context types and mask-stack
//!   machinery, condition/memory traits, and the [`prelude`];
//! * the [`macro@kernel`] and
//!   [`macro@export`] attributes, the [`macro@SpmdValue`] derive, and the
//!   kernel-world function-like macros ([`foreach!`](macro@foreach),
//!   [`foreach_2d!`](macro@foreach_2d), [`foreach_tiled!`](macro@foreach_tiled),
//!   [`unmasked!`](macro@unmasked), [`cif!`](macro@cif),
//!   [`cwhile!`](macro@cwhile)).
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

pub mod cond;
pub mod exec;
pub mod math;
pub mod memory;
pub mod reduce;
pub mod rng;
pub mod value;
pub mod varying;

pub use cond::{SpmdAnd, SpmdEq, SpmdNot, SpmdOr, SpmdOrd};
pub use exec::{
    AllOn, AndCond, BoolGuard, EnterLoop, EnterLoopN, Exec, LoopCond, LoopRemove, MaskedAssign,
    Refresh, SpmdLoop, UniformLoop, VMask, VMaskGuard, VaryingLoop,
};
pub use memory::{ActiveMask, LinearIndex, SpmdRead, SpmdWrite};
pub use value::{SpmdGather, SpmdValue};
pub use varying::{SpmdCast, SpmdCastElement, Varying, NATIVE_LANES};

pub use rustlane_macros::{
    cif, cwhile, export, foreach, foreach_2d, foreach_tiled, kernel, unmasked, SpmdValue,
};

/// Everything a kernel body (hand-expanded or macro-emitted) needs in scope.
///
/// Control-flow / operator machinery and the memory traits are re-exported
/// flat so the macro-emitted method calls (`a.spmd_read(i, __exec)`,
/// `a.spmd_write(i, __exec, v)`) and `foreach` lowering
/// (`LinearIndex::<N>::new(__base)`) resolve against a `use prelude::*` glob.
/// The stdlib namespaces (`math`, `reduce`, `rng`) and the full `memory` module
/// are re-exported as modules so a kernel body can call `math::exp(x)`,
/// `reduce::reduce_add(v)`, `rng::RNGState`, or `memory::gather_field(..)`
/// qualified without a second `use`; their generically-named free functions
/// are intentionally not flattened.
pub mod prelude {
    pub use crate::cond::{SpmdAnd, SpmdEq, SpmdNot, SpmdOr, SpmdOrd};
    pub use crate::exec::{
        AllOn, AndCond, BoolGuard, EnterLoop, EnterLoopN, Exec, LoopCond, LoopRemove, MaskedAssign,
        Refresh, SpmdLoop, UniformLoop, VMask, VMaskGuard, VaryingLoop,
    };
    pub use crate::memory::{ActiveMask, LinearIndex, SpmdRead, SpmdWrite};
    pub use crate::value::{SpmdGather, SpmdValue};
    pub use crate::varying::{SpmdCast, SpmdCastElement, Varying, NATIVE_LANES};
    pub use crate::{math, memory, reduce, rng};
}
