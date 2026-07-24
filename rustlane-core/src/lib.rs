#![feature(portable_simd)]
//! rustlane-core: the runtime half of the `rustlane` SPMD-on-SIMD library.
//!
//! Provides the types and traits that `#[rustlane::kernel]` lowers onto:
//! [`Varying`] values, execution-context types ([`AllOn`], [`BoolGuard`],
//! [`VMask`], [`VMaskGuard`]) with the mask-stack loop machinery, and the
//! condition traits ([`SpmdOrd`], [`SpmdEq`], [`SpmdAnd`], [`SpmdOr`]).

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

/// Everything a kernel body (hand-expanded or macro-emitted) needs in scope.
///
/// Control-flow / operator machinery and the memory traits are re-exported
/// flat so the macro-emitted method calls (`a.spmd_read(i, __exec)`,
/// `a.spmd_write(i, __exec, v)`) and `foreach` lowering
/// (`LinearIndex::<N>::new(__base)`) resolve against a `use prelude::*` glob.
/// The stdlib namespaces (`math`, `reduce`, `rng`) and
/// the full `memory` module are re-exported as modules so a kernel body can
/// call `math::exp(x)`, `reduce::reduce_add(v)`, `rng::RNGState`, or
/// `memory::gather_field(..)` qualified without a second `use`; their
/// generically-named free functions are intentionally NOT flattened.
pub mod prelude {
    pub use crate::cond::{SpmdAnd, SpmdEq, SpmdNot, SpmdOr, SpmdOrd};
    pub use crate::exec::{
        AllOn, AndCond, BoolGuard, EnterLoop, EnterLoopN, Exec, LoopCond, LoopRemove,
        MaskedAssign, Refresh, SpmdLoop, UniformLoop, VMask, VMaskGuard, VaryingLoop,
    };
    pub use crate::memory::{ActiveMask, LinearIndex, SpmdRead, SpmdWrite};
    pub use crate::value::{SpmdGather, SpmdValue};
    pub use crate::varying::{SpmdCast, SpmdCastElement, Varying, NATIVE_LANES};
    pub use crate::{math, memory, reduce, rng};
}
