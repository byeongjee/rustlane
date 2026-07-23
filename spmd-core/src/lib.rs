#![feature(portable_simd)]

pub mod cond;
pub mod exec;
pub mod math;
pub mod memory;
pub mod reduce;
pub mod rng;
pub mod varying;

pub use cond::{SpmdAnd, SpmdEq, SpmdNot, SpmdOr, SpmdOrd};
pub use exec::{
    AllOn, AndCond, BoolGuard, EnterLoop, EnterLoopN, Exec, LoopCond, LoopRemove, MaskedAssign,
    Refresh, SpmdLoop, UniformLoop, VMask, VMaskGuard, VaryingLoop,
};
pub use memory::{ActiveMask, LinearIndex, SpmdRead, SpmdWrite};
pub use varying::{SpmdCast, SpmdCastElement, Varying, NATIVE_LANES};

pub mod prelude {
    pub use crate::cond::{SpmdAnd, SpmdEq, SpmdNot, SpmdOr, SpmdOrd};
    pub use crate::exec::{
        AllOn, AndCond, BoolGuard, EnterLoop, EnterLoopN, Exec, LoopCond, LoopRemove,
        MaskedAssign, Refresh, SpmdLoop, UniformLoop, VMask, VMaskGuard, VaryingLoop,
    };
    pub use crate::memory::{ActiveMask, LinearIndex, SpmdRead, SpmdWrite};
    pub use crate::varying::{SpmdCast, SpmdCastElement, Varying, NATIVE_LANES};
    pub use crate::{math, memory, reduce, rng};
}
