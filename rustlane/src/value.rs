//! `SpmdValue` / `SpmdGather`: the foundation for `#[derive(SpmdValue)]`
//! varying structs. A scalar "value type" `S` (a `#[repr(C)]` struct
//! of numeric fields, or a primitive numeric leaf) has an associated SoA
//! varying representation `<S as SpmdValue>::Varying<N>` — for a struct this is
//! the derive-generated `VaryingS<N>` with field-wise varying fields; for a
//! primitive it is `Varying<S, N>`.
//!
//! The derive emits, per struct `S`:
//! - the `VaryingS<N>` SoA struct (each field recurses through `SpmdValue`; a
//!   `#[spmd(uniform)]` field stays scalar);
//! - `impl SpmdValue for S` — the `Varying<N>` association and `splat`
//!   (splat-from-uniform-`S`), available for every value struct;
//! - `impl SpmdGather for S` and an inherent `select`, ONLY when the struct has
//!   no `#[spmd(uniform)]` field (an "all-varying" struct). Gathering or
//!   blending a uniform field is ill-defined, so a struct with a uniform field
//!   simply lacks these — a compile error at the use site;
//! - `MaskedAssign` for `VaryingS<N>`: all four exec contexts for an
//!   all-varying struct; only `AllOn`/`BoolGuard` when a uniform field is
//!   present, so a whole-struct assignment under a varying mask is a
//!   missing-impl compile error (same rule as a uniform local).
//!
//! The AoS gather ([`SpmdGather::gather_fields`]) is built on
//! [`crate::memory::gather_field`]: indexing `&[S]` with a `Varying<i32, N>`
//! gathers each leaf primitive field with one strided gather (ISPC's AoS
//! behaviour), composing byte offsets through nested `SpmdValue` fields. A
//! downstream `SpmdRead<Varying<i32, N>, E> for [S]` impl is impossible (orphan
//! rule: `[S]` is not a local type) and a blanket one here collides with the
//! primitive gather impl in `memory.rs`; so the derive exposes the gather as an
//! inherent `VaryingS::<N>::gather(base, idx, exec)` rather than the `a[i]`
//! sugar.

use crate::memory::{gather_field, ActiveMask};
use crate::varying::Varying;
use core::simd::{LaneCount, SupportedLaneCount};

/// A scalar value type with an SoA varying representation. Implemented for the
/// primitive numeric leaves below and, via `#[derive(SpmdValue)]`, for every
/// `#[repr(C)]` struct of `SpmdValue` fields (including ones with
/// `#[spmd(uniform)]` fields).
pub trait SpmdValue: Copy {
    /// The SoA varying representation at lane count `N`.
    type Varying<const N: usize>: Copy
    where
        LaneCount<N>: SupportedLaneCount;

    /// Broadcast one uniform value to every lane (splat-from-uniform-`S`).
    fn splat<const N: usize>(self) -> Self::Varying<N>
    where
        LaneCount<N>: SupportedLaneCount;
}

/// A value type whose AoS layout can be gathered field-by-field. Implemented
/// for the primitive leaves and for `#[derive(SpmdValue)]` structs with NO
/// `#[spmd(uniform)]` field (a uniform field has no per-lane value to gather).
pub trait SpmdGather: SpmdValue {
    /// Gather this value out of an AoS slice of *outer* element type `Base`,
    /// where each `Self` starts at byte offset `field_offset` inside a `Base`
    /// element. Leaf fields become one [`gather_field`] each; nested
    /// `SpmdValue` fields recurse with a composed offset.
    ///
    /// # Safety
    /// `field_offset` must be the true byte offset of a `Self`-typed field
    /// within `Base` (the derive supplies `offset_of!` chains). Inactive and
    /// out-of-bounds lanes are never addressed (see [`gather_field`]).
    unsafe fn gather_fields<Base, const N: usize, E>(
        base: &[Base],
        idx: Varying<i32, N>,
        field_offset: usize,
        exec: E,
    ) -> Self::Varying<N>
    where
        E: ActiveMask<N> + Copy,
        LaneCount<N>: SupportedLaneCount;
}

macro_rules! impl_prim_spmd_value {
    ($($t:ty),* $(,)?) => { $(
        impl SpmdValue for $t {
            type Varying<const N: usize> = Varying<$t, N>
            where
                LaneCount<N>: SupportedLaneCount;

            #[inline(always)]
            fn splat<const N: usize>(self) -> Varying<$t, N>
            where
                LaneCount<N>: SupportedLaneCount,
            {
                Varying::splat(self)
            }
        }

        impl SpmdGather for $t {
            #[inline(always)]
            unsafe fn gather_fields<Base, const N: usize, E>(
                base: &[Base],
                idx: Varying<i32, N>,
                field_offset: usize,
                exec: E,
            ) -> Varying<$t, N>
            where
                E: ActiveMask<N> + Copy,
                LaneCount<N>: SupportedLaneCount,
            {
                // SAFETY: forwarded from the caller's `field_offset` contract;
                // `$t: SimdElement + Default` holds for every primitive here.
                unsafe { gather_field::<Base, $t, N, E>(base, idx, field_offset, exec) }
            }
        }
    )* };
}

impl_prim_spmd_value!(f32, f64, i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);
