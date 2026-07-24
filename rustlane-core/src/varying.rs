//! `Varying<T, N>`: the SPMD varying value type (one value per program
//! instance / SIMD lane), plus its operator surface and masked-assignment
//! impls. The `#[kernel]` macro rewrites the surface type `Varying<T>` to
//! `Varying<T, N>` with its own const `N`; everything here is fully
//! const-generic so that rewrite is purely syntactic.

use crate::exec::{AllOn, BoolGuard, MaskedAssign, VMask, VMaskGuard};
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div,
    DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub,
    SubAssign,
};
use core::simd::{LaneCount, Mask, Simd, SimdCast, SimdElement, SupportedLaneCount};

/// Default lane count for the current build target (8 on aarch64, where
/// `Simd<f32, 8>` maps to paired, double-pumped NEON q-registers; 8 is also
/// the x86 baseline choice). Runtime-dispatch shims pick per-target widths
/// later; this is the width a width-agnostic caller should instantiate.
pub const NATIVE_LANES: usize = 8;

/// A varying value: `T` per lane, `N` lanes. Thin newtype over
/// `std::simd::Simd` — the inner vector is `pub` on purpose (the runtime and
/// hand-expanded kernels reach through it; the macro never does).
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(transparent)]
pub struct Varying<T, const N: usize>(pub Simd<T, N>)
where
    T: SimdElement,
    LaneCount<N>: SupportedLaneCount;

impl<T: SimdElement, const N: usize> Varying<T, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    /// Broadcast a uniform value to all lanes.
    #[inline(always)]
    pub fn splat(v: T) -> Self {
        Self(Simd::splat(v))
    }

    /// Per-lane values from an array.
    #[inline(always)]
    pub fn from_array(a: [T; N]) -> Self {
        Self(Simd::from_array(a))
    }

    /// Wrap an existing `std::simd` vector.
    #[inline(always)]
    pub fn from_simd(s: Simd<T, N>) -> Self {
        Self(s)
    }

    /// Per-lane values as an array.
    #[inline(always)]
    pub fn to_array(self) -> [T; N] {
        self.0.to_array()
    }

    /// The lane count `N` (ISPC `programCount`).
    #[inline(always)]
    pub const fn lanes() -> usize {
        N
    }

    /// Lane-wise select: `self` where `mask` is set, `other` elsewhere.
    /// The canonical condition currency is `Mask<i32, N>`; it is cast to the
    /// element's native mask width (free for 32-bit elements).
    #[inline(always)]
    pub fn select(self, mask: Mask<i32, N>, other: Self) -> Self {
        Self(mask.cast::<T::Mask>().select(self.0, other.0))
    }
}

/// Per-element-type cast dispatch. `std::simd` exposes `cast` on the three
/// category traits (`SimdFloat`/`SimdInt`/`SimdUint`), not on `Simd` itself,
/// so a generic `Varying::cast` needs this shim. Implemented for exactly the
/// primitive numeric element types; treat as sealed.
pub trait SpmdCastElement: SimdElement + SimdCast {
    fn cast_simd<U: SimdCast, const N: usize>(v: Simd<Self, N>) -> Simd<U, N>
    where
        LaneCount<N>: SupportedLaneCount;
}

macro_rules! impl_cast_element {
    ($cat:path : $($t:ty),* $(,)?) => { $(
        impl SpmdCastElement for $t {
            #[inline(always)]
            fn cast_simd<U: SimdCast, const N: usize>(v: Simd<$t, N>) -> Simd<U, N>
            where
                LaneCount<N>: SupportedLaneCount,
            {
                <Simd<$t, N> as $cat>::cast::<U>(v)
            }
        }
    )* };
}

impl_cast_element!(core::simd::num::SimdFloat: f32, f64);
impl_cast_element!(core::simd::num::SimdInt: i8, i16, i32, i64, isize);
impl_cast_element!(core::simd::num::SimdUint: u8, u16, u32, u64, usize);

impl<T: SpmdCastElement, const N: usize> Varying<T, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    /// Lane-wise numeric cast (`as`-semantics), via [`SpmdCast`].
    #[inline(always)]
    pub fn cast<U: SimdElement + SimdCast>(self) -> Varying<U, N> {
        Varying(T::cast_simd::<U, N>(self.0))
    }
}

impl<T: SimdElement + Default, const N: usize> Default for Varying<T, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline(always)]
    fn default() -> Self {
        Self::splat(T::default())
    }
}

macro_rules! impl_varying_binop {
    ($($op:ident, $fn:ident, $opas:ident, $fnas:ident);* $(;)?) => { $(
        impl<T: SimdElement, const N: usize> $op for Varying<T, N>
        where
            LaneCount<N>: SupportedLaneCount,
            Simd<T, N>: $op<Output = Simd<T, N>>,
        {
            type Output = Varying<T, N>;
            #[inline(always)]
            fn $fn(self, rhs: Self) -> Varying<T, N> {
                Varying($op::$fn(self.0, rhs.0))
            }
        }

        impl<T: SimdElement, const N: usize> $op<T> for Varying<T, N>
        where
            LaneCount<N>: SupportedLaneCount,
            Simd<T, N>: $op<Output = Simd<T, N>>,
        {
            type Output = Varying<T, N>;
            #[inline(always)]
            fn $fn(self, rhs: T) -> Varying<T, N> {
                Varying($op::$fn(self.0, Simd::splat(rhs)))
            }
        }

        impl<T: SimdElement, const N: usize> $opas for Varying<T, N>
        where
            LaneCount<N>: SupportedLaneCount,
            Simd<T, N>: $op<Output = Simd<T, N>>,
        {
            #[inline(always)]
            fn $fnas(&mut self, rhs: Self) {
                self.0 = $op::$fn(self.0, rhs.0);
            }
        }

        impl<T: SimdElement, const N: usize> $opas<T> for Varying<T, N>
        where
            LaneCount<N>: SupportedLaneCount,
            Simd<T, N>: $op<Output = Simd<T, N>>,
        {
            #[inline(always)]
            fn $fnas(&mut self, rhs: T) {
                self.0 = $op::$fn(self.0, Simd::splat(rhs));
            }
        }
    )* };
}

impl_varying_binop!(
    Add, add, AddAssign, add_assign;
    Sub, sub, SubAssign, sub_assign;
    Mul, mul, MulAssign, mul_assign;
    Div, div, DivAssign, div_assign;
    Rem, rem, RemAssign, rem_assign;
    BitAnd, bitand, BitAndAssign, bitand_assign;
    BitOr, bitor, BitOrAssign, bitor_assign;
    BitXor, bitxor, BitXorAssign, bitxor_assign;
    Shl, shl, ShlAssign, shl_assign;
    Shr, shr, ShrAssign, shr_assign;
);

impl<T: SimdElement, const N: usize> Neg for Varying<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    Simd<T, N>: Neg<Output = Simd<T, N>>,
{
    type Output = Varying<T, N>;
    #[inline(always)]
    fn neg(self) -> Varying<T, N> {
        Varying(-self.0)
    }
}

macro_rules! impl_scalar_lhs_op {
    ($t:ty, $op:ident, $fn:ident) => {
        impl<const N: usize> $op<Varying<$t, N>> for $t
        where
            LaneCount<N>: SupportedLaneCount,
            Simd<$t, N>: $op<Output = Simd<$t, N>>,
        {
            type Output = Varying<$t, N>;
            #[inline(always)]
            fn $fn(self, rhs: Varying<$t, N>) -> Varying<$t, N> {
                Varying($op::$fn(Simd::splat(self), rhs.0))
            }
        }
    };
}

macro_rules! impl_scalar_lhs_ops {
    ($($t:ty),* $(,)?) => { $(
        impl_scalar_lhs_op!($t, Add, add);
        impl_scalar_lhs_op!($t, Sub, sub);
        impl_scalar_lhs_op!($t, Mul, mul);
        impl_scalar_lhs_op!($t, Div, div);
        impl_scalar_lhs_op!($t, Rem, rem);
        impl_scalar_lhs_op!($t, BitAnd, bitand);
        impl_scalar_lhs_op!($t, BitOr, bitor);
        impl_scalar_lhs_op!($t, BitXor, bitxor);
        impl_scalar_lhs_op!($t, Shl, shl);
        impl_scalar_lhs_op!($t, Shr, shr);
    )* };
}

impl_scalar_lhs_ops!(f32, f64, i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);

/// Rewrite target for `as` casts: the macro emits
/// `rustlane::SpmdCast::<U>::spmd_cast(expr)` where `U` is the ELEMENT target
/// type token from the source (`x as f32`). Uniform inputs produce `U`;
/// varying inputs produce `Varying<U, N>` — the associated `Out` type
/// resolves it, so the macro stays type-blind.
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot be cast to element type `{Target}` in an rustlane kernel",
    label = "unsupported `as` cast",
    note = "rustlane casts support the primitive numeric types and `Varying` of them"
)]
pub trait SpmdCast<Target> {
    type Out;
    fn spmd_cast(self) -> Self::Out;
}

impl<T, U, const N: usize> SpmdCast<U> for Varying<T, N>
where
    T: SpmdCastElement,
    U: SimdElement + SimdCast,
    LaneCount<N>: SupportedLaneCount,
{
    type Out = Varying<U, N>;
    #[inline(always)]
    fn spmd_cast(self) -> Varying<U, N> {
        Varying(T::cast_simd::<U, N>(self.0))
    }
}

macro_rules! impl_scalar_cast_to {
    ($from:ty => $($to:ty),* $(,)?) => { $(
        impl SpmdCast<$to> for $from {
            type Out = $to;
            #[inline(always)]
            fn spmd_cast(self) -> $to {
                self as $to
            }
        }
    )* };
}

macro_rules! impl_scalar_cast_from {
    ($($from:ty),* $(,)?) => { $(
        impl_scalar_cast_to!($from => i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, f32, f64);
    )* };
}

impl_scalar_cast_from!(f32, f64, i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);

impl<T: SimdElement, const N: usize> MaskedAssign<AllOn> for Varying<T, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline(always)]
    fn masked_assign(&mut self, _exec: AllOn, value: Self) {
        *self = value;
    }
}

impl<T: SimdElement, const N: usize> MaskedAssign<AllOn, T> for Varying<T, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline(always)]
    fn masked_assign(&mut self, _exec: AllOn, value: T) {
        self.0 = Simd::splat(value);
    }
}

impl<T: SimdElement, const N: usize> MaskedAssign<BoolGuard> for Varying<T, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline(always)]
    fn masked_assign(&mut self, _exec: BoolGuard, value: Self) {
        *self = value;
    }
}

impl<T: SimdElement, const N: usize> MaskedAssign<BoolGuard, T> for Varying<T, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline(always)]
    fn masked_assign(&mut self, _exec: BoolGuard, value: T) {
        self.0 = Simd::splat(value);
    }
}

impl<T: SimdElement, const N: usize> MaskedAssign<VMask<N>> for Varying<T, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline(always)]
    fn masked_assign(&mut self, exec: VMask<N>, value: Self) {
        self.0 = exec.0.cast::<T::Mask>().select(value.0, self.0);
    }
}

impl<T: SimdElement, const N: usize> MaskedAssign<VMask<N>, T> for Varying<T, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline(always)]
    fn masked_assign(&mut self, exec: VMask<N>, value: T) {
        self.0 = exec.0.cast::<T::Mask>().select(Simd::splat(value), self.0);
    }
}

impl<T: SimdElement, const N: usize> MaskedAssign<VMaskGuard<N>> for Varying<T, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline(always)]
    fn masked_assign(&mut self, exec: VMaskGuard<N>, value: Self) {
        let m = exec.0 & Mask::splat(exec.1);
        self.0 = m.cast::<T::Mask>().select(value.0, self.0);
    }
}

impl<T: SimdElement, const N: usize> MaskedAssign<VMaskGuard<N>, T> for Varying<T, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline(always)]
    fn masked_assign(&mut self, exec: VMaskGuard<N>, value: T) {
        let m = exec.0 & Mask::splat(exec.1);
        self.0 = m.cast::<T::Mask>().select(Simd::splat(value), self.0);
    }
}

impl<const N: usize> MaskedAssign<AllOn> for Mask<i32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline(always)]
    fn masked_assign(&mut self, _exec: AllOn, value: Self) {
        *self = value;
    }
}

impl<const N: usize> MaskedAssign<AllOn, bool> for Mask<i32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline(always)]
    fn masked_assign(&mut self, _exec: AllOn, value: bool) {
        *self = Mask::splat(value);
    }
}

impl<const N: usize> MaskedAssign<BoolGuard> for Mask<i32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline(always)]
    fn masked_assign(&mut self, _exec: BoolGuard, value: Self) {
        *self = value;
    }
}

impl<const N: usize> MaskedAssign<BoolGuard, bool> for Mask<i32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline(always)]
    fn masked_assign(&mut self, _exec: BoolGuard, value: bool) {
        *self = Mask::splat(value);
    }
}

impl<const N: usize> MaskedAssign<VMask<N>> for Mask<i32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline(always)]
    fn masked_assign(&mut self, exec: VMask<N>, value: Self) {
        *self = exec.0.select_mask(value, *self);
    }
}

impl<const N: usize> MaskedAssign<VMask<N>, bool> for Mask<i32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline(always)]
    fn masked_assign(&mut self, exec: VMask<N>, value: bool) {
        *self = exec.0.select_mask(Mask::splat(value), *self);
    }
}

impl<const N: usize> MaskedAssign<VMaskGuard<N>> for Mask<i32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline(always)]
    fn masked_assign(&mut self, exec: VMaskGuard<N>, value: Self) {
        let m = exec.0 & Mask::splat(exec.1);
        *self = m.select_mask(value, *self);
    }
}

impl<const N: usize> MaskedAssign<VMaskGuard<N>, bool> for Mask<i32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline(always)]
    fn masked_assign(&mut self, exec: VMaskGuard<N>, value: bool) {
        let m = exec.0 & Mask::splat(exec.1);
        *self = m.select_mask(Mask::splat(value), *self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: usize = 4;
    type Vf = Varying<f32, N>;
    type Vi = Varying<i32, N>;

    #[test]
    fn constructors_and_accessors() {
        assert_eq!(Vf::splat(1.5).to_array(), [1.5; N]);
        assert_eq!(Vi::from_array([1, 2, 3, 4]).to_array(), [1, 2, 3, 4]);
        assert_eq!(Vi::from_simd(Simd::splat(7)).to_array(), [7; N]);
        assert_eq!(Vf::lanes(), N);
        assert_eq!(Vi::default().to_array(), [0; N]);
        assert_eq!(NATIVE_LANES, 8);
    }

    #[test]
    fn arithmetic_all_four_combinations() {
        let v = Vf::from_array([1.0, 2.0, 3.0, 4.0]);
        let w = Vf::splat(2.0);
        assert_eq!((v + w).to_array(), [3.0, 4.0, 5.0, 6.0]);
        assert_eq!((v * 2.0).to_array(), [2.0, 4.0, 6.0, 8.0]);
        assert_eq!((2.0 * v).to_array(), [2.0, 4.0, 6.0, 8.0]);
        assert_eq!((10.0 - v).to_array(), [9.0, 8.0, 7.0, 6.0]);
        assert_eq!((v / 2.0).to_array(), [0.5, 1.0, 1.5, 2.0]);
        assert_eq!((-v).to_array(), [-1.0, -2.0, -3.0, -4.0]);

        let mut a = v;
        a += 1.0;
        assert_eq!(a.to_array(), [2.0, 3.0, 4.0, 5.0]);
        a -= Vf::splat(1.0);
        assert_eq!(a.to_array(), v.to_array());
        a *= 3.0;
        a /= Vf::splat(3.0);
        assert_eq!(a.to_array(), v.to_array());
    }

    #[test]
    fn integer_ops() {
        let v = Vi::from_array([1, 2, 3, 4]);
        assert_eq!((v % 2).to_array(), [1, 0, 1, 0]);
        assert_eq!((v & 1).to_array(), [1, 0, 1, 0]);
        assert_eq!((v | 4).to_array(), [5, 6, 7, 4]);
        assert_eq!((v ^ v).to_array(), [0; N]);
        assert_eq!((v << 1).to_array(), [2, 4, 6, 8]);
        assert_eq!((v >> 1).to_array(), [0, 1, 1, 2]);
        assert_eq!((1i32 << Vi::from_array([0, 1, 2, 3])).to_array(), [1, 2, 4, 8]);
        let mut a = v;
        a <<= 2;
        a >>= Vi::splat(2);
        assert_eq!(a.to_array(), v.to_array());
        a &= 6;
        assert_eq!(a.to_array(), [0, 2, 2, 4]);
    }

    #[test]
    fn select_semantics() {
        let m = Mask::<i32, N>::from_array([true, false, true, false]);
        let a = Vi::splat(1);
        let b = Vi::splat(2);
        assert_eq!(a.select(m, b).to_array(), [1, 2, 1, 2]);
        let x = Vf::splat(1.0);
        let y = Vf::splat(2.0);
        assert_eq!(x.select(m, y).to_array(), [1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn mask_lvalue_masked_assign() {
        type M = Mask<i32, N>;
        let exec_m = M::from_array([true, false, true, false]);

        let mut h: M = Default::default();
        h.masked_assign(AllOn, true);
        assert_eq!(h.to_array(), [true; N]);
        h.masked_assign(BoolGuard(true), M::from_array([true, true, false, false]));
        assert_eq!(h.to_array(), [true, true, false, false]);

        let mut h = M::splat(true);
        h.masked_assign(VMask(exec_m), false);
        assert_eq!(h.to_array(), [false, true, false, true]);

        let mut h = M::splat(false);
        h.masked_assign(VMask(exec_m), M::splat(true));
        assert_eq!(h.to_array(), [true, false, true, false]);

        let mut h = M::splat(false);
        h.masked_assign(VMaskGuard(exec_m, true), true);
        assert_eq!(h.to_array(), [true, false, true, false]);
    }

    #[test]
    fn casts() {
        let v = Vf::from_array([1.9, -2.9, 3.1, 4.0]);
        assert_eq!(v.cast::<i32>().to_array(), [1, -2, 3, 4]);
        let w: Varying<f32, N> = SpmdCast::<f32>::spmd_cast(Vi::from_array([1, 2, 3, 4]));
        assert_eq!(w.to_array(), [1.0, 2.0, 3.0, 4.0]);
        let u: i64 = SpmdCast::<i64>::spmd_cast(3.7f32);
        assert_eq!(u, 3);
        let f: f64 = SpmdCast::<f64>::spmd_cast(2u8);
        assert_eq!(f, 2.0);
    }
}
