//! Condition traits: the rewrite targets for comparison and logical
//! operators inside kernels.
//!
//! Rust comparison operators must return `bool`, so `#[kernel]` rewrites
//! `< > <= >= == !=` into these trait calls. The associated `Cond` type
//! carries uniformity: `bool` for uniform comparisons, `Mask<i32, N>` (the
//! canonical condition currency) for varying ones. `&&`/`||` are lowered by
//! the macro into an evaluation of the right-hand side under the
//! lhs-narrowed execution context, followed by the eager
//! [`SpmdAnd`]/[`SpmdOr`] combine.

use crate::varying::Varying;
use core::simd::cmp::{SimdPartialEq, SimdPartialOrd};
use core::simd::{Mask, Simd, SimdElement};

/// Rewrite target for `< > <= >=`.
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot be order-compared with `{Rhs}` in an rustlane kernel",
    label = "no rustlane ordering between these types",
    note = "supported: uniform scalar vs uniform scalar (-> `bool`), and any mix of \
            `Varying<T, N>` and its scalar `T` (-> `Mask<i32, N>`)"
)]
pub trait SpmdOrd<Rhs = Self> {
    /// `bool` (uniform) or `Mask<i32, N>` (varying).
    type Cond;
    fn spmd_lt(self, rhs: Rhs) -> Self::Cond;
    fn spmd_le(self, rhs: Rhs) -> Self::Cond;
    fn spmd_gt(self, rhs: Rhs) -> Self::Cond;
    fn spmd_ge(self, rhs: Rhs) -> Self::Cond;
}

/// Rewrite target for `==` and `!=`.
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot be equality-compared with `{Rhs}` in an rustlane kernel",
    label = "no rustlane equality between these types",
    note = "supported: uniform scalar vs uniform scalar (-> `bool`), and any mix of \
            `Varying<T, N>` and its scalar `T` (-> `Mask<i32, N>`)"
)]
pub trait SpmdEq<Rhs = Self> {
    /// `bool` (uniform) or `Mask<i32, N>` (varying).
    type Cond;
    fn spmd_eq(self, rhs: Rhs) -> Self::Cond;
    fn spmd_ne(self, rhs: Rhs) -> Self::Cond;
}

macro_rules! impl_scalar_cmp {
    ($($t:ty),* $(,)?) => { $(
        impl SpmdOrd for $t {
            type Cond = bool;
            #[inline(always)]
            fn spmd_lt(self, rhs: $t) -> bool { self < rhs }
            #[inline(always)]
            fn spmd_le(self, rhs: $t) -> bool { self <= rhs }
            #[inline(always)]
            fn spmd_gt(self, rhs: $t) -> bool { self > rhs }
            #[inline(always)]
            fn spmd_ge(self, rhs: $t) -> bool { self >= rhs }
        }
        impl SpmdEq for $t {
            type Cond = bool;
            #[inline(always)]
            fn spmd_eq(self, rhs: $t) -> bool { self == rhs }
            #[inline(always)]
            fn spmd_ne(self, rhs: $t) -> bool { self != rhs }
        }
    )* };
}

impl_scalar_cmp!(f32, f64, i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);

impl SpmdEq for bool {
    type Cond = bool;
    #[inline(always)]
    fn spmd_eq(self, rhs: bool) -> bool {
        self == rhs
    }
    #[inline(always)]
    fn spmd_ne(self, rhs: bool) -> bool {
        self != rhs
    }
}

impl<T: SimdElement, const N: usize> SpmdOrd for Varying<T, N>
where
    Simd<T, N>: SimdPartialOrd + SimdPartialEq<Mask = Mask<T::Mask, N>>,
{
    type Cond = Mask<i32, N>;
    #[inline(always)]
    fn spmd_lt(self, rhs: Self) -> Mask<i32, N> {
        self.0.simd_lt(rhs.0).cast::<i32>()
    }
    #[inline(always)]
    fn spmd_le(self, rhs: Self) -> Mask<i32, N> {
        self.0.simd_le(rhs.0).cast::<i32>()
    }
    #[inline(always)]
    fn spmd_gt(self, rhs: Self) -> Mask<i32, N> {
        self.0.simd_gt(rhs.0).cast::<i32>()
    }
    #[inline(always)]
    fn spmd_ge(self, rhs: Self) -> Mask<i32, N> {
        self.0.simd_ge(rhs.0).cast::<i32>()
    }
}

impl<T: SimdElement, const N: usize> SpmdEq for Varying<T, N>
where
    Simd<T, N>: SimdPartialEq<Mask = Mask<T::Mask, N>>,
{
    type Cond = Mask<i32, N>;
    #[inline(always)]
    fn spmd_eq(self, rhs: Self) -> Mask<i32, N> {
        self.0.simd_eq(rhs.0).cast::<i32>()
    }
    #[inline(always)]
    fn spmd_ne(self, rhs: Self) -> Mask<i32, N> {
        self.0.simd_ne(rhs.0).cast::<i32>()
    }
}

impl<T: SimdElement, const N: usize> SpmdOrd<T> for Varying<T, N>
where
    Simd<T, N>: SimdPartialOrd + SimdPartialEq<Mask = Mask<T::Mask, N>>,
{
    type Cond = Mask<i32, N>;
    #[inline(always)]
    fn spmd_lt(self, rhs: T) -> Mask<i32, N> {
        self.0.simd_lt(Simd::splat(rhs)).cast::<i32>()
    }
    #[inline(always)]
    fn spmd_le(self, rhs: T) -> Mask<i32, N> {
        self.0.simd_le(Simd::splat(rhs)).cast::<i32>()
    }
    #[inline(always)]
    fn spmd_gt(self, rhs: T) -> Mask<i32, N> {
        self.0.simd_gt(Simd::splat(rhs)).cast::<i32>()
    }
    #[inline(always)]
    fn spmd_ge(self, rhs: T) -> Mask<i32, N> {
        self.0.simd_ge(Simd::splat(rhs)).cast::<i32>()
    }
}

impl<T: SimdElement, const N: usize> SpmdEq<T> for Varying<T, N>
where
    Simd<T, N>: SimdPartialEq<Mask = Mask<T::Mask, N>>,
{
    type Cond = Mask<i32, N>;
    #[inline(always)]
    fn spmd_eq(self, rhs: T) -> Mask<i32, N> {
        self.0.simd_eq(Simd::splat(rhs)).cast::<i32>()
    }
    #[inline(always)]
    fn spmd_ne(self, rhs: T) -> Mask<i32, N> {
        self.0.simd_ne(Simd::splat(rhs)).cast::<i32>()
    }
}

impl<T: SimdElement, const N: usize> SpmdOrd<Varying<T, N>> for T
where
    Simd<T, N>: SimdPartialOrd + SimdPartialEq<Mask = Mask<T::Mask, N>>,
{
    type Cond = Mask<i32, N>;
    #[inline(always)]
    fn spmd_lt(self, rhs: Varying<T, N>) -> Mask<i32, N> {
        Simd::splat(self).simd_lt(rhs.0).cast::<i32>()
    }
    #[inline(always)]
    fn spmd_le(self, rhs: Varying<T, N>) -> Mask<i32, N> {
        Simd::splat(self).simd_le(rhs.0).cast::<i32>()
    }
    #[inline(always)]
    fn spmd_gt(self, rhs: Varying<T, N>) -> Mask<i32, N> {
        Simd::splat(self).simd_gt(rhs.0).cast::<i32>()
    }
    #[inline(always)]
    fn spmd_ge(self, rhs: Varying<T, N>) -> Mask<i32, N> {
        Simd::splat(self).simd_ge(rhs.0).cast::<i32>()
    }
}

impl<T: SimdElement, const N: usize> SpmdEq<Varying<T, N>> for T
where
    Simd<T, N>: SimdPartialEq<Mask = Mask<T::Mask, N>>,
{
    type Cond = Mask<i32, N>;
    #[inline(always)]
    fn spmd_eq(self, rhs: Varying<T, N>) -> Mask<i32, N> {
        Simd::splat(self).simd_eq(rhs.0).cast::<i32>()
    }
    #[inline(always)]
    fn spmd_ne(self, rhs: Varying<T, N>) -> Mask<i32, N> {
        Simd::splat(self).simd_ne(rhs.0).cast::<i32>()
    }
}

/// Final combine of the `&&` lowering. `a && b` expands to
///
/// ```text
/// {
///     let __c1 = a;
///     let __exec1 = __exec.and_cond(__c1);
///     let __c2 = if __exec1.should_branch() { let __exec = __exec1; b }
///                else { Default::default() };
///     __c1.spmd_and(__c2)
/// }
/// ```
///
/// so the rhs is evaluated under the lhs-narrowed execution context: a
/// uniform lhs short-circuits with a real branch, and a varying lhs masks
/// the rhs's memory accesses and kernel calls lane-wise (`i < n && a[i] > 0`
/// never gathers a lane with `i >= n`). The trait itself is the eager
/// bool/mask AND at the end.
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot appear on the left of `&&` in an rustlane kernel",
    label = "expected `bool` or `Mask<i32, N>`"
)]
pub trait SpmdAnd<Rhs = Self> {
    type Out;
    fn spmd_and(self, rhs: Rhs) -> Self::Out;
}

/// Final combine of the `||` lowering; the rhs is evaluated under the
/// lhs-NEGATED narrowed context (`and_not_cond`). See [`SpmdAnd`].
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot appear on the left of `||` in an rustlane kernel",
    label = "expected `bool` or `Mask<i32, N>`"
)]
pub trait SpmdOr<Rhs = Self> {
    type Out;
    fn spmd_or(self, rhs: Rhs) -> Self::Out;
}

/// Rewrite target for unary `!` on conditions.
#[diagnostic::on_unimplemented(
    message = "`!` cannot be applied to `{Self}` in an rustlane kernel condition",
    label = "expected `bool` or `Mask<i32, N>`"
)]
pub trait SpmdNot {
    type Out;
    fn spmd_not(self) -> Self::Out;
}

impl SpmdAnd for bool {
    type Out = bool;
    #[inline(always)]
    fn spmd_and(self, rhs: bool) -> bool {
        self & rhs
    }
}

impl<const N: usize> SpmdAnd<Mask<i32, N>> for bool {
    type Out = Mask<i32, N>;
    #[inline(always)]
    fn spmd_and(self, rhs: Mask<i32, N>) -> Mask<i32, N> {
        Mask::splat(self) & rhs
    }
}

impl<const N: usize> SpmdAnd for Mask<i32, N> {
    type Out = Mask<i32, N>;
    #[inline(always)]
    fn spmd_and(self, rhs: Mask<i32, N>) -> Mask<i32, N> {
        self & rhs
    }
}

impl<const N: usize> SpmdAnd<bool> for Mask<i32, N> {
    type Out = Mask<i32, N>;
    #[inline(always)]
    fn spmd_and(self, rhs: bool) -> Mask<i32, N> {
        self & Mask::splat(rhs)
    }
}

impl SpmdOr for bool {
    type Out = bool;
    #[inline(always)]
    fn spmd_or(self, rhs: bool) -> bool {
        self | rhs
    }
}

impl<const N: usize> SpmdOr<Mask<i32, N>> for bool {
    type Out = Mask<i32, N>;
    #[inline(always)]
    fn spmd_or(self, rhs: Mask<i32, N>) -> Mask<i32, N> {
        Mask::splat(self) | rhs
    }
}

impl<const N: usize> SpmdOr for Mask<i32, N> {
    type Out = Mask<i32, N>;
    #[inline(always)]
    fn spmd_or(self, rhs: Mask<i32, N>) -> Mask<i32, N> {
        self | rhs
    }
}

impl<const N: usize> SpmdOr<bool> for Mask<i32, N> {
    type Out = Mask<i32, N>;
    #[inline(always)]
    fn spmd_or(self, rhs: bool) -> Mask<i32, N> {
        self | Mask::splat(rhs)
    }
}

impl SpmdNot for bool {
    type Out = bool;
    #[inline(always)]
    fn spmd_not(self) -> bool {
        !self
    }
}

impl<const N: usize> SpmdNot for Mask<i32, N> {
    type Out = Mask<i32, N>;
    #[inline(always)]
    fn spmd_not(self) -> Mask<i32, N> {
        !self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: usize = 4;
    type Vf = Varying<f32, N>;
    type Vi = Varying<i32, N>;
    type M = Mask<i32, N>;

    #[test]
    fn scalar_comparisons_are_bool() {
        assert!(1.0f32.spmd_lt(2.0));
        assert!(2i32.spmd_ge(2));
        assert!(3u8.spmd_ne(4));
        assert!(true.spmd_eq(true));
        assert!(!(5i64.spmd_gt(6)));
    }

    #[test]
    fn varying_comparisons_are_masks() {
        let v = Vi::from_array([1, 2, 3, 4]);
        let w = Vi::splat(3);
        let m: M = v.spmd_lt(w);
        assert_eq!(m.to_array(), [true, true, false, false]);
        let m: M = v.spmd_ge(2);
        assert_eq!(m.to_array(), [false, true, true, true]);
        let m: M = 3.spmd_le(v);
        assert_eq!(m.to_array(), [false, false, true, true]);
        let m: M = v.spmd_eq(2);
        assert_eq!(m.to_array(), [false, true, false, false]);
        let x = Vf::from_array([1.0, 2.5, 3.0, -1.0]);
        let m: M = x.spmd_gt(2.0);
        assert_eq!(m.to_array(), [false, true, true, false]);
        let m: M = x.spmd_ne(Vf::splat(3.0));
        assert_eq!(m.to_array(), [true, true, false, true]);
    }

    #[test]
    fn uniform_and_or() {
        assert!(true.spmd_and(true));
        assert!(!true.spmd_and(false));
        assert!(!false.spmd_and(true));
        assert!(true.spmd_or(false));
        assert!(false.spmd_or(true));
        assert!(!false.spmd_or(false));
    }

    #[test]
    fn mixed_and_or() {
        let m = M::from_array([true, false, true, false]);
        let r: M = false.spmd_and(m);
        assert_eq!(r.to_array(), [false; N]);
        let r: M = true.spmd_and(m);
        assert_eq!(r, m);
        let r: M = true.spmd_or(m);
        assert_eq!(r.to_array(), [true; N]);
        let n = M::from_array([true, true, false, false]);
        assert_eq!(m.spmd_and(n).to_array(), [true, false, false, false]);
        assert_eq!(m.spmd_or(n).to_array(), [true, true, true, false]);
        assert_eq!(m.spmd_and(true), m);
        assert_eq!(m.spmd_or(false), m);
    }

    #[test]
    fn spmd_not_forms() {
        assert!(false.spmd_not());
        let m = M::from_array([true, false, true, false]);
        assert_eq!(m.spmd_not().to_array(), [false, true, false, true]);
    }
}
