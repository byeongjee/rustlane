
use crate::varying::Varying;
use core::ops::{Add, Sub};
use core::simd::num::{SimdFloat, SimdInt, SimdUint};
use core::simd::{LaneCount, Mask, Simd, SimdElement, SupportedLaneCount};


pub trait ReduceElem: SimdElement {
    const ADD_IDENT: Self;
    const MIN_IDENT: Self;
    const MAX_IDENT: Self;

    fn vreduce_add<const N: usize>(v: Simd<Self, N>) -> Self
    where
        LaneCount<N>: SupportedLaneCount;
    fn vreduce_min<const N: usize>(v: Simd<Self, N>) -> Self
    where
        LaneCount<N>: SupportedLaneCount;
    fn vreduce_max<const N: usize>(v: Simd<Self, N>) -> Self
    where
        LaneCount<N>: SupportedLaneCount;
}

macro_rules! impl_reduce_float {
    ($cat:path : $($t:ty),* $(,)?) => { $(
        impl ReduceElem for $t {
            const ADD_IDENT: $t = 0.0;
            const MIN_IDENT: $t = <$t>::INFINITY;
            const MAX_IDENT: $t = <$t>::NEG_INFINITY;
            #[inline(always)]
            fn vreduce_add<const N: usize>(v: Simd<$t, N>) -> $t
            where LaneCount<N>: SupportedLaneCount { <Simd<$t, N> as $cat>::reduce_sum(v) }
            #[inline(always)]
            fn vreduce_min<const N: usize>(v: Simd<$t, N>) -> $t
            where LaneCount<N>: SupportedLaneCount { <Simd<$t, N> as $cat>::reduce_min(v) }
            #[inline(always)]
            fn vreduce_max<const N: usize>(v: Simd<$t, N>) -> $t
            where LaneCount<N>: SupportedLaneCount { <Simd<$t, N> as $cat>::reduce_max(v) }
        }
    )* };
}

macro_rules! impl_reduce_int {
    ($cat:path : $($t:ty),* $(,)?) => { $(
        impl ReduceElem for $t {
            const ADD_IDENT: $t = 0;
            const MIN_IDENT: $t = <$t>::MAX;
            const MAX_IDENT: $t = <$t>::MIN;
            #[inline(always)]
            fn vreduce_add<const N: usize>(v: Simd<$t, N>) -> $t
            where LaneCount<N>: SupportedLaneCount { <Simd<$t, N> as $cat>::reduce_sum(v) }
            #[inline(always)]
            fn vreduce_min<const N: usize>(v: Simd<$t, N>) -> $t
            where LaneCount<N>: SupportedLaneCount { <Simd<$t, N> as $cat>::reduce_min(v) }
            #[inline(always)]
            fn vreduce_max<const N: usize>(v: Simd<$t, N>) -> $t
            where LaneCount<N>: SupportedLaneCount { <Simd<$t, N> as $cat>::reduce_max(v) }
        }
    )* };
}

impl_reduce_float!(SimdFloat: f32, f64);
impl_reduce_int!(SimdInt: i8, i16, i32, i64, isize);
impl_reduce_int!(SimdUint: u8, u16, u32, u64, usize);


#[inline(always)]
pub fn reduce_add<T: ReduceElem, const N: usize>(v: Varying<T, N>) -> T
where
    LaneCount<N>: SupportedLaneCount,
{
    T::vreduce_add(v.0)
}

#[inline(always)]
pub fn reduce_min<T: ReduceElem, const N: usize>(v: Varying<T, N>) -> T
where
    LaneCount<N>: SupportedLaneCount,
{
    T::vreduce_min(v.0)
}

#[inline(always)]
pub fn reduce_max<T: ReduceElem, const N: usize>(v: Varying<T, N>) -> T
where
    LaneCount<N>: SupportedLaneCount,
{
    T::vreduce_max(v.0)
}

#[inline(always)]
pub fn reduce_add_masked<T: ReduceElem, const N: usize>(v: Varying<T, N>, mask: Mask<i32, N>) -> T
where
    LaneCount<N>: SupportedLaneCount,
{
    let sel = mask.cast::<T::Mask>().select(v.0, Simd::splat(T::ADD_IDENT));
    T::vreduce_add(sel)
}

#[inline(always)]
pub fn reduce_min_masked<T: ReduceElem, const N: usize>(v: Varying<T, N>, mask: Mask<i32, N>) -> T
where
    LaneCount<N>: SupportedLaneCount,
{
    let sel = mask.cast::<T::Mask>().select(v.0, Simd::splat(T::MIN_IDENT));
    T::vreduce_min(sel)
}

#[inline(always)]
pub fn reduce_max_masked<T: ReduceElem, const N: usize>(v: Varying<T, N>, mask: Mask<i32, N>) -> T
where
    LaneCount<N>: SupportedLaneCount,
{
    let sel = mask.cast::<T::Mask>().select(v.0, Simd::splat(T::MAX_IDENT));
    T::vreduce_max(sel)
}


#[inline(always)]
pub fn any<const N: usize>(mask: Mask<i32, N>) -> bool
where
    LaneCount<N>: SupportedLaneCount,
{
    mask.any()
}

#[inline(always)]
pub fn all<const N: usize>(mask: Mask<i32, N>) -> bool
where
    LaneCount<N>: SupportedLaneCount,
{
    mask.all()
}

#[inline(always)]
pub fn none<const N: usize>(mask: Mask<i32, N>) -> bool
where
    LaneCount<N>: SupportedLaneCount,
{
    !mask.any()
}


#[inline(always)]
pub fn lanes_iota<const N: usize>() -> Varying<i32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    Varying(Simd::from_array(core::array::from_fn(|i| i as i32)))
}

#[inline(always)]
pub fn broadcast<T: SimdElement, const N: usize>(v: Varying<T, N>, lane: usize) -> Varying<T, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    Varying::splat(v.to_array()[lane])
}

#[inline(always)]
pub fn extract<T: SimdElement, const N: usize>(v: Varying<T, N>, lane: usize) -> T
where
    LaneCount<N>: SupportedLaneCount,
{
    v.to_array()[lane]
}

#[inline(always)]
pub fn insert<T: SimdElement, const N: usize>(
    v: Varying<T, N>,
    lane: usize,
    value: T,
) -> Varying<T, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let mut a = v.to_array();
    a[lane] = value;
    Varying::from_array(a)
}

#[inline(always)]
pub fn rotate<T: SimdElement, const N: usize>(v: Varying<T, N>, k: i32) -> Varying<T, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let a = v.to_array();
    let n = N as i32;
    Varying::from_array(core::array::from_fn(|i| {
        let src = (i as i32 + k).rem_euclid(n);
        a[src as usize]
    }))
}

#[inline(always)]
pub fn shift<T: SimdElement + Default, const N: usize>(v: Varying<T, N>, k: i32) -> Varying<T, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let a = v.to_array();
    let n = N as i32;
    Varying::from_array(core::array::from_fn(|i| {
        let src = i as i32 + k;
        if (0..n).contains(&src) {
            a[src as usize]
        } else {
            T::default()
        }
    }))
}

#[inline(always)]
pub fn exclusive_scan_add<T, const N: usize>(v: Varying<T, N>) -> Varying<T, N>
where
    T: ReduceElem,
    LaneCount<N>: SupportedLaneCount,
    Simd<T, N>: Add<Output = Simd<T, N>> + Sub<Output = Simd<T, N>>,
{
    let orig = v.0;
    let mut x = orig;
    let mut d = 1usize;
    while d < N {
        let a = x.to_array();
        let shifted: Simd<T, N> =
            Simd::from_array(core::array::from_fn(|i| if i >= d { a[i - d] } else { T::ADD_IDENT }));
        x = x + shifted;
        d <<= 1;
    }
    Varying(x - orig)
}


#[inline]
pub fn packed_store_active<T: SimdElement, const N: usize>(
    mask: Mask<i32, N>,
    dst: &mut [T],
    values: Varying<T, N>,
) -> usize
where
    LaneCount<N>: SupportedLaneCount,
{
    debug_assert!(
        dst.len() >= N,
        "packed_store_active: dst needs room for at least N lanes"
    );
    let vals = values.to_array();
    let mut count = 0usize;
    for i in 0..N {
        dst[count] = vals[i];
        count += mask.test(i) as usize;
    }
    count
}

#[inline]
pub fn packed_load_active<T: SimdElement, const N: usize>(
    mask: Mask<i32, N>,
    src: &[T],
    out: &mut Varying<T, N>,
) -> usize
where
    LaneCount<N>: SupportedLaneCount,
{
    let mut a = out.to_array();
    let mut count = 0usize;
    for i in 0..N {
        if mask.test(i) {
            a[i] = src[count];
            count += 1;
        }
    }
    *out = Varying::from_array(a);
    count
}


#[cfg(test)]
mod tests {
    use super::*;

    const N: usize = 8;
    type Vf = Varying<f32, N>;
    type Vi = Varying<i32, N>;
    type M = Mask<i32, N>;

    fn mask(bits: [bool; N]) -> M {
        Mask::from_array(bits)
    }

    #[test]
    fn reductions_over_all_lanes() {
        let v = Vi::from_array([1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(reduce_add(v), 36);
        assert_eq!(reduce_min(v), 1);
        assert_eq!(reduce_max(v), 8);

        let f = Vf::from_array([1.5, -2.0, 3.0, 0.0, 10.0, -1.0, 4.0, 2.5]);
        assert_eq!(reduce_add(f), 18.0);
        assert_eq!(reduce_min(f), -2.0);
        assert_eq!(reduce_max(f), 10.0);

        let u = Varying::<u32, N>::from_array([10, 20, 30, 40, 50, 60, 70, 80]);
        assert_eq!(reduce_add(u), 360);
        assert_eq!(reduce_min(u), 10);
        assert_eq!(reduce_max(u), 80);
    }

    #[test]
    fn masked_reductions_use_identity_for_inactive() {
        let v = Vi::from_array([1, 2, 3, 4, 5, 6, 7, 8]);
        let m = mask([true, false, true, false, true, false, true, false]);
        assert_eq!(reduce_add_masked(v, m), 1 + 3 + 5 + 7);
        assert_eq!(reduce_min_masked(v, m), 1);
        assert_eq!(reduce_max_masked(v, m), 7);

        let off = mask([false; N]);
        assert_eq!(reduce_add_masked(v, off), 0);
        assert_eq!(reduce_min_masked(v, off), i32::MAX);
        assert_eq!(reduce_max_masked(v, off), i32::MIN);

        let f = Vf::from_array([100.0, 1.0, -50.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let fm = mask([false, true, false, true, true, true, true, true]);
        assert_eq!(reduce_min_masked(f, fm), 1.0);
        assert_eq!(reduce_max_masked(f, fm), 6.0);
        assert_eq!(reduce_add_masked(f, fm), 1.0 + 2.0 + 3.0 + 4.0 + 5.0 + 6.0);
    }

    #[test]
    fn mask_predicates() {
        assert!(any(mask([false, false, true, false, false, false, false, false])));
        assert!(!any(mask([false; N])));
        assert!(all(mask([true; N])));
        assert!(!all(mask([true, true, true, true, true, true, true, false])));
        assert!(none(mask([false; N])));
        assert!(!none(mask([false, false, false, false, false, false, false, true])));
    }

    #[test]
    fn lanes_iota_is_program_index() {
        assert_eq!(lanes_iota::<N>().to_array(), [0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(lanes_iota::<4>().to_array(), [0, 1, 2, 3]);
    }

    #[test]
    fn broadcast_extract_insert() {
        let v = Vi::from_array([10, 11, 12, 13, 14, 15, 16, 17]);
        assert_eq!(broadcast(v, 3).to_array(), [13; N]);
        assert_eq!(extract(v, 0), 10);
        assert_eq!(extract(v, 7), 17);
        assert_eq!(
            insert(v, 2, 99).to_array(),
            [10, 11, 99, 13, 14, 15, 16, 17]
        );
    }

    #[test]
    fn rotate_is_cyclic() {
        let v = Vi::from_array([0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(rotate(v, 1).to_array(), [1, 2, 3, 4, 5, 6, 7, 0]);
        assert_eq!(rotate(v, -1).to_array(), [7, 0, 1, 2, 3, 4, 5, 6]);
        assert_eq!(rotate(v, 8).to_array(), v.to_array());
        assert_eq!(rotate(v, 3).to_array(), [3, 4, 5, 6, 7, 0, 1, 2]);
    }

    #[test]
    fn shift_zero_fills() {
        let v = Vi::from_array([1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(shift(v, 2).to_array(), [3, 4, 5, 6, 7, 8, 0, 0]);
        assert_eq!(shift(v, -2).to_array(), [0, 0, 1, 2, 3, 4, 5, 6]);
        assert_eq!(shift(v, 8).to_array(), [0; N]);
        assert_eq!(shift(v, 0).to_array(), v.to_array());
    }

    #[test]
    fn exclusive_scan_add_prefix_sums() {
        let v = Vi::from_array([1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(
            exclusive_scan_add(v).to_array(),
            [0, 1, 3, 6, 10, 15, 21, 28]
        );
        let ones = Vi::splat(1);
        assert_eq!(exclusive_scan_add(ones).to_array(), [0, 1, 2, 3, 4, 5, 6, 7]);

        let f = Vf::from_array([2.0, -1.0, 3.5, 0.5, 4.0, 1.0, -2.0, 6.0]);
        let scan = exclusive_scan_add(f).to_array();
        let src = f.to_array();
        let mut acc = 0.0f32;
        for i in 0..N {
            assert_eq!(scan[i], acc, "exclusive scan lane {i}");
            acc += src[i];
        }

        let w = Varying::<i32, 4>::from_array([5, 10, 15, 20]);
        assert_eq!(exclusive_scan_add(w).to_array(), [0, 5, 15, 30]);
    }

    #[test]
    fn packed_store_active_compacts() {
        let vals = Vi::from_array([10, 20, 30, 40, 50, 60, 70, 80]);
        let m = mask([true, false, true, false, true, false, true, false]);
        let mut dst = [0i32; N];
        let count = packed_store_active(m, &mut dst, vals);
        assert_eq!(count, 4);
        assert_eq!(&dst[..count], &[10, 30, 50, 70]);

        let mut dst2 = [0i32; N];
        let c2 = packed_store_active(mask([true; N]), &mut dst2, vals);
        assert_eq!(c2, N);
        assert_eq!(dst2, [10, 20, 30, 40, 50, 60, 70, 80]);

        let mut dst3 = [-1i32; N];
        let c3 = packed_store_active(mask([false; N]), &mut dst3, vals);
        assert_eq!(c3, 0);
    }

    #[test]
    fn packed_load_active_scatters_into_active_lanes() {
        let src = [100i32, 200, 300, 400];
        let m = mask([true, false, true, false, true, false, true, false]);
        let mut out = Vi::splat(-1);
        let count = packed_load_active(m, &src, &mut out);
        assert_eq!(count, 4);
        assert_eq!(out.to_array(), [100, -1, 200, -1, 300, -1, 400, -1]);
    }

    #[test]
    fn packed_store_then_load_roundtrip() {
        let vals = Vi::from_array([7, 8, 9, 10, 11, 12, 13, 14]);
        let m = mask([false, true, true, false, true, false, false, true]);
        let mut buf = [0i32; N];
        let stored = packed_store_active(m, &mut buf, vals);
        let mut back = Vi::splat(0);
        let loaded = packed_load_active(m, &buf[..stored], &mut back);
        assert_eq!(stored, loaded);
        let orig = vals.to_array();
        let got = back.to_array();
        for i in 0..N {
            if m.test(i) {
                assert_eq!(got[i], orig[i], "lane {i}");
            }
        }
    }
}
