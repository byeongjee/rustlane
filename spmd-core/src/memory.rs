
use crate::exec::{AllOn, BoolGuard, Exec, VMask, VMaskGuard};
use crate::varying::Varying;
use core::ops::Add;
use core::simd::cmp::SimdPartialOrd;
use core::simd::num::{SimdInt, SimdUint};
use core::simd::ptr::SimdConstPtr;
use core::simd::{LaneCount, Mask, Simd, SimdElement, SupportedLaneCount};


#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct LinearIndex<const N: usize> {
    base: usize,
}

impl<const N: usize> LinearIndex<N> {
    #[inline(always)]
    pub fn new(base: usize) -> Self {
        Self { base }
    }

    #[inline(always)]
    pub fn base(self) -> usize {
        self.base
    }

    #[inline(always)]
    pub fn to_varying(self) -> Varying<i32, N>
    where
        LaneCount<N>: SupportedLaneCount,
    {
        let iota = Simd::<i32, N>::from_array(core::array::from_fn(|l| l as i32));
        Varying(Simd::splat(self.base as i32) + iota)
    }
}

impl<const N: usize> Add<i32> for LinearIndex<N> {
    type Output = LinearIndex<N>;
    #[inline(always)]
    fn add(self, rhs: i32) -> LinearIndex<N> {
        LinearIndex {
            base: self.base.wrapping_add(rhs as usize),
        }
    }
}

impl<const N: usize> Add<LinearIndex<N>> for i32 {
    type Output = LinearIndex<N>;
    #[inline(always)]
    fn add(self, rhs: LinearIndex<N>) -> LinearIndex<N> {
        LinearIndex {
            base: rhs.base.wrapping_add(self as usize),
        }
    }
}

impl<const N: usize> Add<Varying<i32, N>> for LinearIndex<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    type Output = Varying<i32, N>;
    #[inline(always)]
    fn add(self, rhs: Varying<i32, N>) -> Varying<i32, N> {
        self.to_varying() + rhs
    }
}

impl<const N: usize> Add<LinearIndex<N>> for Varying<i32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    type Output = Varying<i32, N>;
    #[inline(always)]
    fn add(self, rhs: LinearIndex<N>) -> Varying<i32, N> {
        self + rhs.to_varying()
    }
}

impl<const N: usize> From<LinearIndex<N>> for Varying<i32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline(always)]
    fn from(li: LinearIndex<N>) -> Varying<i32, N> {
        li.to_varying()
    }
}


pub trait ActiveMask<const N: usize>
where
    LaneCount<N>: SupportedLaneCount,
{
    fn active_mask(self) -> Mask<i32, N>;

    #[inline(always)]
    fn all_known_active(&self) -> bool {
        false
    }
}

impl<const N: usize> ActiveMask<N> for AllOn
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline(always)]
    fn active_mask(self) -> Mask<i32, N> {
        Mask::splat(true)
    }
    #[inline(always)]
    fn all_known_active(&self) -> bool {
        true
    }
}

impl<const N: usize> ActiveMask<N> for BoolGuard
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline(always)]
    fn active_mask(self) -> Mask<i32, N> {
        Mask::splat(self.0)
    }
    #[inline(always)]
    fn all_known_active(&self) -> bool {
        self.0
    }
}

impl<const N: usize> ActiveMask<N> for VMask<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline(always)]
    fn active_mask(self) -> Mask<i32, N> {
        self.0
    }
}

impl<const N: usize> ActiveMask<N> for VMaskGuard<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline(always)]
    fn active_mask(self) -> Mask<i32, N> {
        self.0 & Mask::splat(self.1)
    }
}


#[diagnostic::on_unimplemented(
    message = "`[{Self}]` cannot be spmd-indexed by `{Idx}` under context `{E}`",
    label = "unsupported (slice, index, exec) combination",
    note = "index kinds: `usize` (plain load), `LinearIndex<N>` (contiguous \
            load), `Varying<i32, N>` (gather)"
)]
pub trait SpmdRead<Idx, E> {
    type Out;
    fn spmd_read(&self, idx: Idx, exec: E) -> Self::Out;

    unsafe fn spmd_read_unchecked(&self, idx: Idx, exec: E) -> Self::Out;
}

#[diagnostic::on_unimplemented(
    message = "`[{Self}]` cannot be spmd-written at `{Idx}` under context `{E}` with `{V}`",
    label = "unsupported (slice, index, exec, value) combination",
    note = "a `usize` (uniform) store is only allowed under uniform control \
            flow; under a varying mask use a `Varying<i32, N>` index (scatter) \
            or a `LinearIndex<N>` (contiguous store)"
)]
pub trait SpmdWrite<Idx, E, V> {
    fn spmd_write(&mut self, idx: Idx, exec: E, value: V);

    unsafe fn spmd_write_unchecked(&mut self, idx: Idx, exec: E, value: V);
}


impl<T: Copy, E: Exec> SpmdRead<usize, E> for [T] {
    type Out = T;
    #[inline(always)]
    fn spmd_read(&self, i: usize, _exec: E) -> T {
        self[i]
    }
    #[inline(always)]
    unsafe fn spmd_read_unchecked(&self, i: usize, _exec: E) -> T {
        unsafe { *self.get_unchecked(i) }
    }
}

macro_rules! impl_scalar_write {
    ($($E:ty),* $(,)?) => { $(
        impl<T: Copy> SpmdWrite<usize, $E, T> for [T] {
            #[inline(always)]
            fn spmd_write(&mut self, i: usize, _exec: $E, value: T) {
                self[i] = value;
            }
            #[inline(always)]
            unsafe fn spmd_write_unchecked(&mut self, i: usize, _exec: $E, value: T) {
                unsafe { *self.get_unchecked_mut(i) = value };
            }
        }
    )* };
}
impl_scalar_write!(AllOn, BoolGuard);

macro_rules! impl_varying_elem_write_masked {
    ($($E:ty),* $(,)?) => { $(
        impl<T: SimdElement, const N: usize> SpmdWrite<usize, $E, Varying<T, N>>
            for [Varying<T, N>]
        where
            LaneCount<N>: SupportedLaneCount,
        {
            #[inline(always)]
            fn spmd_write(&mut self, i: usize, exec: $E, value: Varying<T, N>) {
                crate::exec::MaskedAssign::masked_assign(&mut self[i], exec, value);
            }
            #[inline(always)]
            unsafe fn spmd_write_unchecked(&mut self, i: usize, exec: $E, value: Varying<T, N>) {
                let slot = unsafe { self.get_unchecked_mut(i) };
                crate::exec::MaskedAssign::masked_assign(slot, exec, value);
            }
        }
    )* };
}
impl_varying_elem_write_masked!(VMask<N>, VMaskGuard<N>);

macro_rules! impl_varying_elem_write_scalar_rhs {
    ($($E:ty),* $(,)?) => { $(
        impl<T: SimdElement, const N: usize> SpmdWrite<usize, $E, T> for [Varying<T, N>]
        where
            LaneCount<N>: SupportedLaneCount,
        {
            #[inline(always)]
            fn spmd_write(&mut self, i: usize, exec: $E, value: T) {
                crate::exec::MaskedAssign::masked_assign(&mut self[i], exec, value);
            }
            #[inline(always)]
            unsafe fn spmd_write_unchecked(&mut self, i: usize, exec: $E, value: T) {
                let slot = unsafe { self.get_unchecked_mut(i) };
                crate::exec::MaskedAssign::masked_assign(slot, exec, value);
            }
        }
    )* };
}
impl_varying_elem_write_scalar_rhs!(AllOn, BoolGuard, VMask<N>, VMaskGuard<N>);


macro_rules! impl_linear_plain {
    ($($E:ty),* $(,)?) => { $(
        impl<T: SimdElement, const N: usize> SpmdRead<LinearIndex<N>, $E> for [T]
        where
            LaneCount<N>: SupportedLaneCount,
        {
            type Out = Varying<T, N>;
            #[inline(always)]
            fn spmd_read(&self, idx: LinearIndex<N>, _exec: $E) -> Varying<T, N> {
                let b = idx.base;
                Varying(Simd::from_slice(&self[b..b + N]))
            }
            #[inline(always)]
            unsafe fn spmd_read_unchecked(&self, idx: LinearIndex<N>, _exec: $E) -> Varying<T, N> {
                let b = idx.base;
                Varying(Simd::from_slice(unsafe { self.get_unchecked(b..b + N) }))
            }
        }

        impl<T: SimdElement, const N: usize> SpmdWrite<LinearIndex<N>, $E, Varying<T, N>> for [T]
        where
            LaneCount<N>: SupportedLaneCount,
        {
            #[inline(always)]
            fn spmd_write(&mut self, idx: LinearIndex<N>, _exec: $E, value: Varying<T, N>) {
                let b = idx.base;
                value.0.copy_to_slice(&mut self[b..b + N]);
            }
            #[inline(always)]
            unsafe fn spmd_write_unchecked(
                &mut self,
                idx: LinearIndex<N>,
                _exec: $E,
                value: Varying<T, N>,
            ) {
                let b = idx.base;
                value.0.copy_to_slice(unsafe { self.get_unchecked_mut(b..b + N) });
            }
        }
    )* };
}
impl_linear_plain!(AllOn, BoolGuard);

macro_rules! impl_linear_masked {
    ($E:ty, |$e:ident| $mask:expr) => {
        impl<T: SimdElement + Default, const N: usize> SpmdRead<LinearIndex<N>, $E> for [T]
        where
            LaneCount<N>: SupportedLaneCount,
        {
            type Out = Varying<T, N>;
            #[inline(always)]
            fn spmd_read(&self, idx: LinearIndex<N>, exec: $E) -> Varying<T, N> {
                let b = idx.base.min(self.len());
                let m: Mask<i32, N> = {
                    let $e = exec;
                    $mask
                };
                debug_assert!(
                    linear_in_bounds::<N>(idx.base, self.len(), m),
                    "spmd_read: LinearIndex out of bounds on an active lane"
                );
                Varying(Simd::load_select(&self[b..], m.cast::<T::Mask>(), Simd::default()))
            }
            #[inline(always)]
            unsafe fn spmd_read_unchecked(&self, idx: LinearIndex<N>, exec: $E) -> Varying<T, N> {
                let b = idx.base.min(self.len());
                let m: Mask<i32, N> = {
                    let $e = exec;
                    $mask
                };
                Varying(unsafe {
                    Simd::load_select_unchecked(
                        self.get_unchecked(b..),
                        m.cast::<T::Mask>(),
                        Simd::default(),
                    )
                })
            }
        }

        impl<T: SimdElement, const N: usize> SpmdWrite<LinearIndex<N>, $E, Varying<T, N>> for [T]
        where
            LaneCount<N>: SupportedLaneCount,
        {
            #[inline(always)]
            fn spmd_write(&mut self, idx: LinearIndex<N>, exec: $E, value: Varying<T, N>) {
                let b = idx.base.min(self.len());
                let m: Mask<i32, N> = {
                    let $e = exec;
                    $mask
                };
                debug_assert!(
                    linear_in_bounds::<N>(idx.base, self.len(), m),
                    "spmd_write: LinearIndex out of bounds on an active lane"
                );
                value.0.store_select(&mut self[b..], m.cast::<T::Mask>());
            }
            #[inline(always)]
            unsafe fn spmd_write_unchecked(
                &mut self,
                idx: LinearIndex<N>,
                exec: $E,
                value: Varying<T, N>,
            ) {
                let b = idx.base.min(self.len());
                let m: Mask<i32, N> = {
                    let $e = exec;
                    $mask
                };
                value
                    .0
                    .store_select_unchecked(unsafe { self.get_unchecked_mut(b..) }, m.cast::<T::Mask>());
            }
        }
    };
}
impl_linear_masked!(VMask<N>, |e| e.0);
impl_linear_masked!(VMaskGuard<N>, |e| e.0 & Mask::splat(e.1));


impl<T: SimdElement + Default, const N: usize, E: ActiveMask<N>> SpmdRead<Varying<i32, N>, E>
    for [T]
where
    LaneCount<N>: SupportedLaneCount,
{
    type Out = Varying<T, N>;
    #[inline(always)]
    fn spmd_read(&self, idx: Varying<i32, N>, exec: E) -> Varying<T, N> {
        let all_active = exec.all_known_active();
        let active = exec.active_mask();
        let idxs = idx.cast::<usize>().0;
        debug_assert!(
            gather_in_bounds::<N>(idxs, self.len(), active),
            "spmd_read: gather index out of bounds on an active lane"
        );
        if all_active {
            if all_lanes_in_bounds(idx.0, self.len()) {
                return Varying(unsafe {
                    Simd::gather_select_unchecked(
                        self,
                        Mask::splat(true),
                        idxs,
                        Simd::default(),
                    )
                });
            }
        } else {
            let sel = active.select(idx.0, Simd::splat(0));
            if all_lanes_in_bounds(sel, self.len()) {
                return Varying(unsafe {
                    Simd::gather_select_unchecked(
                        self,
                        active.cast::<isize>(),
                        sel.cast::<usize>(),
                        Simd::default(),
                    )
                });
            }
        }
        Varying(Simd::gather_select(self, active.cast::<isize>(), idxs, Simd::default()))
    }
    #[inline(always)]
    unsafe fn spmd_read_unchecked(&self, idx: Varying<i32, N>, exec: E) -> Varying<T, N> {
        let active = exec.active_mask();
        let idxs = idx.cast::<usize>().0;
        Varying(unsafe {
            Simd::gather_select_unchecked(self, active.cast::<isize>(), idxs, Simd::default())
        })
    }
}

impl<T: SimdElement, const N: usize, E: ActiveMask<N>> SpmdWrite<Varying<i32, N>, E, Varying<T, N>>
    for [T]
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline(always)]
    fn spmd_write(&mut self, idx: Varying<i32, N>, exec: E, value: Varying<T, N>) {
        let all_active = exec.all_known_active();
        let active = exec.active_mask();
        let idxs = idx.cast::<usize>().0;
        debug_assert!(
            gather_in_bounds::<N>(idxs, self.len(), active),
            "spmd_write: scatter index out of bounds on an active lane"
        );
        if all_active {
            if all_lanes_in_bounds(idx.0, self.len()) {
                unsafe {
                    value.0.scatter_select_unchecked(self, Mask::splat(true), idxs);
                }
                return;
            }
        } else {
            let sel = active.select(idx.0, Simd::splat(0));
            if all_lanes_in_bounds(sel, self.len()) {
                unsafe {
                    value
                        .0
                        .scatter_select_unchecked(self, active.cast::<isize>(), sel.cast::<usize>());
                }
                return;
            }
        }
        value.0.scatter_select(self, active.cast::<isize>(), idxs);
    }
    #[inline(always)]
    unsafe fn spmd_write_unchecked(&mut self, idx: Varying<i32, N>, exec: E, value: Varying<T, N>) {
        let active = exec.active_mask();
        let idxs = idx.cast::<usize>().0;
        unsafe {
            value.0.scatter_select_unchecked(self, active.cast::<isize>(), idxs);
        }
    }
}


#[inline(always)]
fn all_lanes_in_bounds<const N: usize>(idxs: Simd<i32, N>, len: usize) -> bool
where
    LaneCount<N>: SupportedLaneCount,
{
    len <= i32::MAX as usize && (idxs.cast::<u32>().reduce_max() as usize) < len
}


#[inline(always)]
fn iota_usize<const N: usize>() -> Simd<usize, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    Simd::from_array(core::array::from_fn(|l| l))
}

#[inline(always)]
fn linear_in_bounds<const N: usize>(base: usize, len: usize, active: Mask<i32, N>) -> bool
where
    LaneCount<N>: SupportedLaneCount,
{
    let idxs = Simd::splat(base) + iota_usize::<N>();
    let oob = idxs.simd_ge(Simd::splat(len)).cast::<i32>();
    !(active & oob).any()
}

#[inline(always)]
fn gather_in_bounds<const N: usize>(idxs: Simd<usize, N>, len: usize, active: Mask<i32, N>) -> bool
where
    LaneCount<N>: SupportedLaneCount,
{
    let oob = idxs.simd_ge(Simd::splat(len)).cast::<i32>();
    !(active & oob).any()
}


#[inline]
pub unsafe fn gather_field<S, F, const N: usize, E>(
    base: &[S],
    idx: Varying<i32, N>,
    field_offset: usize,
    exec: E,
) -> Varying<F, N>
where
    F: SimdElement + Default,
    E: ActiveMask<N>,
    LaneCount<N>: SupportedLaneCount,
{
    let active = exec.active_mask();
    let idxs = idx.cast::<usize>().0;
    let len = base.len();
    debug_assert!(
        gather_in_bounds::<N>(idxs, len, active),
        "gather_field: element index out of bounds on an active lane"
    );
    let in_bounds = idxs.simd_lt(Simd::splat(len)).cast::<i32>();
    let enable = (active & in_bounds).cast::<isize>();

    let elem_ptrs: Simd<*const S, N> = Simd::splat(base.as_ptr()).wrapping_add(idxs);
    let field_ptrs = elem_ptrs
        .cast::<u8>()
        .wrapping_add(Simd::splat(field_offset))
        .cast::<F>();

    Varying(unsafe { Simd::gather_select_ptr(field_ptrs, enable, Simd::default()) })
}


#[inline]
pub fn scatter_add<T, const N: usize, E>(
    dst: &mut [T],
    idx: Varying<i32, N>,
    exec: E,
    values: Varying<T, N>,
) where
    T: SimdElement + core::ops::AddAssign,
    E: ActiveMask<N>,
    LaneCount<N>: SupportedLaneCount,
{
    let active = exec.active_mask();
    let idxs = idx.cast::<usize>().0;
    debug_assert!(
        gather_in_bounds::<N>(idxs, dst.len(), active),
        "scatter_add: index out of bounds on an active lane"
    );
    let vals = values.to_array();
    let ix = idxs.to_array();
    for l in 0..N {
        if active.test(l) {
            if let Some(slot) = dst.get_mut(ix[l]) {
                *slot += vals[l];
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    const N: usize = 4;
    type Vi = Varying<i32, N>;

    fn mask(bits: [bool; N]) -> Mask<i32, N> {
        Mask::from_array(bits)
    }


    #[test]
    fn usize_plain_read_write() {
        let a = [10i32, 11, 12, 13];
        assert_eq!(a[..].spmd_read(2usize, AllOn), 12);
        assert_eq!(a[..].spmd_read(0usize, BoolGuard(true)), 10);
        assert_eq!(a[..].spmd_read(3usize, VMask::<N>(mask([true; N]))), 13);

        let mut b = [0i32; 4];
        b[..].spmd_write(1usize, AllOn, 99);
        b[..].spmd_write(3usize, BoolGuard(true), 77);
        assert_eq!(b, [0, 99, 0, 77]);

        assert_eq!(unsafe { a[..].spmd_read_unchecked(1usize, AllOn) }, 11);
        unsafe { b[..].spmd_write_unchecked(0usize, AllOn, 5) };
        assert_eq!(b[0], 5);
    }


    #[test]
    fn linear_contiguous_unmasked() {
        let a: Vec<i32> = (0..8).collect();
        let r = a[..].spmd_read(LinearIndex::<N>::new(0), AllOn);
        assert_eq!(r.to_array(), [0, 1, 2, 3]);
        let r = a[..].spmd_read(LinearIndex::<N>::new(2), BoolGuard(true));
        assert_eq!(r.to_array(), [2, 3, 4, 5]);

        let mut out = vec![0i32; 8];
        out[..].spmd_write(LinearIndex::<N>::new(4), AllOn, Varying::from_array([7, 8, 9, 10]));
        assert_eq!(out, [0, 0, 0, 0, 7, 8, 9, 10]);

        let r = unsafe { a[..].spmd_read_unchecked(LinearIndex::<N>::new(1), AllOn) };
        assert_eq!(r.to_array(), [1, 2, 3, 4]);
    }


    #[test]
    fn linear_contiguous_masked_tail() {
        let a: Vec<i32> = (10..16).collect(); 
        let tail_mask = VMask::<N>::first(2); 
        let r = a[..].spmd_read(LinearIndex::<N>::new(4), tail_mask);
        assert_eq!(r.to_array(), [14, 15, 0, 0]);

        let mut out = vec![0i32; 6];
        out[..].spmd_write(LinearIndex::<N>::new(4), tail_mask, Varying::from_array([1, 2, 3, 4]));
        assert_eq!(out, [0, 0, 0, 0, 1, 2]);

        let r = a[..].spmd_read(LinearIndex::<N>::new(4), VMaskGuard::<N>(mask([true, true, false, false]), true));
        assert_eq!(r.to_array(), [14, 15, 0, 0]);
    }


    #[test]
    fn gather_scatter_unmasked() {
        let a: Vec<i32> = (10..19).collect(); 
        let idx = Vi::from_array([3, 0, 5, 1]);
        let r = a[..].spmd_read(idx, AllOn);
        assert_eq!(r.to_array(), [13, 10, 15, 11]);

        let mut out = vec![0i32; 9];
        out[..].spmd_write(idx, AllOn, Varying::from_array([-1, -2, -3, -4]));
        assert_eq!(out, [-2, -4, 0, -1, 0, -3, 0, 0, 0]);
    }


    #[test]
    fn gather_scatter_masked() {
        let a: Vec<i32> = (10..19).collect();
        let idx = Vi::from_array([3, 0, 5, 1]);
        let m = VMask::<N>(mask([true, false, true, false]));
        let r = a[..].spmd_read(idx, m);
        assert_eq!(r.to_array(), [13, 0, 15, 0]);

        let mut out = vec![0i32; 9];
        out[..].spmd_write(idx, m, Varying::from_array([100, 200, 300, 400]));
        assert_eq!(out, [0, 0, 0, 100, 0, 300, 0, 0, 0]);
    }

    #[test]
    fn inactive_lanes_never_touch_memory() {
        let a = [10i32, 11, 12, 13];
        let idx = Vi::from_array([1, 999_999, 2, -5]);
        let m = VMask::<N>(mask([true, false, true, false]));
        let r = a[..].spmd_read(idx, m); 
        assert_eq!(r.to_array(), [11, 0, 12, 0]);

        let mut out = [0i32, 1, 2, 3];
        let widx = Vi::from_array([0, 999_999, 2, -5]);
        out[..].spmd_write(widx, m, Varying::from_array([100, 200, 300, 400]));
        assert_eq!(out, [100, 1, 300, 3]); 

        #[repr(C)]
        #[derive(Clone, Copy)]
        struct P {
            a: i32,
            b: i32,
        }
        let pts = [P { a: 0, b: 10 }, P { a: 1, b: 11 }, P { a: 2, b: 12 }];
        let off = core::mem::offset_of!(P, b);
        let f = unsafe { gather_field::<P, i32, N, _>(&pts, idx, off, m) };
        assert_eq!(f.to_array(), [11, 0, 12, 0]);
    }


    #[test]
    fn aos_field_gather() {
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct Particle {
            id: i32,
            mass: f32,
            charge: i32,
        }
        let ps = [
            Particle { id: 0, mass: 1.0, charge: -1 },
            Particle { id: 1, mass: 2.0, charge: 0 },
            Particle { id: 2, mass: 3.0, charge: 1 },
            Particle { id: 3, mass: 4.0, charge: 2 },
        ];
        let idx = Vi::from_array([0, 2, 1, 3]);

        let ids = unsafe {
            gather_field::<Particle, i32, N, _>(&ps, idx, core::mem::offset_of!(Particle, id), AllOn)
        };
        assert_eq!(ids.to_array(), [0, 2, 1, 3]);

        let mass = unsafe {
            gather_field::<Particle, f32, N, _>(&ps, idx, core::mem::offset_of!(Particle, mass), AllOn)
        };
        assert_eq!(mass.to_array(), [1.0, 3.0, 2.0, 4.0]);

        let charge = unsafe {
            gather_field::<Particle, i32, N, _>(
                &ps,
                idx,
                core::mem::offset_of!(Particle, charge),
                AllOn,
            )
        };
        assert_eq!(charge.to_array(), [-1, 1, 0, 2]);
    }


    #[test]
    fn linear_index_arithmetic() {
        let i = LinearIndex::<N>::new(5);
        assert_eq!(i.to_varying().to_array(), [5, 6, 7, 8]);
        assert_eq!((i + 3).base(), 8);
        assert_eq!((3 + i).base(), 8);
        assert_eq!((i + (-2)).base(), 3);
        let v: Varying<i32, N> = i + Varying::from_array([0, 10, 20, 30]);
        assert_eq!(v.to_array(), [5, 16, 27, 38]);
        let v2: Varying<i32, N> = Varying::from_array([100, 100, 100, 100]) + i;
        assert_eq!(v2.to_array(), [105, 106, 107, 108]);
        let v3: Varying<i32, N> = i.into();
        assert_eq!(v3.to_array(), [5, 6, 7, 8]);

        let a: Vec<i32> = (0..16).collect();
        let base = LinearIndex::<N>::new(2);
        let contiguous = a[..].spmd_read(base, AllOn);
        let gathered = a[..].spmd_read(base.to_varying(), AllOn);
        assert_eq!(contiguous.to_array(), gathered.to_array());
        assert_eq!(contiguous.to_array(), [2, 3, 4, 5]);
    }


    #[test]
    fn unchecked_matches_checked() {
        let a: Vec<i32> = (0..16).collect();
        let idx = Vi::from_array([3, 7, 1, 9]);
        let m = VMask::<N>(mask([true, true, false, true]));

        let checked = a[..].spmd_read(idx, m);
        let unchecked = unsafe { a[..].spmd_read_unchecked(idx, m) };
        assert_eq!(checked.to_array(), unchecked.to_array());

        let li = LinearIndex::<N>::new(5);
        let checked = a[..].spmd_read(li, VMask::<N>(mask([true; N])));
        let unchecked = unsafe { a[..].spmd_read_unchecked(li, VMask::<N>(mask([true; N]))) };
        assert_eq!(checked.to_array(), unchecked.to_array());

        let mut o1 = vec![0i32; 16];
        let mut o2 = vec![0i32; 16];
        let val = Varying::from_array([10, 20, 30, 40]);
        o1[..].spmd_write(idx, m, val);
        unsafe { o2[..].spmd_write_unchecked(idx, m, val) };
        assert_eq!(o1, o2);
    }

    #[test]
    fn inactive_lanes_never_touch_memory_unchecked() {
        let a = [10i32, 11, 12, 13];
        let idx = Vi::from_array([1, 999_999, 2, -5]);
        let m = VMask::<N>(mask([true, false, true, false]));
        let r = unsafe { a[..].spmd_read_unchecked(idx, m) };
        assert_eq!(r.to_array(), [11, 0, 12, 0]);

        let mut out = [0i32, 1, 2, 3];
        let widx = Vi::from_array([0, 999_999, 2, -5]);
        unsafe { out[..].spmd_write_unchecked(widx, m, Varying::from_array([100, 200, 300, 400])) };
        assert_eq!(out, [100, 1, 300, 3]);
    }


    #[test]
    fn gather_scatter_fast_path_boundaries() {
        let a: Vec<i32> = (100..108).collect(); 
        let idx = Vi::from_array([0, 7, 3, 7]);

        let r = a[..].spmd_read(idx, AllOn);
        assert_eq!(r.to_array(), [100, 107, 103, 107]);
        let r = a[..].spmd_read(idx, BoolGuard(true));
        assert_eq!(r.to_array(), [100, 107, 103, 107]);
        let r = a[..].spmd_read(idx, BoolGuard(false));
        assert_eq!(r.to_array(), [0; N]);

        let mut out = vec![0i32; 8];
        out[..].spmd_write(idx, AllOn, Varying::from_array([1, 2, 3, 4]));
        assert_eq!(out, [1, 0, 0, 3, 0, 0, 0, 4]);
        let mut out = vec![0i32; 8];
        out[..].spmd_write(idx, BoolGuard(true), Varying::from_array([1, 2, 3, 4]));
        assert_eq!(out, [1, 0, 0, 3, 0, 0, 0, 4]);
        let mut out = vec![9i32; 8];
        out[..].spmd_write(idx, BoolGuard(false), Varying::from_array([1, 2, 3, 4]));
        assert_eq!(out, [9; 8]);

        let gidx = Vi::from_array([0, -1, 7, 999_999]);
        let m = VMask::<N>(mask([true, false, true, false]));
        let r = a[..].spmd_read(gidx, m);
        assert_eq!(r.to_array(), [100, 0, 107, 0]);
        let mut out = vec![0i32; 8];
        out[..].spmd_write(gidx, m, Varying::from_array([1, 2, 3, 4]));
        assert_eq!(out, [1, 0, 0, 0, 0, 0, 0, 3]);

        let mut out = vec![0i32; 8];
        out[..].spmd_write(
            Vi::from_array([2, -7, 2, 999]),
            m,
            Varying::from_array([10, 20, 30, 40]),
        );
        assert_eq!(out, [0, 0, 30, 0, 0, 0, 0, 0]);
    }


    #[test]
    fn gather_scatter_all_inactive_garbage() {
        let none = VMask::<N>(mask([false; N]));
        let garbage = Vi::from_array([i32::MIN, -1, i32::MAX, 999_999]);

        let a = [5i32, 6, 7, 8];
        let r = a[..].spmd_read(garbage, none);
        assert_eq!(r.to_array(), [0; N]);
        let mut out = [1i32, 2, 3, 4];
        out[..].spmd_write(garbage, none, Varying::splat(-1));
        assert_eq!(out, [1, 2, 3, 4]);

        let e: [i32; 0] = [];
        let r = e[..].spmd_read(garbage, none);
        assert_eq!(r.to_array(), [0; N]);
        let mut e: [i32; 0] = [];
        e[..].spmd_write(garbage, none, Varying::splat(-1));
    }

    #[test]
    fn masked_linear_base_past_end_all_inactive() {
        let a = [1i32, 2, 3, 4];
        let none = VMask::<N>(mask([false; N]));
        let r = a[..].spmd_read(LinearIndex::<N>::new(6), none);
        assert_eq!(r.to_array(), [0; N]);
        let r = unsafe { a[..].spmd_read_unchecked(LinearIndex::<N>::new(6), none) };
        assert_eq!(r.to_array(), [0; N]);

        let mut out = [7i32, 7, 7, 7];
        out[..].spmd_write(LinearIndex::<N>::new(6), none, Varying::splat(-1));
        unsafe { out[..].spmd_write_unchecked(LinearIndex::<N>::new(9), none, Varying::splat(-1)) };
        assert_eq!(out, [7, 7, 7, 7]);

        let g = VMaskGuard::<N>(mask([false; N]), true);
        let r = a[..].spmd_read(LinearIndex::<N>::new(100), g);
        assert_eq!(r.to_array(), [0; N]);
    }


    #[test]
    fn varying_array_masked_usize_write() {
        type Vf = Varying<f32, N>;
        let m = VMask::<N>(mask([true, false, true, false]));

        let mut v = [Vf::splat(1.0); 3];
        v[..].spmd_write(1usize, m, Vf::splat(9.0));
        assert_eq!(v[1].to_array(), [9.0, 1.0, 9.0, 1.0]);
        assert_eq!(v[0].to_array(), [1.0; N]);

        let g = VMaskGuard::<N>(mask([true, false, true, false]), true);
        v[..].spmd_write(2usize, g, Vf::splat(5.0));
        assert_eq!(v[2].to_array(), [5.0, 1.0, 5.0, 1.0]);

        v[..].spmd_write(0usize, AllOn, 2.0f32);
        assert_eq!(v[0].to_array(), [2.0; N]);
        v[..].spmd_write(0usize, BoolGuard(true), 3.0f32);
        assert_eq!(v[0].to_array(), [3.0; N]);
        v[..].spmd_write(0usize, m, 4.0f32);
        assert_eq!(v[0].to_array(), [4.0, 3.0, 4.0, 3.0]);

        unsafe { v[..].spmd_write_unchecked(1usize, m, Vf::splat(-1.0)) };
        assert_eq!(v[1].to_array(), [-1.0, 1.0, -1.0, 1.0]);
        unsafe { v[..].spmd_write_unchecked(2usize, AllOn, 0.5f32) };
        assert_eq!(v[2].to_array(), [0.5; N]);

        let r: Vf = v[..].spmd_read(1usize, m);
        assert_eq!(r.to_array(), [-1.0, 1.0, -1.0, 1.0]);
    }


    #[test]
    fn scatter_add_handles_lane_conflicts() {
        let mut img = [0.0f32; 4];
        let idx = Vi::from_array([1, 1, 1, 3]);
        let vals = Varying::from_array([1.0f32, 2.0, 4.0, 8.0]);
        scatter_add(&mut img, idx, AllOn, vals);
        assert_eq!(img, [0.0, 7.0, 0.0, 8.0]);

        let mut img = [0.0f32; 4];
        let idx = Vi::from_array([0, 999_999, 0, -5]);
        let m = VMask::<N>(mask([true, false, true, false]));
        scatter_add(&mut img, idx, m, vals);
        assert_eq!(img, [5.0, 0.0, 0.0, 0.0]);

        let mut hist = [0i32; 3];
        scatter_add(
            &mut hist,
            Vi::from_array([2, 2, 2, 2]),
            AllOn,
            Varying::from_array([1, 1, 1, 1]),
        );
        assert_eq!(hist, [0, 0, 4]);
    }
}
