
use core::simd::cmp::SimdPartialOrd;
use core::simd::{LaneCount, Mask, Simd, SupportedLaneCount};


#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct AllOn;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BoolGuard(pub bool);

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct VMask<const N: usize>(pub Mask<i32, N>)
where
    LaneCount<N>: SupportedLaneCount;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct VMaskGuard<const N: usize>(pub Mask<i32, N>, pub bool)
where
    LaneCount<N>: SupportedLaneCount;

impl<const N: usize> VMask<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline(always)]
    pub fn full() -> Self {
        VMask(Mask::splat(true))
    }

    #[inline(always)]
    pub fn first(k: usize) -> Self {
        let iota = Simd::<i32, N>::from_array(core::array::from_fn(|i| i as i32));
        VMask(iota.simd_lt(Simd::splat(k as i32)))
    }
}


#[diagnostic::on_unimplemented(
    message = "`{Self}` is not an rustlane execution context",
    label = "expected `AllOn`, `BoolGuard`, `VMask<N>` or `VMaskGuard<N>`",
    note = "`__exec` values are produced by the `#[rustlane::kernel]` rewrite; \
            they should not be constructed from arbitrary types"
)]
pub trait Exec: Copy {
    const UNIFORM: bool;

    fn should_branch(self) -> bool;

    fn any(self) -> bool;

    #[inline(always)]
    fn is_statically_uniform(self) -> bool {
        Self::UNIFORM
    }
}

impl Exec for AllOn {
    const UNIFORM: bool = true;
    #[inline(always)]
    fn should_branch(self) -> bool {
        true
    }
    #[inline(always)]
    fn any(self) -> bool {
        true
    }
}

impl Exec for BoolGuard {
    const UNIFORM: bool = true;
    #[inline(always)]
    fn should_branch(self) -> bool {
        self.0
    }
    #[inline(always)]
    fn any(self) -> bool {
        self.0
    }
}

impl<const N: usize> Exec for VMask<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    const UNIFORM: bool = false;
    #[inline(always)]
    fn should_branch(self) -> bool {
        true
    }
    #[inline(always)]
    fn any(self) -> bool {
        self.0.any()
    }
}

impl<const N: usize> Exec for VMaskGuard<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    const UNIFORM: bool = false;
    #[inline(always)]
    fn should_branch(self) -> bool {
        self.1
    }
    #[inline(always)]
    fn any(self) -> bool {
        self.1 && self.0.any()
    }
}


#[diagnostic::on_unimplemented(
    message = "an rustlane `if`/`while` condition must be `bool` (uniform) or `Mask<i32, N>` (varying), not `{C}`",
    label = "cannot narrow execution context `{Self}` by this condition",
    note = "comparisons on `Varying` values produce `Mask<i32, N>`; \
            comparisons on uniform values produce `bool`"
)]
pub trait AndCond<C>: Sized {
    type Out: Exec;
    fn and_cond(self, cond: C) -> Self::Out;
    fn and_not_cond(self, cond: C) -> Self::Out;
}

impl AndCond<bool> for AllOn {
    type Out = BoolGuard;
    #[inline(always)]
    fn and_cond(self, cond: bool) -> BoolGuard {
        BoolGuard(cond)
    }
    #[inline(always)]
    fn and_not_cond(self, cond: bool) -> BoolGuard {
        BoolGuard(!cond)
    }
}

impl<const N: usize> AndCond<Mask<i32, N>> for AllOn
where
    LaneCount<N>: SupportedLaneCount,
{
    type Out = VMask<N>;
    #[inline(always)]
    fn and_cond(self, cond: Mask<i32, N>) -> VMask<N> {
        VMask(cond)
    }
    #[inline(always)]
    fn and_not_cond(self, cond: Mask<i32, N>) -> VMask<N> {
        VMask(!cond)
    }
}

impl AndCond<bool> for BoolGuard {
    type Out = BoolGuard;
    #[inline(always)]
    fn and_cond(self, cond: bool) -> BoolGuard {
        BoolGuard(self.0 & cond)
    }
    #[inline(always)]
    fn and_not_cond(self, cond: bool) -> BoolGuard {
        BoolGuard(self.0 & !cond)
    }
}

impl<const N: usize> AndCond<Mask<i32, N>> for BoolGuard
where
    LaneCount<N>: SupportedLaneCount,
{
    type Out = VMask<N>;
    #[inline(always)]
    fn and_cond(self, cond: Mask<i32, N>) -> VMask<N> {
        VMask(cond & Mask::splat(self.0))
    }
    #[inline(always)]
    fn and_not_cond(self, cond: Mask<i32, N>) -> VMask<N> {
        VMask(!cond & Mask::splat(self.0))
    }
}

impl<const N: usize> AndCond<Mask<i32, N>> for VMask<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    type Out = VMask<N>;
    #[inline(always)]
    fn and_cond(self, cond: Mask<i32, N>) -> VMask<N> {
        VMask(self.0 & cond)
    }
    #[inline(always)]
    fn and_not_cond(self, cond: Mask<i32, N>) -> VMask<N> {
        VMask(self.0 & !cond)
    }
}

impl<const N: usize> AndCond<bool> for VMask<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    type Out = VMaskGuard<N>;
    #[inline(always)]
    fn and_cond(self, cond: bool) -> VMaskGuard<N> {
        VMaskGuard(self.0, cond)
    }
    #[inline(always)]
    fn and_not_cond(self, cond: bool) -> VMaskGuard<N> {
        VMaskGuard(self.0, !cond)
    }
}

impl<const N: usize> AndCond<Mask<i32, N>> for VMaskGuard<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    type Out = VMask<N>;
    #[inline(always)]
    fn and_cond(self, cond: Mask<i32, N>) -> VMask<N> {
        VMask(self.0 & cond & Mask::splat(self.1))
    }
    #[inline(always)]
    fn and_not_cond(self, cond: Mask<i32, N>) -> VMask<N> {
        VMask(self.0 & !cond & Mask::splat(self.1))
    }
}

impl<const N: usize> AndCond<bool> for VMaskGuard<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    type Out = VMaskGuard<N>;
    #[inline(always)]
    fn and_cond(self, cond: bool) -> VMaskGuard<N> {
        VMaskGuard(self.0, self.1 & cond)
    }
    #[inline(always)]
    fn and_not_cond(self, cond: bool) -> VMaskGuard<N> {
        VMaskGuard(self.0, self.1 & !cond)
    }
}


#[diagnostic::on_unimplemented(
    message = "cannot assign to a value of type `{Self}` under execution context `{E}`",
    label = "this assignment target cannot be written under the current control-flow mask",
    note = "assigning to a uniform (scalar) variable under VARYING control flow is not \
            supported: every lane would race on one location. Make the variable a \
            `Varying`, hoist the assignment out of the varying branch, or wrap it in \
            `unmasked!` if all-lanes semantics are intended"
)]
pub trait MaskedAssign<E, V = Self> {
    fn masked_assign(&mut self, exec: E, value: V);
}

macro_rules! impl_scalar_masked_assign {
    ($($t:ty),* $(,)?) => { $(
        impl MaskedAssign<AllOn> for $t {
            #[inline(always)]
            fn masked_assign(&mut self, _exec: AllOn, value: $t) {
                *self = value;
            }
        }
        impl MaskedAssign<BoolGuard> for $t {
            #[inline(always)]
            fn masked_assign(&mut self, _exec: BoolGuard, value: $t) {
                *self = value;
            }
        }
    )* };
}

impl_scalar_masked_assign!(f32, f64, i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, bool);



#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct UniformLoop(pub bool);

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct VaryingLoop<const N: usize>(pub Mask<i32, N>)
where
    LaneCount<N>: SupportedLaneCount;

#[diagnostic::on_unimplemented(
    message = "cannot start an rustlane `while` loop from context `{Self}` with a `{C}` condition",
    label = "unsupported loop-condition / context combination",
    note = "a uniform (`bool`) `while` condition inside VARYING control flow is not \
            supported in v1: rewrite as `loop {{ if !cond {{ break; }} .. }}` or make \
            the condition varying"
)]
pub trait EnterLoop<C> {
    type LoopState: SpmdLoop;
    fn enter_loop(self, first_cond: C) -> Self::LoopState;
}

impl EnterLoop<bool> for AllOn {
    type LoopState = UniformLoop;
    #[inline(always)]
    fn enter_loop(self, first_cond: bool) -> UniformLoop {
        UniformLoop(first_cond)
    }
}

impl<const N: usize> EnterLoop<Mask<i32, N>> for AllOn
where
    LaneCount<N>: SupportedLaneCount,
{
    type LoopState = VaryingLoop<N>;
    #[inline(always)]
    fn enter_loop(self, first_cond: Mask<i32, N>) -> VaryingLoop<N> {
        VaryingLoop(first_cond)
    }
}

impl EnterLoop<bool> for BoolGuard {
    type LoopState = UniformLoop;
    #[inline(always)]
    fn enter_loop(self, first_cond: bool) -> UniformLoop {
        UniformLoop(self.0 & first_cond)
    }
}

impl<const N: usize> EnterLoop<Mask<i32, N>> for BoolGuard
where
    LaneCount<N>: SupportedLaneCount,
{
    type LoopState = VaryingLoop<N>;
    #[inline(always)]
    fn enter_loop(self, first_cond: Mask<i32, N>) -> VaryingLoop<N> {
        VaryingLoop(first_cond & Mask::splat(self.0))
    }
}

impl<const N: usize> EnterLoop<Mask<i32, N>> for VMask<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    type LoopState = VaryingLoop<N>;
    #[inline(always)]
    fn enter_loop(self, first_cond: Mask<i32, N>) -> VaryingLoop<N> {
        VaryingLoop(self.0 & first_cond)
    }
}

impl<const N: usize> EnterLoop<Mask<i32, N>> for VMaskGuard<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    type LoopState = VaryingLoop<N>;
    #[inline(always)]
    fn enter_loop(self, first_cond: Mask<i32, N>) -> VaryingLoop<N> {
        VaryingLoop(self.0 & first_cond & Mask::splat(self.1))
    }
}


#[diagnostic::on_unimplemented(
    message = "cannot start an rustlane `for`/`loop` from context `{Self}`",
    label = "not an rustlane execution context"
)]
pub trait EnterLoopN<const N: usize>
where
    LaneCount<N>: SupportedLaneCount,
{
    fn enter_loop_n(self) -> VaryingLoop<N>;
}

impl<const N: usize> EnterLoopN<N> for AllOn
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline(always)]
    fn enter_loop_n(self) -> VaryingLoop<N> {
        VaryingLoop(Mask::splat(true))
    }
}

impl<const N: usize> EnterLoopN<N> for BoolGuard
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline(always)]
    fn enter_loop_n(self) -> VaryingLoop<N> {
        VaryingLoop(Mask::splat(self.0))
    }
}

impl<const N: usize> EnterLoopN<N> for VMask<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline(always)]
    fn enter_loop_n(self) -> VaryingLoop<N> {
        VaryingLoop(self.0)
    }
}

impl<const N: usize> EnterLoopN<N> for VMaskGuard<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline(always)]
    fn enter_loop_n(self) -> VaryingLoop<N> {
        VaryingLoop(self.0 & Mask::splat(self.1))
    }
}

#[diagnostic::on_unimplemented(
    message = "rustlane loop state `{Self}` cannot be narrowed by a `{C}` condition",
    label = "loop condition type changed between iterations, or unsupported combination",
    note = "a `while` condition must keep one type (`bool` or `Mask<i32, N>`) across \
            iterations"
)]
pub trait LoopCond<C>: Sized {
    fn and_cond(self, cond: C) -> Self;
}

impl LoopCond<bool> for UniformLoop {
    #[inline(always)]
    fn and_cond(self, cond: bool) -> Self {
        UniformLoop(self.0 & cond)
    }
}

impl<const N: usize> LoopCond<Mask<i32, N>> for VaryingLoop<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline(always)]
    fn and_cond(self, cond: Mask<i32, N>) -> Self {
        VaryingLoop(self.0 & cond)
    }
}

#[diagnostic::on_unimplemented(
    message = "`{Self}` is not an rustlane loop state",
    label = "expected `UniformLoop` or `VaryingLoop<N>`"
)]
pub trait SpmdLoop: Copy {
    type IterExec: Exec;

    fn any(self) -> bool;

    fn current(self) -> Self::IterExec;

    #[inline(always)]
    fn iter_mask(self) -> Self {
        self
    }
}

impl SpmdLoop for UniformLoop {
    type IterExec = AllOn;
    #[inline(always)]
    fn any(self) -> bool {
        self.0
    }
    #[inline(always)]
    fn current(self) -> AllOn {
        AllOn
    }
}

impl<const N: usize> SpmdLoop for VaryingLoop<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    type IterExec = VMask<N>;
    #[inline(always)]
    fn any(self) -> bool {
        self.0.any()
    }
    #[inline(always)]
    fn current(self) -> VMask<N> {
        VMask(self.0)
    }
}

#[diagnostic::on_unimplemented(
    message = "`break`/`continue`/`return` under execution context `{E}` is not supported in this loop",
    label = "varying exit inside a loop whose state is uniform",
    note = "a VARYING `break`/`continue`/`return` inside a `while` loop with a UNIFORM \
            condition is not supported in v1: use `for`/`loop` form, or make the loop \
            condition varying"
)]
pub trait LoopRemove<E> {
    fn remove(&mut self, exec: E);
}

impl LoopRemove<AllOn> for UniformLoop {
    #[inline(always)]
    fn remove(&mut self, _exec: AllOn) {
        self.0 = false;
    }
}

impl LoopRemove<BoolGuard> for UniformLoop {
    #[inline(always)]
    fn remove(&mut self, exec: BoolGuard) {
        self.0 &= !exec.0;
    }
}

impl<const N: usize> LoopRemove<AllOn> for VaryingLoop<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline(always)]
    fn remove(&mut self, _exec: AllOn) {
        self.0 = Mask::splat(false);
    }
}

impl<const N: usize> LoopRemove<BoolGuard> for VaryingLoop<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline(always)]
    fn remove(&mut self, exec: BoolGuard) {
        self.0 &= Mask::splat(!exec.0);
    }
}

impl<const N: usize> LoopRemove<VMask<N>> for VaryingLoop<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline(always)]
    fn remove(&mut self, exec: VMask<N>) {
        self.0 &= !exec.0;
    }
}

impl<const N: usize> LoopRemove<VMaskGuard<N>> for VaryingLoop<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline(always)]
    fn remove(&mut self, exec: VMaskGuard<N>) {
        self.0 &= !(exec.0 & Mask::splat(exec.1));
    }
}

#[diagnostic::on_unimplemented(
    message = "cannot refresh execution context `{Self}` against loop state `{L}`",
    label = "unsupported context/loop combination",
    note = "this usually means a varying `break`/`continue`/`return` sits inside a loop \
            with a uniform condition, which is not supported in v1"
)]
pub trait Refresh<L> {
    type Out: Exec;
    fn refresh(self, state: &L) -> Self::Out;
}

impl Refresh<UniformLoop> for AllOn {
    type Out = AllOn;
    #[inline(always)]
    fn refresh(self, _state: &UniformLoop) -> AllOn {
        self
    }
}

impl Refresh<UniformLoop> for BoolGuard {
    type Out = BoolGuard;
    #[inline(always)]
    fn refresh(self, _state: &UniformLoop) -> BoolGuard {
        self
    }
}

impl<const N: usize> Refresh<VaryingLoop<N>> for AllOn
where
    LaneCount<N>: SupportedLaneCount,
{
    type Out = VMask<N>;
    #[inline(always)]
    fn refresh(self, state: &VaryingLoop<N>) -> VMask<N> {
        VMask(state.0)
    }
}

impl<const N: usize> Refresh<VaryingLoop<N>> for BoolGuard
where
    LaneCount<N>: SupportedLaneCount,
{
    type Out = VMask<N>;
    #[inline(always)]
    fn refresh(self, state: &VaryingLoop<N>) -> VMask<N> {
        VMask(state.0 & Mask::splat(self.0))
    }
}

impl<const N: usize> Refresh<VaryingLoop<N>> for VMask<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    type Out = VMask<N>;
    #[inline(always)]
    fn refresh(self, state: &VaryingLoop<N>) -> VMask<N> {
        VMask(self.0 & state.0)
    }
}

impl<const N: usize> Refresh<VaryingLoop<N>> for VMaskGuard<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    type Out = VMaskGuard<N>;
    #[inline(always)]
    fn refresh(self, state: &VaryingLoop<N>) -> VMaskGuard<N> {
        VMaskGuard(self.0 & state.0, self.1)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::varying::Varying;

    const N: usize = 4;
    type M = Mask<i32, N>;

    fn mask(bits: [bool; N]) -> M {
        Mask::from_array(bits)
    }


    #[test]
    fn and_cond_allon_bool_is_boolguard() {
        let g: BoolGuard = AllOn.and_cond(true);
        assert!(g.should_branch());
        let g = AllOn.and_cond(false);
        assert!(!g.should_branch());
        let g = AllOn.and_not_cond(false);
        assert!(g.should_branch());
    }

    #[test]
    fn and_cond_allon_mask_is_vmask() {
        let m = mask([true, false, true, false]);
        let e: VMask<N> = AllOn.and_cond(m);
        assert_eq!(e.0, m);
        assert!(e.should_branch()); 
        let e = AllOn.and_not_cond(m);
        assert_eq!(e.0, !m);
    }

    #[test]
    fn and_cond_boolguard_mask_is_vmask() {
        let m = mask([true, true, false, false]);
        let e: VMask<N> = BoolGuard(true).and_cond(m);
        assert_eq!(e.0, m);
    }

    #[test]
    fn and_cond_vmask_mask_ands() {
        let a = mask([true, true, false, false]);
        let b = mask([true, false, true, false]);
        let e: VMask<N> = VMask(a).and_cond(b);
        assert_eq!(e.0, a & b);
        let e = VMask(a).and_not_cond(b);
        assert_eq!(e.0, a & !b);
    }

    #[test]
    fn and_cond_vmask_bool_is_vmaskguard() {
        let a = mask([true, false, true, false]);
        let e: VMaskGuard<N> = VMask(a).and_cond(true);
        assert_eq!(e.0, a);
        assert!(e.should_branch());
        let e = VMask(a).and_cond(false);
        assert!(!e.should_branch());
        let e = VMask(a).and_not_cond(false);
        assert!(e.should_branch());
    }

    #[test]
    fn and_cond_vmaskguard_mask_is_vmask() {
        let a = mask([true, true, true, false]);
        let b = mask([false, true, true, true]);
        let e: VMask<N> = VMaskGuard(a, true).and_cond(b);
        assert_eq!(e.0, a & b);
    }

    #[test]
    fn should_branch_vs_any() {
        let empty = VMask::<N>(mask([false; N]));
        assert!(empty.should_branch());
        assert!(!empty.any());
        assert!(AllOn.should_branch() && AllOn.any());
        assert!(AllOn.is_statically_uniform());
        assert!(BoolGuard(false).is_statically_uniform());
        assert!(!empty.is_statically_uniform());
        assert!(!VMaskGuard::<N>(mask([true; N]), true).is_statically_uniform());
    }


    #[test]
    fn masked_assign_matrix() {
        let m = mask([true, false, true, false]);

        let mut v = Varying::<i32, N>::splat(1);
        v.masked_assign(AllOn, Varying::from_array([9, 8, 7, 6]));
        assert_eq!(v.to_array(), [9, 8, 7, 6]);

        let mut v = Varying::<i32, N>::splat(1);
        v.masked_assign(BoolGuard(true), Varying::splat(5));
        assert_eq!(v.to_array(), [5; N]);

        let mut v = Varying::<i32, N>::splat(0);
        v.masked_assign(VMask(m), Varying::splat(7));
        assert_eq!(v.to_array(), [7, 0, 7, 0]);

        let mut v = Varying::<i32, N>::splat(0);
        v.masked_assign(VMask(m), 3);
        assert_eq!(v.to_array(), [3, 0, 3, 0]);

        let mut v = Varying::<i32, N>::splat(0);
        v.masked_assign(VMaskGuard(m, true), Varying::splat(2));
        assert_eq!(v.to_array(), [2, 0, 2, 0]);

        let mut x = 1.5f32;
        x.masked_assign(AllOn, 2.5);
        assert_eq!(x, 2.5);
        x.masked_assign(BoolGuard(true), 3.5);
        assert_eq!(x, 3.5);
        let mut b = false;
        b.masked_assign(AllOn, true);
        assert!(b);
    }


    #[test]
    fn uniform_while_template() {
        fn kernel(n: i32) -> i32 {
            let __exec = AllOn;
            let mut acc = 0i32;
            let mut i = 0i32;
            let mut __loop = __exec.enter_loop(crate::cond::SpmdOrd::spmd_lt(i, n));
            loop {
                if !__loop.any() {
                    break;
                }
                let __exec = __loop.current();
                acc.masked_assign(__exec, acc + i);
                i.masked_assign(__exec, i + 1);
                let __c = crate::cond::SpmdOrd::spmd_lt(i, n);
                __loop = __loop.and_cond(__c);
            }
            acc
        }
        assert_eq!(kernel(5), 0 + 1 + 2 + 3 + 4);
        assert_eq!(kernel(0), 0); 
    }

    #[test]
    fn varying_while_template() {
        fn kernel(count: Varying<i32, N>) -> Varying<i32, N> {
            let __exec = AllOn;
            let mut i = Varying::<i32, N>::splat(0);
            let mut __loop = __exec.enter_loop(crate::cond::SpmdOrd::spmd_lt(i, count));
            loop {
                if !__loop.any() {
                    break;
                }
                let __exec = __loop.current();
                i.masked_assign(__exec, i + 1);
                let __c = crate::cond::SpmdOrd::spmd_lt(i, count);
                __loop = __loop.and_cond(__c);
            }
            i
        }
        let count = Varying::from_array([0, 1, 3, 7]);
        assert_eq!(kernel(count).to_array(), [0, 1, 3, 7]);
    }

    #[test]
    fn for_with_varying_break_template() {
        fn kernel(stop: Varying<i32, N>, max_iter: i32) -> Varying<i32, N> {
            let __exec = AllOn;
            let mut ret = Varying::<i32, N>::splat(0);
            let mut __loop = EnterLoopN::<N>::enter_loop_n(__exec);
            for i in 0..max_iter {
                if !__loop.any() {
                    break;
                }
                let __exec = __loop.current();
                {
                    let __c = crate::cond::SpmdOrd::spmd_ge(Varying::splat(i), stop);
                    let __exec1 = __exec.and_cond(__c);
                    if __exec1.should_branch() {
                        if __exec1.is_statically_uniform() {
                            break;
                        }
                        __loop.remove(__exec1);
                    }
                }
                let __exec = __exec.refresh(&__loop);
                ret.masked_assign(__exec, i + 1);
            }
            ret
        }
        let stop = Varying::from_array([0, 2, 5, 9]);
        assert_eq!(kernel(stop, 6).to_array(), [0, 2, 5, 6]);
    }

    #[test]
    fn for_with_varying_continue_template() {
        fn kernel(skip_below: Varying<i32, N>, n: i32) -> Varying<i32, N> {
            let __exec = AllOn;
            let mut acc = Varying::<i32, N>::splat(0);
            let mut __loop = EnterLoopN::<N>::enter_loop_n(__exec);
            for i in 0..n {
                if !__loop.any() {
                    break;
                }
                let mut __iter = __loop.iter_mask();
                let __exec = __loop.current();
                {
                    let __c = crate::cond::SpmdOrd::spmd_lt(Varying::splat(i), skip_below);
                    let __exec1 = __exec.and_cond(__c);
                    if __exec1.should_branch() {
                        if __exec1.is_statically_uniform() {
                            continue;
                        }
                        __iter.remove(__exec1);
                    }
                }
                let __exec = __exec.refresh(&__iter);
                acc.masked_assign(__exec, acc + 1);
            }
            acc
        }
        let skip = Varying::from_array([0, 2, 5, 100]);
        assert_eq!(kernel(skip, 8).to_array(), [8, 6, 3, 0]);
    }

    #[test]
    fn bare_loop_uniform_break_under_varying_state() {
        fn kernel(n: i32) -> i32 {
            let __exec = AllOn;
            let mut count = 0i32;
            let mut __loop = EnterLoopN::<N>::enter_loop_n(__exec);
            loop {
                if !__loop.any() {
                    break;
                }
                let __exec = __loop.current();
                {
                    let __c = crate::cond::SpmdOrd::spmd_ge(count, n);
                    let __exec1 = __exec.and_cond(__c);
                    if __exec1.should_branch() {
                        if __exec1.is_statically_uniform() {
                            break;
                        }
                        __loop.remove(__exec1);
                    }
                }
                let __exec = __exec.refresh(&__loop);
                if __exec.any() {
                    count += 1;
                }
            }
            count
        }
        assert_eq!(kernel(3), 3);
    }

    #[test]
    fn varying_return_template() {
        fn kernel(flag: Varying<i32, N>) -> Varying<i32, N> {
            let __exec = AllOn;
            let mut __ret = Varying::<i32, N>::default();
            let mut __fn = EnterLoopN::<N>::enter_loop_n(__exec);
            {
                let __c = crate::cond::SpmdOrd::spmd_gt(flag, 0);
                let __exec1 = __exec.and_cond(__c);
                if __exec1.should_branch() {
                    __ret.masked_assign(__exec1, 100);
                    if __exec1.is_statically_uniform() {
                        return __ret;
                    }
                    __fn.remove(__exec1);
                }
            }
            let __exec = __exec.refresh(&__fn);
            __ret.masked_assign(__exec, flag - 1);
            __ret
        }
        let flag = Varying::from_array([1, 0, 5, -2]);
        assert_eq!(kernel(flag).to_array(), [100, -1, 100, -3]);
    }

    #[test]
    fn vmask_first_and_full() {
        assert_eq!(VMask::<N>::first(0).0, mask([false; N]));
        assert_eq!(VMask::<N>::first(2).0, mask([true, true, false, false]));
        assert_eq!(VMask::<N>::first(9).0, mask([true; N]));
        assert_eq!(VMask::<N>::full().0, mask([true; N]));
    }

    #[test]
    fn refresh_matrix() {
        let l = VaryingLoop::<N>(mask([true, false, true, true]));
        let e = VMask(mask([true, true, false, true])).refresh(&l);
        assert_eq!(e.0, mask([true, false, false, true]));
        let e = AllOn.refresh(&l);
        assert_eq!(e.0, l.0);
        let e = VMaskGuard(mask([true; N]), true).refresh(&l);
        assert_eq!(e.0, l.0);
        assert!(e.should_branch());
        let ul = UniformLoop(true);
        let _: AllOn = AllOn.refresh(&ul);
        let g: BoolGuard = BoolGuard(true).refresh(&ul);
        assert!(g.should_branch());
    }
}
