
use crate::memory::{gather_field, ActiveMask};
use crate::varying::Varying;
use core::simd::{LaneCount, SupportedLaneCount};

pub trait SpmdValue: Copy {
    type Varying<const N: usize>: Copy
    where
        LaneCount<N>: SupportedLaneCount;

    fn splat<const N: usize>(self) -> Self::Varying<N>
    where
        LaneCount<N>: SupportedLaneCount;
}

pub trait SpmdGather: SpmdValue {
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
                unsafe { gather_field::<Base, $t, N, E>(base, idx, field_offset, exec) }
            }
        }
    )* };
}

impl_prim_spmd_value!(f32, f64, i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);
