
use crate::varying::Varying;
use core::simd::cmp::{SimdPartialEq, SimdPartialOrd};
use core::simd::num::{SimdFloat, SimdInt, SimdUint};
use core::simd::{LaneCount, Simd, SimdElement, SupportedLaneCount};
use std::simd::StdFloat;


#[inline(always)]
pub fn sqrt<T, const N: usize>(x: Varying<T, N>) -> Varying<T, N>
where
    T: SimdElement,
    LaneCount<N>: SupportedLaneCount,
    Simd<T, N>: StdFloat,
{
    Varying(x.0.sqrt())
}

#[inline(always)]
pub fn abs<T, const N: usize>(x: Varying<T, N>) -> Varying<T, N>
where
    T: SimdElement,
    LaneCount<N>: SupportedLaneCount,
    Simd<T, N>: SimdFloat,
{
    Varying(x.0.abs())
}

pub trait MinMaxElem: SimdElement {
    fn vmin<const N: usize>(a: Simd<Self, N>, b: Simd<Self, N>) -> Simd<Self, N>
    where
        LaneCount<N>: SupportedLaneCount;
    fn vmax<const N: usize>(a: Simd<Self, N>, b: Simd<Self, N>) -> Simd<Self, N>
    where
        LaneCount<N>: SupportedLaneCount;
}

macro_rules! impl_minmax_elem {
    ($cat:path : $($t:ty),* $(,)?) => { $(
        impl MinMaxElem for $t {
            #[inline(always)]
            fn vmin<const N: usize>(a: Simd<$t, N>, b: Simd<$t, N>) -> Simd<$t, N>
            where LaneCount<N>: SupportedLaneCount { <Simd<$t, N> as $cat>::simd_min(a, b) }
            #[inline(always)]
            fn vmax<const N: usize>(a: Simd<$t, N>, b: Simd<$t, N>) -> Simd<$t, N>
            where LaneCount<N>: SupportedLaneCount { <Simd<$t, N> as $cat>::simd_max(a, b) }
        }
    )* };
}

impl_minmax_elem!(SimdFloat: f32, f64);
impl_minmax_elem!(core::simd::cmp::SimdOrd: i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);

#[inline(always)]
pub fn min<T, const N: usize>(a: Varying<T, N>, b: Varying<T, N>) -> Varying<T, N>
where
    T: MinMaxElem,
    LaneCount<N>: SupportedLaneCount,
{
    Varying(T::vmin(a.0, b.0))
}

#[inline(always)]
pub fn max<T, const N: usize>(a: Varying<T, N>, b: Varying<T, N>) -> Varying<T, N>
where
    T: MinMaxElem,
    LaneCount<N>: SupportedLaneCount,
{
    Varying(T::vmax(a.0, b.0))
}

#[inline(always)]
pub fn floor<T, const N: usize>(x: Varying<T, N>) -> Varying<T, N>
where
    T: SimdElement,
    LaneCount<N>: SupportedLaneCount,
    Simd<T, N>: StdFloat,
{
    Varying(x.0.floor())
}

#[inline(always)]
pub fn ceil<T, const N: usize>(x: Varying<T, N>) -> Varying<T, N>
where
    T: SimdElement,
    LaneCount<N>: SupportedLaneCount,
    Simd<T, N>: StdFloat,
{
    Varying(x.0.ceil())
}

#[inline(always)]
pub fn round<T, const N: usize>(x: Varying<T, N>) -> Varying<T, N>
where
    T: SimdElement,
    LaneCount<N>: SupportedLaneCount,
    Simd<T, N>: StdFloat,
{
    Varying(x.0.round())
}

#[inline(always)]
pub fn clamp<T, const N: usize>(
    x: Varying<T, N>,
    lo: Varying<T, N>,
    hi: Varying<T, N>,
) -> Varying<T, N>
where
    T: MinMaxElem,
    LaneCount<N>: SupportedLaneCount,
{
    Varying(T::vmin(T::vmax(x.0, lo.0), hi.0))
}

#[inline(always)]
pub fn lerp<T, const N: usize>(a: Varying<T, N>, b: Varying<T, N>, t: Varying<T, N>) -> Varying<T, N>
where
    T: SimdElement,
    LaneCount<N>: SupportedLaneCount,
    Simd<T, N>: StdFloat + core::ops::Sub<Output = Simd<T, N>>,
{
    Varying((b.0 - a.0).mul_add(t.0, a.0))
}


#[inline(always)]
pub fn rsqrt<const N: usize>(x: Varying<f32, N>) -> Varying<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    Varying(rsqrt_simd(x.0))
}

#[inline(always)]
pub fn rcp<const N: usize>(x: Varying<f32, N>) -> Varying<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    Varying(rcp_simd(x.0))
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn rsqrt_simd<const N: usize>(x: Simd<f32, N>) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    use core::arch::aarch64::{vld1q_f32, vmulq_f32, vrsqrteq_f32, vrsqrtsq_f32, vst1q_f32};
    let a = x.to_array();
    let mut out = [0.0f32; N];
    let mut i = 0;
    while i + 4 <= N {
        unsafe {
            let v = vld1q_f32(a.as_ptr().add(i));
            let mut e = vrsqrteq_f32(v);
            e = vmulq_f32(e, vrsqrtsq_f32(vmulq_f32(v, e), e));
            e = vmulq_f32(e, vrsqrtsq_f32(vmulq_f32(v, e), e));
            vst1q_f32(out.as_mut_ptr().add(i), e);
        }
        i += 4;
    }
    while i < N {
        out[i] = 1.0 / a[i].sqrt();
        i += 1;
    }
    Simd::from_array(out)
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn rsqrt_simd<const N: usize>(x: Simd<f32, N>) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    Simd::splat(1.0) / x.sqrt()
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn rcp_simd<const N: usize>(x: Simd<f32, N>) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    use core::arch::aarch64::{vld1q_f32, vmulq_f32, vrecpeq_f32, vrecpsq_f32, vst1q_f32};
    let a = x.to_array();
    let mut out = [0.0f32; N];
    let mut i = 0;
    while i + 4 <= N {
        unsafe {
            let v = vld1q_f32(a.as_ptr().add(i));
            let mut e = vrecpeq_f32(v);
            e = vmulq_f32(e, vrecpsq_f32(v, e));
            e = vmulq_f32(e, vrecpsq_f32(v, e));
            vst1q_f32(out.as_mut_ptr().add(i), e);
        }
        i += 4;
    }
    while i < N {
        out[i] = 1.0 / a[i];
        i += 1;
    }
    Simd::from_array(out)
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn rcp_simd<const N: usize>(x: Simd<f32, N>) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    Simd::splat(1.0) / x
}


#[inline(always)]
pub fn exp<const N: usize>(x: Varying<f32, N>) -> Varying<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    Varying(exp_simd(x.0))
}

#[inline(always)]
pub fn log<const N: usize>(x: Varying<f32, N>) -> Varying<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    Varying(log_simd(x.0))
}

#[inline(always)]
pub fn pow<const N: usize>(a: Varying<f32, N>, b: Varying<f32, N>) -> Varying<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    Varying(exp_simd(b.0 * log_simd(a.0)))
}

#[inline(always)]
pub fn sin<const N: usize>(x: Varying<f32, N>) -> Varying<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    Varying(sincos_simd(x.0, false))
}

#[inline(always)]
pub fn cos<const N: usize>(x: Varying<f32, N>) -> Varying<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    Varying(sincos_simd(x.0, true))
}


#[inline(always)]
fn exp_simd<const N: usize>(x_full: Simd<f32, N>) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    const LN2_PART1: f32 = 0.6931457519;
    const LN2_PART2: f32 = 1.4286067653e-6;
    const ONE_OVER_LN2: f32 = 1.44269502162933349609375;

    const C2: f32 = 0.4999999105930328369140625;
    const C3: f32 = 0.166668415069580078125;
    const C4: f32 = 4.16539050638675689697265625e-2;
    const C5: f32 = 8.378830738365650177001953125e-3;
    const C6: f32 = 1.304379315115511417388916015625e-3;
    const C7: f32 = 2.7555381529964506626129150390625e-4;

    let scaled = x_full * Simd::splat(ONE_OVER_LN2);
    let k_real = scaled.floor();
    let k = k_real.cast::<i32>();

    let x = x_full - k_real * Simd::splat(LN2_PART1);
    let x = x - k_real * Simd::splat(LN2_PART2);

    let mut result = x.mul_add(Simd::splat(C7), Simd::splat(C6));
    result = x.mul_add(result, Simd::splat(C5));
    result = x.mul_add(result, Simd::splat(C4));
    result = x.mul_add(result, Simd::splat(C3));
    result = x.mul_add(result, Simd::splat(C2));
    result = x.mul_add(result, Simd::splat(1.0));
    result = x.mul_add(result, Simd::splat(1.0));

    let biased_n = (k + Simd::splat(127)) << Simd::splat(23);
    let two_to_the_n = Simd::<f32, N>::from_bits(biased_n.cast::<u32>());
    result *= two_to_the_n;

    let overflow = k_real.simd_gt(Simd::splat(127.0));
    let underflow = k_real.simd_le(Simd::splat(-127.0));
    result = overflow.select(Simd::splat(f32::INFINITY), result);
    underflow.select(Simd::splat(0.0), result)
}


#[inline(always)]
fn range_reduce_log<const N: usize>(input: Simd<f32, N>) -> (Simd<f32, N>, Simd<i32, N>)
where
    LaneCount<N>: SupportedLaneCount,
{
    const NONEXPONENT_MASK: u32 = 0x807F_FFFF;
    const EXPONENT_NEG1: u32 = 126 << 23; 

    let int_version = input.to_bits(); 
    let biased_exponent = (int_version >> Simd::splat(23)).cast::<i32>(); 
    let offset_exponent = biased_exponent + Simd::splat(1);
    let exponent = offset_exponent - Simd::splat(127);
    let blended =
        (int_version & Simd::splat(NONEXPONENT_MASK)) | Simd::splat(EXPONENT_NEG1);
    let reduced = Simd::<f32, N>::from_bits(blended);
    (reduced, exponent)
}

#[inline(always)]
fn log_simd<const N: usize>(x_full: Simd<f32, N>) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    const LN2: f32 = 6.931471825e-01;

    const C02: f32 = -4.9999991060e-01;
    const C03: f32 = 3.3335432410e-01;
    const C04: f32 = -2.4996809660e-01;
    const C05: f32 = 1.9873061776e-01;
    const C06: f32 = -1.6905537250e-01;
    const C07: f32 = 1.6525325180e-01;
    const C08: f32 = -7.3399633170e-02;
    const C09: f32 = -4.0101176130e-03;
    const C10: f32 = -4.4349682330e-01;

    let x_repr = x_full.to_bits(); 

    const UNSTABLE_RANGE_SIZE: f32 = 0.285;
    let start_repr: u32 = 1.0f32.to_bits();
    let end_repr: u32 = (1.0f32 + UNSTABLE_RANGE_SIZE).to_bits() + 1;
    let unstable_range_ulp_size: u32 = end_repr - start_repr;
    let close_to_one =
        (x_repr - Simd::splat(start_repr)).simd_lt(Simd::splat(unstable_range_ulp_size));

    let (reduced, exponent) = range_reduce_log(x_full);
    let reduced = close_to_one.select(x_full, reduced);
    let scale = close_to_one.select(
        Simd::splat(0.0),
        exponent.cast::<f32>() * Simd::splat(LN2),
    );

    let x = reduced - Simd::splat(1.0);
    let x2 = x.mul_add(x, Simd::splat(1e-30));

    let mut result = x.mul_add(Simd::splat(C10), Simd::splat(C09));
    result = x2.mul_add(result, x.mul_add(Simd::splat(C08), Simd::splat(C07)));
    result = x2.mul_add(result, x.mul_add(Simd::splat(C06), Simd::splat(C05)));
    result = x2.mul_add(result, x.mul_add(Simd::splat(C04), Simd::splat(C03)));
    result = x2.mul_add(result, x.mul_add(Simd::splat(C02), Simd::splat(1.0)));
    x.mul_add(result, scale)
}


#[inline(always)]
fn sincos_simd<const N: usize>(x_full: Simd<f32, N>, want_cos: bool) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    const PI_OVER_TWO: f32 = 1.57079637050628662109375;
    const TWO_OVER_PI: f32 = 0.636619746685028076171875;

    const SIN_C2: f32 = -0.16666667163372039794921875;
    const SIN_C4: f32 = 8.333347737789154052734375e-3;
    const SIN_C6: f32 = -1.9842604524455964565277099609375e-4;
    const SIN_C8: f32 = 2.760012648650445044040679931640625e-6;
    const SIN_C10: f32 = -2.50293279435709337121807038784027099609375e-8;

    const COS_C2: f32 = -0.5;
    const COS_C4: f32 = 4.166664183139801025390625e-2;
    const COS_C6: f32 = -1.388833043165504932403564453125e-3;
    const COS_C8: f32 = 2.47562347794882953166961669921875e-5;
    const COS_C10: f32 = -2.59630184018533327616751194000244140625e-7;

    let scaled = x_full * Simd::splat(TWO_OVER_PI);
    let k_real = scaled.floor();
    let k = k_real.cast::<i32>();

    let x = x_full - k_real * Simd::splat(PI_OVER_TWO);
    let k_mod4 = k & Simd::splat(3);

    let (usecos, flip_sign) = if want_cos {
        let usecos = k_mod4.simd_eq(Simd::splat(0)) | k_mod4.simd_eq(Simd::splat(2));
        let flip = k_mod4.simd_eq(Simd::splat(1)) | k_mod4.simd_eq(Simd::splat(2));
        (usecos, flip)
    } else {
        let usecos = k_mod4.simd_eq(Simd::splat(1)) | k_mod4.simd_eq(Simd::splat(3));
        let flip = k_mod4.simd_gt(Simd::splat(1));
        (usecos, flip)
    };

    let outside = usecos.select(Simd::splat(1.0), x);
    let c2 = usecos.select(Simd::splat(COS_C2), Simd::splat(SIN_C2));
    let c4 = usecos.select(Simd::splat(COS_C4), Simd::splat(SIN_C4));
    let c6 = usecos.select(Simd::splat(COS_C6), Simd::splat(SIN_C6));
    let c8 = usecos.select(Simd::splat(COS_C8), Simd::splat(SIN_C8));
    let c10 = usecos.select(Simd::splat(COS_C10), Simd::splat(SIN_C10));

    let x2 = x * x;
    let mut formula = x2.mul_add(c10, c8);
    formula = x2.mul_add(formula, c6);
    formula = x2.mul_add(formula, c4);
    formula = x2.mul_add(formula, c2);
    formula = x2.mul_add(formula, Simd::splat(1.0));
    formula *= outside;

    flip_sign.select(-formula, formula)
}


#[cfg(test)]
mod tests {
    use super::*;

    const N: usize = 8;
    type Vf = Varying<f32, N>;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }


    #[test]
    fn thin_wrappers_f32() {
        let v = Vf::from_array([1.0, 4.0, 9.0, 16.0, 25.0, 0.25, 100.0, 2.0]);
        for (g, r) in sqrt(v).to_array().iter().zip(v.to_array().iter()) {
            assert!(approx(*g, r.sqrt(), 1e-6));
        }
        let n = Varying::<f32, N>::from_array([-1.5, 2.5, -3.0, 4.0, -0.0, 7.0, -8.0, 9.0]);
        assert_eq!(abs(n).to_array(), [1.5, 2.5, 3.0, 4.0, 0.0, 7.0, 8.0, 9.0]);

        let a = Vf::from_array([1.0, 5.0, 3.0, 8.0, 2.0, 6.0, 4.0, 7.0]);
        let b = Vf::splat(4.0);
        assert_eq!(min(a, b).to_array(), [1.0, 4.0, 3.0, 4.0, 2.0, 4.0, 4.0, 4.0]);
        assert_eq!(max(a, b).to_array(), [4.0, 5.0, 4.0, 8.0, 4.0, 6.0, 4.0, 7.0]);

        let f = Vf::from_array([1.2, -1.2, 2.7, -2.7, 3.5, -3.5, 0.9, -0.4]);
        assert_eq!(floor(f).to_array(), [1.0, -2.0, 2.0, -3.0, 3.0, -4.0, 0.0, -1.0]);
        assert_eq!(ceil(f).to_array(), [2.0, -1.0, 3.0, -2.0, 4.0, -3.0, 1.0, -0.0]);
        assert_eq!(round(f).to_array(), [1.0, -1.0, 3.0, -3.0, 4.0, -4.0, 1.0, -0.0]);

        let c = clamp(a, Vf::splat(3.0), Vf::splat(6.0));
        assert_eq!(c.to_array(), [3.0, 5.0, 3.0, 6.0, 3.0, 6.0, 4.0, 6.0]);

        let l = lerp(Vf::splat(10.0), Vf::splat(20.0), Vf::splat(0.25));
        assert_eq!(l.to_array(), [12.5; N]);
    }

    #[test]
    fn min_max_clamp_integers() {
        type Vi = Varying<i32, N>;
        let v = Vi::from_array([-3, 0, 5, 63, 64, 100, -1, 7]);
        let c = clamp(v, Vi::splat(0), Vi::splat(63));
        assert_eq!(c.to_array(), [0, 0, 5, 63, 63, 63, 0, 7]);
        assert_eq!(min(v, Vi::splat(5)).to_array(), [-3, 0, 5, 5, 5, 5, -1, 5]);
        assert_eq!(max(v, Vi::splat(5)).to_array(), [5, 5, 5, 63, 64, 100, 5, 7]);
        type Vu = Varying<u32, N>;
        let u = Vu::from_array([0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(clamp(u, Vu::splat(2), Vu::splat(5)).to_array(), [2, 2, 2, 3, 4, 5, 5, 5]);
    }

    #[test]
    fn thin_wrappers_f64_trivial() {
        let v = Varying::<f64, N>::from_array([1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0]);
        assert_eq!(
            sqrt(v).to_array(),
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );
        let a = Varying::<f64, N>::splat(-2.5);
        assert_eq!(abs(a).to_array(), [2.5; N]);
        let lo = Varying::<f64, N>::splat(0.0);
        let hi = Varying::<f64, N>::splat(3.0);
        assert_eq!(clamp(v, lo, hi).to_array(), [1.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]);
    }


    #[test]
    fn rsqrt_is_within_a_few_ulp() {
        let xs = [0.5f32, 1.0, 2.0, 3.0, 7.0, 100.0, 0.01, 1e6];
        let v = Vf::from_array(xs);
        let got = rsqrt(v).to_array();
        let mut max_ulp = 0.0f32;
        for (g, x) in got.iter().zip(xs.iter()) {
            let want = 1.0 / x.sqrt();
            let ulp = ((g - want).abs() / (want * f32::EPSILON)).abs();
            max_ulp = max_ulp.max(ulp);
        }
        assert!(max_ulp < 3.0, "rsqrt max ulp = {max_ulp}");
    }

    #[test]
    fn rcp_is_within_a_few_ulp() {
        let xs = [0.5f32, 1.0, 2.0, 3.0, 7.0, 100.0, 0.01, 1e6];
        let v = Vf::from_array(xs);
        let got = rcp(v).to_array();
        let mut max_ulp = 0.0f32;
        for (g, x) in got.iter().zip(xs.iter()) {
            let want = 1.0 / x;
            let ulp = ((g - want).abs() / (want * f32::EPSILON)).abs();
            max_ulp = max_ulp.max(ulp);
        }
        assert!(max_ulp < 3.0, "rcp max ulp = {max_ulp}");
    }

    #[test]
    fn rsqrt_odd_widths_use_fallback() {
        let v = Varying::<f32, 2>::from_array([4.0, 9.0]);
        let g = rsqrt(v).to_array();
        assert!(approx(g[0], 0.5, 1e-3) && approx(g[1], 1.0 / 3.0, 1e-3));
    }


    fn max_rel_err(
        xs: &[f32],
        f: impl Fn(Vf) -> Vf,
        r: impl Fn(f64) -> f64,
        rel_floor: f64,
    ) -> (f64, f64) {
        let mut max_rel = 0.0f64;
        let mut max_abs = 0.0f64;
        for chunk in xs.chunks(N) {
            let mut buf = [xs[0]; N];
            buf[..chunk.len()].copy_from_slice(chunk);
            let got = f(Varying::from_array(buf)).to_array();
            for (i, &x) in chunk.iter().enumerate() {
                let want = r(x as f64);
                let abs = (got[i] as f64 - want).abs();
                max_abs = max_abs.max(abs);
                if want.abs() > rel_floor {
                    max_rel = max_rel.max(abs / want.abs());
                }
            }
        }
        (max_rel, max_abs)
    }

    fn linspace(lo: f32, hi: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| lo + (hi - lo) * (i as f32) / (n as f32 - 1.0))
            .collect()
    }

    fn geomspace(lo: f32, hi: f32, n: usize) -> Vec<f32> {
        let (llo, lhi) = (lo.ln(), hi.ln());
        (0..n)
            .map(|i| (llo + (lhi - llo) * (i as f32) / (n as f32 - 1.0)).exp())
            .collect()
    }

    #[test]
    fn exp_accuracy() {
        let xs = linspace(-87.0, 87.0, 4001);
        let (rel, _abs) = max_rel_err(&xs, exp, f64::exp, 1e-300);
        eprintln!("exp max rel err over [-87,87] = {rel:e}");
        assert!(rel < 1e-6, "exp rel err {rel:e} !< 1e-6");
    }

    #[test]
    fn log_accuracy() {
        let mut xs = geomspace(1e-30, 1e30, 6001);
        xs.extend(linspace(0.5, 2.0, 2000));
        let (rel, _abs) = max_rel_err(&xs, log, f64::ln, 1e-4);
        eprintln!("log max rel err over (1e-30,1e30] = {rel:e}");
        assert!(rel < 1e-6, "log rel err {rel:e} !< 1e-6");
    }

    #[test]
    fn sin_cos_accuracy() {
        let pi = std::f32::consts::PI;
        let xs = linspace(-4.0 * pi, 4.0 * pi, 8001);
        let (_sr, sa) = max_rel_err(&xs, sin, f64::sin, f64::INFINITY);
        let (_cr, ca) = max_rel_err(&xs, cos, f64::cos, f64::INFINITY);
        eprintln!("sin max abs err over [-4pi,4pi] = {sa:e}");
        eprintln!("cos max abs err over [-4pi,4pi] = {ca:e}");
        assert!(sa < 1e-6, "sin abs err {sa:e}");
        assert!(ca < 1e-6, "cos abs err {ca:e}");
    }

    #[test]
    fn pow_spot_checks() {
        let a = Vf::from_array([2.0, 3.0, 4.0, 9.0, 10.0, 5.0, 2.0, 1.5]);
        let b = Vf::from_array([10.0, 0.5, 0.5, 0.5, 3.0, 2.0, -1.0, 4.0]);
        let got = pow(a, b).to_array();
        let av = a.to_array();
        let bv = b.to_array();
        let mut max_rel = 0.0f64;
        for i in 0..N {
            let want = (av[i] as f64).powf(bv[i] as f64);
            max_rel = max_rel.max((got[i] as f64 - want).abs() / want.abs());
        }
        eprintln!("pow max rel err (spot) = {max_rel:e}");
        assert!(max_rel < 1e-5, "pow rel err {max_rel:e}");
    }

    #[test]
    fn exp_log_roundtrip() {
        let xs = geomspace(1e-6, 1e6, 40);
        let v = Vf::from_array(core::array::from_fn(|i| xs[i % xs.len()]));
        let back = exp(log(v)).to_array();
        for (g, x) in back.iter().zip(v.to_array().iter()) {
            assert!(approx(g / x, 1.0, 1e-5), "roundtrip {g} vs {x}");
        }
    }
}
