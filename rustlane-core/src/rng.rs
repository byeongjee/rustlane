
use crate::varying::Varying;
use core::simd::num::SimdFloat;
use core::simd::{LaneCount, Simd, SupportedLaneCount};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RNGState<const N: usize>
where
    LaneCount<N>: SupportedLaneCount,
{
    pub z1: Varying<u32, N>,
    pub z2: Varying<u32, N>,
    pub z3: Varying<u32, N>,
    pub z4: Varying<u32, N>,
}

impl<const N: usize> RNGState<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline(always)]
    pub fn new(seeds: Varying<u32, N>) -> Self {
        let mut s = RNGState {
            z1: Varying::splat(0),
            z2: Varying::splat(0),
            z3: Varying::splat(0),
            z4: Varying::splat(0),
        };
        s.seed_rng(seeds);
        s
    }

    #[inline(always)]
    pub fn seed_rng(&mut self, seed: Varying<u32, N>) {
        self.z1 = seed;
        self.z2 = seed ^ 0xbeeff00d_u32;
        self.z3 = ((seed & 0x0000ffff_u32) << 16u32) | (seed >> 16u32);
        self.z4 = ((seed & 0x000000ff_u32) << 24u32)
            | ((seed & 0x0000ff00_u32) << 8u32)
            | ((seed & 0x00ff0000_u32) >> 8u32)
            | ((seed & 0xff000000_u32) >> 24u32);
    }

    #[inline(always)]
    pub fn random_u32(&mut self) -> Varying<u32, N> {
        let mut z1 = self.z1;
        let mut z2 = self.z2;
        let mut z3 = self.z3;
        let mut z4 = self.z4;

        let mut b = ((z1 << 6u32) ^ z1) >> 13u32;
        z1 = ((z1 & 0xfffffffe_u32) << 18u32) ^ b;
        b = ((z2 << 2u32) ^ z2) >> 27u32;
        z2 = ((z2 & 0xfffffff8_u32) << 2u32) ^ b;
        b = ((z3 << 13u32) ^ z3) >> 21u32;
        z3 = ((z3 & 0xfffffff0_u32) << 7u32) ^ b;
        b = ((z4 << 3u32) ^ z4) >> 12u32;
        z4 = ((z4 & 0xffffff80_u32) << 13u32) ^ b;

        self.z1 = z1;
        self.z2 = z2;
        self.z3 = z3;
        self.z4 = z4;
        z1 ^ z2 ^ z3 ^ z4
    }

    #[inline(always)]
    pub fn frandom(&mut self) -> Varying<f32, N> {
        let irand = self.random_u32() & 0x007fffff_u32;
        let bits = irand | 0x3f800000_u32;
        let f = <Simd<f32, N> as SimdFloat>::from_bits(bits.0);
        Varying(f) - 1.0f32
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    const VECTORS: &[(u32, [u32; 4], [u32; 3])] = &[
        (0x1, [1, 3203395596, 65536, 16777216], [0xfb3f5128, 0xbcff80e6, 0xbbb4200a]),
        (0x2a, [42, 3203395623, 2752512, 704643072], [0xee106289, 0x9aff5462, 0xee091f2b]),
        (
            0x12345678,
            [305419896, 2900076149, 1450709556, 2018915346],
            [0x1008d415, 0x58c11c4d, 0x8ebb09ea],
        ),
        (0x7, [7, 3203395594, 458752, 117440512], [0xf8243728, 0x50fcb1be, 0x7904ef6a]),
        (0x0, [0, 3203395597, 0, 0], [0xfbbfc028, 0xeeff00a2, 0xbbfc028a]),
    ];

    struct ScalarRng {
        z: [u32; 4],
    }
    impl ScalarRng {
        fn seed(seed: u32) -> Self {
            let z1 = seed;
            let z2 = seed ^ 0xbeeff00d;
            let z3 = ((seed & 0xffff) << 16) | (seed >> 16);
            let z4 = ((seed & 0xff) << 24)
                | ((seed & 0xff00) << 8)
                | ((seed & 0xff0000) >> 8)
                | ((seed & 0xff000000) >> 24);
            ScalarRng { z: [z1, z2, z3, z4] }
        }
        fn next(&mut self) -> u32 {
            let [mut z1, mut z2, mut z3, mut z4] = self.z;
            let mut b = ((z1 << 6) ^ z1) >> 13;
            z1 = ((z1 & 0xfffffffe) << 18) ^ b;
            b = ((z2 << 2) ^ z2) >> 27;
            z2 = ((z2 & 0xfffffff8) << 2) ^ b;
            b = ((z3 << 13) ^ z3) >> 21;
            z3 = ((z3 & 0xfffffff0) << 7) ^ b;
            b = ((z4 << 3) ^ z4) >> 12;
            z4 = ((z4 & 0xffffff80) << 13) ^ b;
            self.z = [z1, z2, z3, z4];
            z1 ^ z2 ^ z3 ^ z4
        }
    }

    fn frand_ref(u: u32) -> f32 {
        f32::from_bits(0x3f80_0000 | (u & 0x007f_ffff)) - 1.0
    }

    #[test]
    fn seed_matches_ispc_state() {
        for &(seed, state, _) in VECTORS {
            let s = RNGState::<8>::new(Varying::splat(seed));
            assert_eq!(s.z1.to_array(), [state[0]; 8], "z1 seed={seed:#x}");
            assert_eq!(s.z2.to_array(), [state[1]; 8], "z2 seed={seed:#x}");
            assert_eq!(s.z3.to_array(), [state[2]; 8], "z3 seed={seed:#x}");
            assert_eq!(s.z4.to_array(), [state[3]; 8], "z4 seed={seed:#x}");
        }
    }

    #[test]
    fn random_u32_bit_parity_n8() {
        for &(seed, _, outs) in VECTORS {
            let mut s = RNGState::<8>::new(Varying::splat(seed));
            for &expect in &outs {
                let got = s.random_u32();
                assert_eq!(got.to_array(), [expect; 8], "seed={seed:#x} expect={expect:#010x}");
            }
        }
    }

    #[test]
    fn random_u32_bit_parity_n1() {
        for &(seed, _, outs) in VECTORS {
            let mut s = RNGState::<1>::new(Varying::splat(seed));
            for &expect in &outs {
                assert_eq!(s.random_u32().to_array(), [expect], "seed={seed:#x}");
            }
        }
    }

    #[test]
    fn n1_agrees_with_n8_and_scalar() {
        for &(seed, _, _) in VECTORS {
            let mut s1 = RNGState::<1>::new(Varying::splat(seed));
            let mut s8 = RNGState::<8>::new(Varying::splat(seed));
            let mut sref = ScalarRng::seed(seed);
            for _ in 0..16 {
                let r1 = s1.random_u32().to_array()[0];
                let r8 = s8.random_u32().to_array();
                let rref = sref.next();
                assert_eq!(r1, rref, "N=1 vs scalar seed={seed:#x}");
                assert_eq!(r8, [rref; 8], "N=8 vs scalar seed={seed:#x}");
            }
        }
    }

    #[test]
    fn frandom_matches_reference_and_is_unit_interval() {
        for &(seed, _, _) in VECTORS {
            let mut s = RNGState::<8>::new(Varying::splat(seed));
            let mut sref = ScalarRng::seed(seed);
            for _ in 0..16 {
                let f = s.frandom().to_array();
                let expect = frand_ref(sref.next());
                for &lane in &f {
                    assert_eq!(lane, expect, "frandom seed={seed:#x}");
                    assert!((0.0..1.0).contains(&lane), "frandom out of [0,1): {lane}");
                }
            }
        }
    }

    #[test]
    fn varying_seeds_are_independent_per_lane() {
        let seeds = [1u32, 42, 0x12345678, 7, 0, 999, 0xdead_beef, 0x8000_0001];
        let mut s = RNGState::<8>::new(Varying::from_array(seeds));
        let mut refs: [ScalarRng; 8] = core::array::from_fn(|i| ScalarRng::seed(seeds[i]));
        for _ in 0..16 {
            let got = s.random_u32().to_array();
            let want: [u32; 8] = core::array::from_fn(|i| refs[i].next());
            assert_eq!(got, want);
        }
    }
}
