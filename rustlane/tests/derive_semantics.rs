#![feature(portable_simd)]

use core::simd::Mask;
use rustlane::{kernel, AllOn, MaskedAssign, SpmdValue, VMask, Varying};

#[derive(SpmdValue, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(SpmdValue, Clone, Copy)]
#[repr(C)]
struct Ray {
    origin: Vec3,
    dir: Vec3,
}

#[derive(SpmdValue, Clone, Copy)]
#[repr(C)]
struct Hit {
    t: f32,
    #[spmd(uniform)]
    id: i32,
}

#[kernel]
impl Vec3 {
    fn dot(a: VaryingVec3<N>, b: VaryingVec3<N>) -> Varying<f32> {
        a.x * b.x + a.y * b.y + a.z * b.z
    }

    fn cross(a: VaryingVec3<N>, b: VaryingVec3<N>) -> VaryingVec3<N> {
        VaryingVec3 {
            x: a.y * b.z - a.z * b.y,
            y: a.z * b.x - a.x * b.z,
            z: a.x * b.y - a.y * b.x,
        }
    }

    fn normalize(a: VaryingVec3<N>) -> VaryingVec3<N> {
        let inv = 1.0 / math::sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
        VaryingVec3 {
            x: a.x * inv,
            y: a.y * inv,
            z: a.z * inv,
        }
    }
}

#[kernel]
fn choose_dot(
    sel: Varying<f32>,
    a: VaryingVec3<N>,
    b: VaryingVec3<N>,
    c: VaryingVec3<N>,
) -> Varying<f32> {
    let mut p = a;
    if sel > 1.0 {
        p = b;
    } else {
        if sel > 0.0 {
            p = c;
        }
    }
    p.x * p.x + p.y * p.y + p.z * p.z
}

fn ref_dot(a: Vec3, b: Vec3) -> f32 {
    a.x * b.x + a.y * b.y + a.z * b.z
}
fn ref_cross(a: Vec3, b: Vec3) -> Vec3 {
    Vec3 {
        x: a.y * b.z - a.z * b.y,
        y: a.z * b.x - a.x * b.z,
        z: a.x * b.y - a.y * b.x,
    }
}
fn ref_normalize(a: Vec3) -> Vec3 {
    let inv = 1.0 / (a.x * a.x + a.y * a.y + a.z * a.z).sqrt();
    Vec3 {
        x: a.x * inv,
        y: a.y * inv,
        z: a.z * inv,
    }
}
fn ref_choose_dot(sel: f32, a: Vec3, b: Vec3, c: Vec3) -> f32 {
    let p = if sel > 1.0 {
        b
    } else if sel > 0.0 {
        c
    } else {
        a
    };
    ref_dot(p, p)
}

fn v3_lanes<const N: usize>(f: impl Fn(usize) -> Vec3) -> VaryingVec3<N>
where
    core::simd::LaneCount<N>: core::simd::SupportedLaneCount,
{
    VaryingVec3 {
        x: Varying::from_array(core::array::from_fn(|l| f(l).x)),
        y: Varying::from_array(core::array::from_fn(|l| f(l).y)),
        z: Varying::from_array(core::array::from_fn(|l| f(l).z)),
    }
}
fn v3_lane<const N: usize>(v: VaryingVec3<N>, l: usize) -> Vec3
where
    core::simd::LaneCount<N>: core::simd::SupportedLaneCount,
{
    Vec3 {
        x: v.x.to_array()[l],
        y: v.y.to_array()[l],
        z: v.z.to_array()[l],
    }
}

fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1e-5 * (1.0 + a.abs().max(b.abs()))
}
fn approx_v3(a: Vec3, b: Vec3) -> bool {
    approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z)
}

fn sample(l: usize) -> Vec3 {
    let f = l as f32;
    Vec3 {
        x: 1.0 + f,
        y: 2.0 - 0.5 * f,
        z: -1.0 + 0.25 * f,
    }
}
fn sample2(l: usize) -> Vec3 {
    let f = l as f32;
    Vec3 {
        x: 0.5 * f - 1.0,
        y: 3.0 - f,
        z: 2.0 + 0.5 * f,
    }
}

#[test]
fn methods_dot_cross_normalize() {
    let a8 = v3_lanes::<8>(sample);
    let b8 = v3_lanes::<8>(sample2);
    let d8 = Vec3::dot::<8, _>(AllOn, a8, b8);
    let c8 = Vec3::cross::<8, _>(AllOn, a8, b8);
    let n8 = Vec3::normalize::<8, _>(AllOn, a8);
    for l in 0..8 {
        assert!(
            approx(d8.to_array()[l], ref_dot(sample(l), sample2(l))),
            "dot lane {l}"
        );
        assert!(
            approx_v3(v3_lane(c8, l), ref_cross(sample(l), sample2(l))),
            "cross lane {l}"
        );
        assert!(
            approx_v3(v3_lane(n8, l), ref_normalize(sample(l))),
            "normalize lane {l}"
        );
    }
    for l in 0..8 {
        let a1 = v3_lanes::<1>(|_| sample(l));
        let b1 = v3_lanes::<1>(|_| sample2(l));
        assert_eq!(
            Vec3::dot::<1, _>(AllOn, a1, b1).to_array()[0],
            d8.to_array()[l],
            "dot N=1 vs N=8 lane {l}"
        );
        assert_eq!(
            v3_lane(Vec3::cross::<1, _>(AllOn, a1, b1), 0),
            v3_lane(c8, l)
        );
        assert_eq!(
            v3_lane(Vec3::normalize::<1, _>(AllOn, a1), 0),
            v3_lane(n8, l)
        );
    }
}

#[test]
fn nested_varying_control_flow() {
    let third = |l: usize| Vec3 {
        x: l as f32,
        y: 1.0,
        z: 2.0,
    };
    let sels = [-1.0f32, 0.5, 1.5, 2.0, 0.0, 0.9, 1.1, -0.3];
    let a8 = v3_lanes::<8>(sample);
    let b8 = v3_lanes::<8>(sample2);
    let c8 = v3_lanes::<8>(third);
    let r8 = choose_dot::<8, _>(AllOn, Varying::from_array(sels), a8, b8, c8);
    for l in 0..8 {
        let want = ref_choose_dot(sels[l], sample(l), sample2(l), third(l));
        assert!(approx(r8.to_array()[l], want), "choose_dot lane {l}");
    }
    for l in 0..8 {
        let a1 = v3_lanes::<1>(|_| sample(l));
        let b1 = v3_lanes::<1>(|_| sample2(l));
        let c1 = v3_lanes::<1>(|_| third(l));
        let r1 = choose_dot::<1, _>(AllOn, Varying::from_array([sels[l]]), a1, b1, c1);
        assert_eq!(
            r1.to_array()[0],
            r8.to_array()[l],
            "choose_dot N=1 vs N=8 lane {l}"
        );
    }
}

#[test]
fn aos_gather_vec3() {
    let data: Vec<Vec3> = (0..6)
        .map(|i| Vec3 {
            x: i as f32,
            y: 10.0 + i as f32,
            z: 20.0 + i as f32,
        })
        .collect();
    let idx8 = Varying::<i32, 8>::from_array([5, 0, 3, 1, 4, 2, 0, 5]);
    let g8 = VaryingVec3::gather(&data, idx8, AllOn);
    for l in 0..8 {
        assert_eq!(
            v3_lane(g8, l),
            data[idx8.to_array()[l] as usize],
            "gather lane {l}"
        );
    }
    let m = VMask::<8>(Mask::from_array([
        true, false, true, false, true, false, true, false,
    ]));
    let idxm = Varying::<i32, 8>::from_array([5, 999, 3, -1, 4, 12345, 0, -7]);
    let gm = VaryingVec3::gather(&data, idxm, m);
    for l in 0..8 {
        if l % 2 == 0 {
            assert_eq!(
                v3_lane(gm, l),
                data[idxm.to_array()[l] as usize],
                "masked gather lane {l}"
            );
        } else {
            assert_eq!(
                v3_lane(gm, l),
                Vec3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0
                },
                "masked-off lane {l}"
            );
        }
    }
    for l in 0..8 {
        let i1 = Varying::<i32, 1>::from_array([idx8.to_array()[l]]);
        assert_eq!(
            v3_lane(VaryingVec3::gather(&data, i1, AllOn), 0),
            v3_lane(g8, l)
        );
    }
}

#[test]
fn aos_gather_nested_ray() {
    let rays: Vec<Ray> = (0..4)
        .map(|i| Ray {
            origin: Vec3 {
                x: i as f32,
                y: i as f32 + 0.5,
                z: i as f32 + 0.25,
            },
            dir: Vec3 {
                x: -(i as f32),
                y: 1.0,
                z: 0.0,
            },
        })
        .collect();
    let idx = Varying::<i32, 8>::from_array([3, 0, 2, 1, 0, 3, 1, 2]);
    let g = VaryingRay::gather(&rays, idx, AllOn);
    for l in 0..8 {
        let r = rays[idx.to_array()[l] as usize];
        assert_eq!(v3_lane(g.origin, l), r.origin, "ray.origin lane {l}");
        assert_eq!(v3_lane(g.dir, l), r.dir, "ray.dir lane {l}");
    }
    for l in 0..8 {
        let i1 = Varying::<i32, 1>::from_array([idx.to_array()[l]]);
        let g1 = VaryingRay::gather(&rays, i1, AllOn);
        assert_eq!(v3_lane(g1.origin, 0), v3_lane(g.origin, l));
        assert_eq!(v3_lane(g1.dir, 0), v3_lane(g.dir, l));
    }
}

#[test]
fn mixed_struct_uniform_field() {
    let h = <Hit as SpmdValue>::splat::<8>(Hit { t: 2.5, id: 7 });
    assert_eq!(h.t.to_array(), [2.5; 8]);
    assert_eq!(h.id, 7);

    let src = VaryingHit {
        t: Varying::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        id: 42,
    };
    let mut hv = <Hit as SpmdValue>::splat::<8>(Hit { t: 0.0, id: 0 });
    MaskedAssign::masked_assign(&mut hv, AllOn, src);
    assert_eq!(hv.t.to_array(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert_eq!(hv.id, 42);

    let mut hv2 = <Hit as SpmdValue>::splat::<8>(Hit { t: 0.0, id: 1 });
    MaskedAssign::masked_assign(&mut hv2, rustlane::BoolGuard(true), src);
    assert_eq!(hv2.id, 42);
}
