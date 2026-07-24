#![feature(portable_simd)]

use core::simd::Mask;
use rustlane_core::prelude::*;

const N: usize = 4;
type Vf = Varying<f32, N>;
type Vi = Varying<i32, N>;


fn foreach_scale(a: &[f32], out: &mut [f32]) {
    let __exec = AllOn;
    let _ = __exec;
    {
        let __n = a.len();
        let mut __base = 0usize;
        while __base + N <= __n {
            let __exec = AllOn;
            let i = LinearIndex::<N>::new(__base);
            let x = a.spmd_read(i, __exec);
            out.spmd_write(i, __exec, x * 2.0f32 + 1.0f32);
            __base += N;
        }
        if __base < __n {
            let __exec = VMask::<N>::first(__n - __base);
            let i = LinearIndex::<N>::new(__base);
            let x = a.spmd_read(i, __exec);
            out.spmd_write(i, __exec, x * 2.0f32 + 1.0f32);
        }
    }
}

#[test]
fn foreach_template_main_and_tail() {
    let a: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let mut out = vec![0.0f32; 6];
    foreach_scale(&a, &mut out);
    let want: Vec<f32> = a.iter().map(|x| x * 2.0 + 1.0).collect();
    assert_eq!(out, want);
}


fn if_else_compound(x: Vf, t: f32) -> (Vf, Vf) {
    let __exec = AllOn;
    let mut y = Varying::<f32, N>::splat(1.0);
    let mut z = Varying::<f32, N>::splat(0.0);
    {
        let __c = x.spmd_gt(t);
        let __exec1 = __exec.and_cond(__c);
        if __exec1.should_branch() {
            y.masked_assign(__exec1, y + x);
        }
        let __exec1 = __exec.and_not_cond(__c);
        if __exec1.should_branch() {
            y.masked_assign(__exec1, 0.0f32);
        }
    }
    {
        let __exec = AllOn;
        z.masked_assign(__exec, y + 1.0f32);
    }
    (y, z)
}

#[test]
fn if_else_and_unmasked_templates() {
    let x = Vf::from_array([1.0, 3.0, 0.5, 4.0]);
    let (y, z) = if_else_compound(x, 2.0);
    assert_eq!(y.to_array(), [0.0, 4.0, 0.0, 5.0]);
    assert_eq!(z.to_array(), [1.0, 5.0, 1.0, 6.0]);
}

fn if_else_uniform(flag: bool) -> Vf {
    let __exec = AllOn;
    let mut y = Varying::<f32, N>::splat(0.0);
    {
        let __c = flag;
        let __exec1 = __exec.and_cond(__c);
        if __exec1.should_branch() {
            y.masked_assign(__exec1, 1.0f32);
        }
        let __exec1 = __exec.and_not_cond(__c);
        if __exec1.should_branch() {
            y.masked_assign(__exec1, -1.0f32);
        }
    }
    y
}

#[test]
fn if_else_uniform_template() {
    assert_eq!(if_else_uniform(true).to_array(), [1.0; N]);
    assert_eq!(if_else_uniform(false).to_array(), [-1.0; N]);
}


fn masked_scatter_increment(hist: &mut [i32], idx: Vi, sel: Vi) {
    let __exec = AllOn;
    {
        let __c = sel.spmd_gt(0);
        let __exec1 = __exec.and_cond(__c);
        if __exec1.should_branch() {
            let __t = hist.spmd_read(idx, __exec1) + 1;
            hist.spmd_write(idx, __exec1, __t);
        }
    }
}

#[test]
fn gather_modify_scatter_template() {
    let mut hist = [10i32, 20, 30, 40, 50];
    masked_scatter_increment(&mut hist, Vi::from_array([4, 0, 2, 1]), Vi::from_array([1, 0, 5, -1]));
    assert_eq!(hist, [10, 20, 31, 40, 51]);
}


#[test]
fn logical_and_cast_and_literal_index_emissions() {
    let p = Vf::from_array([0.5, 1.5, 2.5, 3.5]);
    let inside = p.spmd_ge(1.0f32).spmd_and(|| p.spmd_le(3.0f32));
    assert_eq!(inside.to_array(), [false, true, true, false]);
    let outside = inside.spmd_not();
    assert_eq!(outside.to_array(), [true, false, false, true]);
    let m = true.spmd_and(|| p.spmd_gt(2.0f32));
    assert_eq!(m.to_array(), [false, false, true, true]);

    let v: Varying<i32, N> = SpmdCast::<i32>::spmd_cast(p);
    assert_eq!(v.to_array(), [0, 1, 2, 3]);
    let u: f32 = SpmdCast::<f32>::spmd_cast(3i32);
    assert_eq!(u, 3.0);

    let coef = [2.0f32, 0.5, 0.25, 0.125];
    let c0 = coef.spmd_read(0, AllOn);
    assert_eq!(c0, 2.0);

    let xoffsets = [0i32, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3];
    let o = LinearIndex::<N>::new(4);
    let xo = xoffsets.spmd_read(o, AllOn);
    assert_eq!(xo.to_array(), [2, 3, 2, 3]);
    let xg = xoffsets.spmd_read(Vi::from_array([0, 5, 10, 15]), AllOn);
    assert_eq!(xg.to_array(), [0, 3, 0, 3]);
}


fn stencil_row(a: &[f32], out: &mut [f32], base: usize, c0: f32, c1: f32) {
    let __exec = AllOn;
    let index = LinearIndex::<N>::new(base);
    let center = a.spmd_read(index, __exec);
    let right = a.spmd_read(index + 1, __exec);
    let left = a.spmd_read(index + (-1), __exec);
    out.spmd_write(index, __exec, center * c0 + (right + left) * c1);
}

#[test]
fn stencil_linear_offsets() {
    let a: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let mut out = vec![0.0f32; 8];
    stencil_row(&a, &mut out, 2, 2.0, 0.5);
    for i in 2..6 {
        let want = a[i] * 2.0 + (a[i + 1] + a[i - 1]) * 0.5;
        assert_eq!(out[i], want, "element {i}");
    }
    let li = LinearIndex::<N>::new(2);
    let demoted: Vi = li + Vi::splat(0);
    let g = a.spmd_read(demoted, AllOn);
    let l = a.spmd_read(li, AllOn);
    assert_eq!(g.to_array(), l.to_array());
}


const STEPS: usize = 8;

fn binomial_shape_allon(s: Vf) -> Vf {
    let __exec = AllOn;
    let mut v = [Varying::<f32, N>::splat(0.0); STEPS];
    for j in 0..STEPS {
        v.spmd_write(j, __exec, s * (j as f32));
    }
    for j in (0..STEPS - 1).rev() {
        for k in 0..=j {
            let t = (v.spmd_read(k, __exec) + v.spmd_read(k + 1, __exec)) * 0.5f32;
            v.spmd_write(k, __exec, t);
        }
    }
    v.spmd_read(0, __exec)
}

fn binomial_shape_masked(s: Vf, active: usize) -> Vf {
    let __exec = VMask::<N>::first(active);
    let mut v = [Varying::<f32, N>::splat(0.0); STEPS];
    for j in 0..STEPS {
        v.spmd_write(j, __exec, s * (j as f32));
    }
    for j in (0..STEPS - 1).rev() {
        for k in 0..=j {
            let t = (v.spmd_read(k, __exec) + v.spmd_read(k + 1, __exec)) * 0.5f32;
            v.spmd_write(k, __exec, t);
        }
    }
    v.spmd_read(0, __exec)
}

#[test]
fn binomial_local_varying_array() {
    let s = Vf::from_array([1.0, 2.0, 3.0, 4.0]);
    let full = binomial_shape_allon(s);
    let tail = binomial_shape_masked(s, 2);
    let f = full.to_array();
    let t = tail.to_array();
    assert_eq!(t[0], f[0]);
    assert_eq!(t[1], f[1]);
    assert_eq!(t[2], 0.0);
    assert_eq!(t[3], 0.0);
    for lane in 0..N {
        let mut v = [0.0f32; STEPS];
        for (j, slot) in v.iter_mut().enumerate() {
            *slot = s.to_array()[lane] * (j as f32);
        }
        for j in (0..STEPS - 1).rev() {
            for k in 0..=j {
                v[k] = 0.5 * (v[k] + v[k + 1]);
            }
        }
        assert_eq!(f[lane], v[0], "lane {lane}");
    }
}


fn tri_hit_shape(b1: Vf, t: Vf, mint: f32, maxt: f32) -> Mask<i32, N> {
    let __exec = AllOn;
    let mut hit: Mask<i32, N> = Mask::splat(true);
    {
        let __c = b1.spmd_lt(0.0f32).spmd_or(|| b1.spmd_gt(1.0f32));
        let __exec1 = __exec.and_cond(__c);
        if __exec1.should_branch() {
            hit.masked_assign(__exec1, false);
        }
    }
    {
        let __c = t.spmd_lt(mint).spmd_or(|| t.spmd_gt(maxt));
        let __exec1 = __exec.and_cond(__c);
        if __exec1.should_branch() {
            hit.masked_assign(__exec1, false);
        }
    }
    hit
}

#[test]
fn varying_bool_local_mask_lvalue() {
    let b1 = Vf::from_array([0.5, -0.1, 0.9, 1.2]);
    let t = Vf::from_array([1.0, 1.0, 99.0, 1.0]);
    let hit = tri_hit_shape(b1, t, 0.0, 10.0);
    assert_eq!(hit.to_array(), [true, false, false, false]);

    fn intersect_shape(t0: Vf, t1: Vf) -> Mask<i32, N> {
        let __exec = AllOn;
        let mut __ret: Mask<i32, N> = Default::default();
        let mut __fn = EnterLoopN::<N>::enter_loop_n(__exec);
        {
            let __c = t0.spmd_le(t1);
            let __exec1 = __exec.and_cond(__c);
            if __exec1.should_branch() {
                __ret.masked_assign(__exec1, true);
                if __exec1.is_statically_uniform() {
                    return __ret;
                }
                __fn.remove(__exec1);
            }
        }
        let __exec = __exec.refresh(&__fn);
        __ret.masked_assign(__exec, false);
        __ret
    }
    let r = intersect_shape(Vf::from_array([0.0, 5.0, 1.0, 9.0]), Vf::splat(4.0));
    assert_eq!(r.to_array(), [true, false, true, false]);
}


fn raymarch_shape(density: &[f32], nx: i32, ny: i32, t1: Vf, cut: f32, vx: Vi, vy: Vi, vz: Vi) -> Vf {
    let __exec = AllOn;
    let mut tau = Varying::<f32, N>::splat(0.0);
    let mut t = Varying::<f32, N>::splat(0.0);

    let x = math::clamp(vx, Vi::splat(0), Vi::splat(nx - 1));
    let y = math::clamp(vy, Vi::splat(0), Vi::splat(ny - 1));
    let z = math::clamp(vz, Vi::splat(0), Vi::splat(1));
    let idx = z * (nx * ny) + y * nx + x;

    {
        let mut __loop = __exec.enter_loop(t.spmd_lt(t1));
        loop {
            if !__loop.any() {
                break;
            }
            let __exec = __loop.current();
            {
                let __c = tau.spmd_gt(cut);
                let __exec1 = __exec.and_cond(__c);
                if __exec1.should_branch() {
                    if __exec1.is_statically_uniform() {
                        break;
                    }
                    __loop.remove(__exec1);
                }
            }
            let __exec = __exec.refresh(&__loop);
            tau.masked_assign(__exec, tau + density.spmd_read(idx, __exec));
            t.masked_assign(__exec, t + 1.0f32);
            let __c = t.spmd_lt(t1);
            __loop = __loop.and_cond(__c);
        }
    }
    math::exp(-tau)
}

#[test]
fn volume_while_break_clamp_gather() {
    let density: Vec<f32> = (0..16).map(|i| 0.1 * i as f32).collect();
    let (nx, ny) = (4, 2);
    let vx = Vi::from_array([-1, 1, 5, 2]);
    let vy = Vi::from_array([0, 3, 1, 0]);
    let vz = Vi::from_array([0, 0, 7, 1]);
    let t1 = Vf::from_array([0.0, 1.0, 2.0, 3.0]);
    let got = raymarch_shape(&density, nx, ny, t1, 1.0e9, vx, vy, vz);

    let cl = |v: i32, hi: i32| v.clamp(0, hi);
    for lane in 0..N {
        let x = cl(vx.to_array()[lane], nx - 1);
        let y = cl(vy.to_array()[lane], ny - 1);
        let z = cl(vz.to_array()[lane], 1);
        let d = density[(z * nx * ny + y * nx + x) as usize];
        let steps = t1.to_array()[lane] as i32;
        let tau = d * steps.max(0) as f32;
        let want = (-tau).exp();
        let g = got.to_array()[lane];
        assert!((g - want).abs() < 1e-5, "lane {lane}: {g} vs {want}");
    }
}


#[test]
fn ao_scatter_add_shape() {
    let mut image = [0.0f32; 2];
    let offset = Vi::from_array([0, 0, 1, 1]);
    let ret = Vf::from_array([0.25, 0.5, 1.0, 2.0]);
    {
        let __exec = AllOn;
        memory::scatter_add(&mut image, offset, __exec, ret);
    }
    assert_eq!(image, [0.75, 3.0]);

    let mut image = [0.0f32; 2];
    let m = VMask::<N>(Mask::from_array([true, false, true, false]));
    memory::scatter_add(&mut image, offset, m, ret);
    assert_eq!(image, [0.25, 1.0]);
}


#[test]
fn ao_rng_math_shape() {
    let y0 = 3i32;
    let pi_lane = reduce::lanes_iota::<N>();
    let seed_i = pi_lane + (y0 << (pi_lane & 15));
    let seeds: Varying<u32, N> = SpmdCast::<u32>::spmd_cast(seed_i);
    let mut rng = rng::RNGState::<N>::new(seeds);

    let theta = math::sqrt(rng.frandom());
    let phi = rng.frandom() * (2.0f32 * core::f32::consts::PI);
    let x = math::cos(phi) * theta;
    let y = math::sin(phi) * theta;
    let z = math::sqrt(math::abs(
        Varying::splat(1.0f32) - theta * theta,
    ));
    let len2 = x * x + y * y + z * z;
    for l in len2.to_array() {
        assert!((l - 1.0).abs() < 1e-4, "unit dir, got {l}");
    }
}
