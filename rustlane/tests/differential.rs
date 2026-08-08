#![feature(portable_simd)]

use rustlane::kernel;
use rustlane::prelude::*;

struct XorShift(u32);

impl XorShift {
    fn new(seed: u32) -> Self {
        XorShift(if seed == 0 { 0x9E37_79B9 } else { seed })
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }

    fn next_i32(&mut self, range: i32) -> i32 {
        let span = (range as u32) * 2 + 1;
        (self.next_u32() % span) as i32 - range
    }

    fn next_f32(&mut self, lo: f32, hi: f32) -> f32 {
        let unit = (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32;
        lo + unit * (hi - lo)
    }
}

const BATCHES: usize = 64;

fn ulp_key(x: f32) -> i64 {
    let bits = x.to_bits() as i32 as i64;
    if bits < 0 {
        0x8000_0000_i64 - bits
    } else {
        bits
    }
}

fn ulp_diff(a: f32, b: f32) -> i64 {
    if a.is_nan() && b.is_nan() {
        return 0;
    }
    (ulp_key(a) - ulp_key(b)).abs()
}

fn run_diff_i32(
    name: &str,
    range: i32,
    k8: impl Fn([i32; 8]) -> [i32; 8],
    k1: impl Fn(i32) -> i32,
    scalar: impl Fn(i32) -> i32,
) {
    let mut rng = XorShift::new(0x1234_5678);
    for _ in 0..BATCHES {
        let mut xs = [0i32; 8];
        for x in &mut xs {
            *x = rng.next_i32(range);
        }
        let got8 = k8(xs);
        for (l, &x) in xs.iter().enumerate() {
            let want = scalar(x);
            assert_eq!(got8[l], want, "{name}: N=8 lane {l} (x={x})");
            assert_eq!(k1(x), want, "{name}: N=1 (x={x})");
        }
    }
}

fn run_diff_f32(
    name: &str,
    ulp: i64,
    lo: f32,
    hi: f32,
    k8: impl Fn([f32; 8]) -> [f32; 8],
    k1: impl Fn(f32) -> f32,
    scalar: impl Fn(f32) -> f32,
) {
    let mut rng = XorShift::new(0x9E37_79B1);
    for _ in 0..BATCHES {
        let mut xs = [0f32; 8];
        for x in &mut xs {
            *x = rng.next_f32(lo, hi);
        }
        let got8 = k8(xs);
        for (l, &x) in xs.iter().enumerate() {
            let want = scalar(x);
            assert!(
                ulp_diff(got8[l], want) <= ulp,
                "{name}: N=8 lane {l} (x={x}): got {} want {want} ({} ulp)",
                got8[l],
                ulp_diff(got8[l], want)
            );
            let g1 = k1(x);
            assert!(
                ulp_diff(g1, want) <= ulp,
                "{name}: N=1 (x={x}): got {g1} want {want} ({} ulp)",
                ulp_diff(g1, want)
            );
        }
    }
}

fn run_scatter_diff(
    name: &str,
    range: i32,
    k8: impl Fn([i32; 8], &mut [i32; 8]),
    k1: impl Fn(i32, &mut [i32; 1]),
    scalar: impl Fn(i32) -> i32,
) {
    let mut rng = XorShift::new(0x0BAD_F00D);
    for _ in 0..BATCHES {
        let mut xs = [0i32; 8];
        for x in &mut xs {
            *x = rng.next_i32(range);
        }
        let mut out8 = [i32::MIN; 8];
        k8(xs, &mut out8);
        for (l, &x) in xs.iter().enumerate() {
            let want = scalar(x);
            assert_eq!(out8[l], want, "{name}: N=8 scatter lane {l} (x={x})");
            let mut out1 = [i32::MIN; 1];
            k1(x, &mut out1);
            assert_eq!(out1[0], want, "{name}: N=1 scatter (x={x})");
        }
    }
}

macro_rules! assert_kernel_matches_scalar {
    (int, range = $r:expr, $kern:ident, $scalar:expr $(,)?) => {
        run_diff_i32(
            stringify!($kern),
            $r,
            |xs| $kern::<8, _>(AllOn, Varying::from_array(xs)).to_array(),
            |x| $kern::<1, _>(AllOn, Varying::from_array([x])).to_array()[0],
            $scalar,
        )
    };
    (int, range = $r:expr, tbl = $tbl:expr, $kern:ident, $scalar:expr $(,)?) => {{
        let tbl = $tbl;
        run_diff_i32(
            stringify!($kern),
            $r,
            |xs| $kern::<8, _>(AllOn, &tbl, Varying::from_array(xs)).to_array(),
            |x| $kern::<1, _>(AllOn, &tbl, Varying::from_array([x])).to_array()[0],
            $scalar,
        )
    }};
    (scatter, range = $r:expr, $kern:ident, $scalar:expr $(,)?) => {
        run_scatter_diff(
            stringify!($kern),
            $r,
            |xs, out| $kern::<8, _>(AllOn, Varying::from_array(xs), out),
            |x, out| $kern::<1, _>(AllOn, Varying::from_array([x]), out),
            $scalar,
        )
    };
    (float, ulp = $u:expr, range = ($lo:expr, $hi:expr), $kern:ident, $scalar:expr $(,)?) => {
        run_diff_f32(
            stringify!($kern),
            $u,
            $lo,
            $hi,
            |xs| $kern::<8, _>(AllOn, Varying::from_array(xs)).to_array(),
            |x| $kern::<1, _>(AllOn, Varying::from_array([x])).to_array()[0],
            $scalar,
        )
    };
}

#[kernel]
fn k_if_accum(x: Varying<i32>) -> Varying<i32> {
    let mut acc = Varying::splat(0);
    if x > 0 {
        acc += x * 2;
    }
    acc
}

#[test]
fn d1_if_accum() {
    assert_kernel_matches_scalar!(int, range = 25, k_if_accum, |x| {
        let mut acc = 0;
        if x > 0 {
            acc += x * 2;
        }
        acc
    });
}

#[kernel]
fn k_ifelse_accum(x: Varying<i32>) -> Varying<i32> {
    let mut acc = Varying::splat(0);
    if x > 3 {
        acc = x;
    } else {
        acc = -x;
    }
    acc
}

#[test]
fn d1_ifelse_accum() {
    assert_kernel_matches_scalar!(int, range = 25, k_ifelse_accum, |x| {
        if x > 3 {
            x
        } else {
            -x
        }
    });
}

#[kernel]
fn k_while_none(x: Varying<i32>) -> Varying<i32> {
    let mut i = Varying::splat(0);
    let mut acc = Varying::splat(0);
    while i < x {
        acc += i;
        i += 1;
    }
    acc
}

#[test]
fn d1_while_none() {
    assert_kernel_matches_scalar!(int, range = 25, k_while_none, |x| {
        let mut i = 0;
        let mut acc = 0;
        while i < x {
            acc += i;
            i += 1;
        }
        acc
    });
}

#[kernel]
fn k_while_break(x: Varying<i32>) -> Varying<i32> {
    let mut i = Varying::splat(0);
    let mut acc = Varying::splat(0);
    while i < x {
        acc += i;
        if acc > 15 {
            break;
        }
        i += 1;
    }
    acc
}

#[test]
fn d1_while_break() {
    assert_kernel_matches_scalar!(int, range = 25, k_while_break, |x| {
        let mut i = 0;
        let mut acc = 0;
        while i < x {
            acc += i;
            if acc > 15 {
                break;
            }
            i += 1;
        }
        acc
    });
}

#[kernel]
fn k_while_continue(x: Varying<i32>) -> Varying<i32> {
    let mut i = Varying::splat(0);
    let mut acc = Varying::splat(0);
    while i < x {
        i += 1;
        if i % 3 == 0 {
            continue;
        }
        acc += i;
    }
    acc
}

#[test]
fn d1_while_continue() {
    assert_kernel_matches_scalar!(int, range = 25, k_while_continue, |x| {
        let mut i = 0;
        let mut acc = 0;
        while i < x {
            i += 1;
            if i % 3 == 0 {
                continue;
            }
            acc += i;
        }
        acc
    });
}

#[kernel]
fn k_for_none(x: Varying<i32>) -> Varying<i32> {
    let mut acc = Varying::splat(0);
    for i in 0..6 {
        acc += x + i;
    }
    acc
}

#[test]
fn d1_for_none() {
    assert_kernel_matches_scalar!(int, range = 25, k_for_none, |x| {
        let mut acc = 0;
        for i in 0..6 {
            acc += x + i;
        }
        acc
    });
}

#[kernel]
fn k_for_continue(x: Varying<i32>) -> Varying<i32> {
    let mut acc = Varying::splat(0);
    for i in 0..8 {
        if (x + i) % 2 == 0 {
            continue;
        }
        acc += 1;
    }
    acc
}

#[test]
fn d1_for_continue() {
    assert_kernel_matches_scalar!(int, range = 25, k_for_continue, |x| {
        let mut acc = 0;
        for i in 0..8 {
            if (x + i) % 2 == 0 {
                continue;
            }
            acc += 1;
        }
        acc
    });
}

#[kernel]
fn k_loop_break(x: Varying<i32>) -> Varying<i32> {
    let mut c = Varying::splat(0);
    loop {
        c += 1;
        if c >= x {
            break;
        }
    }
    c
}

#[test]
fn d1_loop_break() {
    assert_kernel_matches_scalar!(int, range = 25, k_loop_break, |x| {
        let mut c = 0;
        loop {
            c += 1;
            if c >= x {
                break;
            }
        }
        c
    });
}

#[kernel]
fn k_early_return_top(x: Varying<i32>) -> Varying<i32> {
    if x < 0 {
        return Varying::splat(-1);
    }
    let acc = x + 1;
    acc
}

#[test]
fn d1_early_return_top() {
    assert_kernel_matches_scalar!(int, range = 25, k_early_return_top, |x| {
        if x < 0 {
            return -1;
        }
        x + 1
    });
}

#[kernel]
fn k_d2_ifif(x: Varying<i32>) -> Varying<i32> {
    let mut acc = Varying::splat(0);
    if x > 0 {
        if x > 10 {
            acc = 2;
        } else {
            acc = 1;
        }
    }
    acc
}

#[test]
fn d2_ifif() {
    assert_kernel_matches_scalar!(int, range = 25, k_d2_ifif, |x| {
        let mut acc = 0;
        if x > 0 {
            if x > 10 {
                acc = 2;
            } else {
                acc = 1;
            }
        }
        acc
    });
}

#[kernel]
fn k_d2_for_break(x: Varying<i32>) -> Varying<i32> {
    let mut acc = Varying::splat(0);
    for i in 0..6 {
        acc += x + i;
        if acc > 15 {
            break;
        }
        acc += 1;
    }
    acc
}

#[test]
fn d2_for_break_stmt_after() {
    assert_kernel_matches_scalar!(int, range = 25, k_d2_for_break, |x| {
        let mut acc = 0;
        for i in 0..6 {
            acc += x + i;
            if acc > 15 {
                break;
            }
            acc += 1;
        }
        acc
    });
}

#[kernel]
fn k_d2_for_return(x: Varying<i32>) -> Varying<i32> {
    let mut r = Varying::splat(0);
    for i in 0..6 {
        r += x + i;
        if r > 20 {
            return r * 2;
        }
    }
    r
}

#[test]
fn d2_for_return() {
    assert_kernel_matches_scalar!(int, range = 25, k_d2_for_return, |x| {
        let mut r = 0;
        for i in 0..6 {
            r += x + i;
            if r > 20 {
                return r * 2;
            }
        }
        r
    });
}

#[kernel]
fn k_d3_while_for_if(x: Varying<i32>) -> Varying<i32> {
    let mut acc = Varying::splat(0);
    let mut o = Varying::splat(0);
    while o < 3 {
        o += 1;
        for i in 0..4 {
            if (x + i) % 2 == 0 {
                continue;
            }
            acc += x + i;
            if acc > 40 {
                break;
            }
        }
        acc += o;
    }
    acc
}

#[test]
fn d3_while_for_if() {
    assert_kernel_matches_scalar!(int, range = 25, k_d3_while_for_if, |x| {
        let mut acc = 0;
        let mut o = 0;
        while o < 3 {
            o += 1;
            for i in 0..4 {
                if (x + i) % 2 == 0 {
                    continue;
                }
                acc += x + i;
                if acc > 40 {
                    break;
                }
            }
            acc += o;
        }
        acc
    });
}

#[kernel]
fn k_d3_if_while_return(x: Varying<i32>) -> Varying<i32> {
    let mut acc = Varying::splat(0);
    if x > 0 {
        let mut i = Varying::splat(0);
        while i < x {
            i += 1;
            if i > 5 {
                return Varying::splat(100);
            }
            acc += i;
        }
    }
    acc
}

#[test]
fn d3_if_while_return() {
    assert_kernel_matches_scalar!(int, range = 25, k_d3_if_while_return, |x| {
        let mut acc = 0;
        if x > 0 {
            let mut i = 0;
            while i < x {
                i += 1;
                if i > 5 {
                    return 100;
                }
                acc += i;
            }
        }
        acc
    });
}

#[kernel]
fn k_gather(tbl: &[i32], x: Varying<i32>) -> Varying<i32> {
    let j = x & 15;
    let mut acc = tbl[j];
    if x > 0 {
        acc += tbl[x & 7];
    }
    acc
}

#[test]
fn mem_gather() {
    let table: [i32; 16] = std::array::from_fn(|i| i as i32 * 3 - 5);
    assert_kernel_matches_scalar!(int, range = 25, tbl = table, k_gather, |x| {
        let mut acc = table[(x & 15) as usize];
        if x > 0 {
            acc += table[(x & 7) as usize];
        }
        acc
    });
}

#[kernel]
fn k_scatter(x: Varying<i32>, out: &mut [i32]) {
    let idx = reduce::lanes_iota::<N>();
    if x > 3 {
        out[idx] = x * 2;
    } else {
        out[idx] = x - 1;
    }
}

#[test]
fn mem_scatter() {
    assert_kernel_matches_scalar!(scatter, range = 25, k_scatter, |x| {
        if x > 3 {
            x * 2
        } else {
            x - 1
        }
    });
}

#[kernel]
fn k_float_ifelse(x: Varying<f32>) -> Varying<f32> {
    let mut acc = Varying::splat(0.0f32);
    if x > 0.0 {
        acc = x * 2.0 + 1.0;
    } else {
        acc = x - 0.5;
    }
    acc
}

#[test]
fn float_ifelse() {
    assert_kernel_matches_scalar!(float, ulp = 1, range = (-8.0, 8.0), k_float_ifelse, |x| {
        if x > 0.0 {
            x * 2.0 + 1.0
        } else {
            x - 0.5
        }
    });
}

#[kernel]
fn k_float_while(x: Varying<f32>) -> Varying<f32> {
    let mut acc = Varying::splat(0.0f32);
    let mut i = Varying::splat(0.0f32);
    while i < x {
        acc += i * 0.5;
        i += 1.0;
    }
    acc
}

#[test]
fn float_while() {
    assert_kernel_matches_scalar!(float, ulp = 1, range = (0.0, 8.0), k_float_while, |x| {
        let mut acc = 0.0f32;
        let mut i = 0.0f32;
        while i < x {
            acc += i * 0.5;
            i += 1.0;
        }
        acc
    });
}

#[kernel]
fn k_adv_stmt_after_break(x: Varying<i32>) -> Varying<i32> {
    let mut i = Varying::splat(0);
    let mut acc = Varying::splat(0);
    while i < x {
        i += 1;
        if x > 6 {
            if i > 2 {
                acc += 100;
                break;
                acc += 5;
            }
            acc += 7;
        }
        acc += 1;
    }
    acc
}

#[test]
fn adv_stmt_after_break() {
    assert_kernel_matches_scalar!(int, range = 25, k_adv_stmt_after_break, |x| {
        let mut i = 0;
        let mut acc = 0;
        while i < x {
            i += 1;
            if x > 6 {
                if i > 2 {
                    acc += 100;
                    break;
                }
                acc += 7;
            }
            acc += 1;
        }
        acc
    });
}

#[kernel]
fn k_adv_outer_continue(x: Varying<i32>) -> Varying<i32> {
    let mut acc = Varying::splat(0);
    let mut o = Varying::splat(0);
    while o < 4 {
        o += 1;
        let mut j = Varying::splat(0);
        while j < x {
            j += 1;
            if j > 3 {
                break;
            }
            acc += 1;
        }
        if (o + x) % 2 == 0 {
            continue;
        }
        acc += 10;
    }
    acc
}

#[test]
fn adv_outer_continue() {
    assert_kernel_matches_scalar!(int, range = 25, k_adv_outer_continue, |x| {
        let mut acc = 0;
        let mut o = 0;
        while o < 4 {
            o += 1;
            let mut j = 0;
            while j < x {
                j += 1;
                if j > 3 {
                    break;
                }
                acc += 1;
            }
            if (o + x) % 2 == 0 {
                continue;
            }
            acc += 10;
        }
        acc
    });
}

#[kernel]
fn k_adv_triple_return(x: Varying<i32>) -> Varying<i32> {
    let mut acc = Varying::splat(0);
    let mut o = Varying::splat(0);
    while o < 3 {
        o += 1;
        let mut j = Varying::splat(0);
        while j < 3 {
            j += 1;
            if (x + o * 3 + j) % 7 == 0 {
                if x > 5 {
                    return acc - 99;
                }
                acc += 2;
            }
            acc += j;
        }
        acc += o * 10;
    }
    acc + 1
}

#[test]
fn adv_triple_return() {
    assert_kernel_matches_scalar!(int, range = 25, k_adv_triple_return, |x| {
        let mut acc = 0;
        let mut o = 0;
        while o < 3 {
            o += 1;
            let mut j = 0;
            while j < 3 {
                j += 1;
                if (x + o * 3 + j) % 7 == 0 {
                    if x > 5 {
                        return acc - 99;
                    }
                    acc += 2;
                }
                acc += j;
            }
            acc += o * 10;
        }
        acc + 1
    });
}

#[kernel]
fn k_adv_shadow(x: Varying<i32>) -> Varying<i32> {
    let mut acc = Varying::splat(1);
    let t = x + 1;
    if x > 0 {
        let t = x * 2;
        acc += t;
        if t > 10 {
            let t = t + 5;
            acc += t;
        }
    }
    acc += t;
    acc
}

#[test]
fn adv_shadow() {
    assert_kernel_matches_scalar!(int, range = 25, k_adv_shadow, |x| {
        let mut acc = 1;
        let t = x + 1;
        if x > 0 {
            let t = x * 2;
            acc += t;
            if t > 10 {
                let t = t + 5;
                acc += t;
            }
        }
        acc += t;
        acc
    });
}

#[kernel]
fn k_adv_break_continue_same(x: Varying<i32>) -> Varying<i32> {
    let mut i = Varying::splat(0);
    let mut acc = Varying::splat(0);
    while i < x {
        i += 1;
        if i % 3 == 0 {
            continue;
        }
        acc += i;
        if acc > 20 {
            acc += 1000;
            break;
        }
        acc += 2;
    }
    acc
}

#[test]
fn adv_break_continue_same() {
    assert_kernel_matches_scalar!(int, range = 25, k_adv_break_continue_same, |x| {
        let mut i = 0;
        let mut acc = 0;
        while i < x {
            i += 1;
            if i % 3 == 0 {
                continue;
            }
            acc += i;
            if acc > 20 {
                acc += 1000;
                break;
            }
            acc += 2;
        }
        acc
    });
}

#[kernel]
fn k_adv_cif_break(x: Varying<i32>) -> Varying<i32> {
    let mut i = Varying::splat(0);
    let mut acc = Varying::splat(0);
    while i < x {
        i += 1;
        cif!(acc > 12 => {
            break;
        } else {
            acc += i;
        });
        acc += 1;
    }
    acc
}

#[test]
fn adv_cif_break() {
    assert_kernel_matches_scalar!(int, range = 25, k_adv_cif_break, |x| {
        let mut i = 0;
        let mut acc = 0;
        while i < x {
            i += 1;
            if acc > 12 {
                break;
            } else {
                acc += i;
            }
            acc += 1;
        }
        acc
    });
}

#[kernel]
fn k_adv_vmaskguard_break(x: Varying<i32>, flag: i32) -> Varying<i32> {
    let mut i = Varying::splat(0);
    let mut acc = Varying::splat(0);
    while i < x {
        i += 1;
        if x > 4 {
            if flag > 0 {
                break;
            }
        }
        acc += i;
    }
    acc
}

#[test]
fn adv_vmaskguard_break() {
    for flag in [0, 1] {
        run_diff_i32(
            "k_adv_vmaskguard_break",
            25,
            |xs| k_adv_vmaskguard_break::<8, _>(AllOn, Varying::from_array(xs), flag).to_array(),
            |x| k_adv_vmaskguard_break::<1, _>(AllOn, Varying::from_array([x]), flag).to_array()[0],
            |x| {
                let mut i = 0;
                let mut acc = 0;
                while i < x {
                    i += 1;
                    if x > 4 {
                        if flag > 0 {
                            break;
                        }
                    }
                    acc += i;
                }
                acc
            },
        );
    }
}

#[kernel]
fn k_adv_foreach_short(a: &[i32], out: &mut [i32]) {
    foreach!(i in 0..a.len() {
        let v = a[i];
        if v > 0 {
            out[i] = v * 3 + 1;
        } else {
            out[i] = v - 1;
        }
    })
}

#[test]
fn adv_foreach_short_tail() {
    for &n in &[0usize, 1, 3, 5, 7, 8, 9, 13] {
        let mut rng = XorShift::new(0xC0FF_EE00 ^ n as u32);
        let a: Vec<i32> = (0..n).map(|_| rng.next_i32(50)).collect();
        let want: Vec<i32> = a
            .iter()
            .map(|&v| if v > 0 { v * 3 + 1 } else { v - 1 })
            .collect();
        let mut out8 = vec![i32::MIN; n];
        k_adv_foreach_short::<8, _>(AllOn, &a, &mut out8);
        assert_eq!(out8, want, "N=8, n={n}");
        let mut out1 = vec![i32::MIN; n];
        k_adv_foreach_short::<1, _>(AllOn, &a, &mut out1);
        assert_eq!(out1, want, "N=1, n={n}");
    }
}

#[kernel]
fn k_adv_for_break_in_while_continue(x: Varying<i32>) -> Varying<i32> {
    let mut o = Varying::splat(0);
    let mut acc = Varying::splat(0);
    while o < 4 {
        o += 1;
        for _i in 0..5 {
            acc += 1;
            if acc + x > 18 {
                break;
            }
        }
        if (o + x) % 3 == 0 {
            continue;
        }
        acc += 100;
    }
    acc
}

#[test]
fn adv_for_break_in_while_continue() {
    assert_kernel_matches_scalar!(int, range = 25, k_adv_for_break_in_while_continue, |x| {
        let mut o = 0;
        let mut acc = 0;
        while o < 4 {
            o += 1;
            for _i in 0..5 {
                acc += 1;
                if acc + x > 18 {
                    break;
                }
            }
            if (o + x) % 3 == 0 {
                continue;
            }
            acc += 100;
        }
        acc
    });
}

#[kernel]
fn k_adv_gather_cond(tbl: &[i32], x: Varying<i32>) -> Varying<i32> {
    let mut j = x & 7;
    let mut acc = Varying::splat(0);
    while tbl[j] > 0 {
        acc += tbl[j];
        j += 1;
        if j > 12 {
            break;
        }
    }
    acc
}

#[test]
fn adv_gather_in_while_cond() {
    let table: [i32; 16] =
        std::array::from_fn(|i| if i % 5 == 0 { -1 } else { (i as i32 % 4) + 1 });
    assert_kernel_matches_scalar!(int, range = 25, tbl = table, k_adv_gather_cond, |x| {
        let mut j = x & 7;
        let mut acc = 0;
        while table[j as usize] > 0 {
            acc += table[j as usize];
            j += 1;
            if j > 12 {
                break;
            }
        }
        acc
    });
}

#[kernel]
fn k_adv_return_in_while_continue(x: Varying<i32>) -> Varying<i32> {
    let mut i = Varying::splat(0);
    let mut acc = Varying::splat(0);
    while i < x {
        i += 1;
        if (x + i) % 4 == 0 {
            continue;
        }
        if acc > 12 {
            return acc * 10;
        }
        acc += i;
    }
    acc
}

#[test]
fn adv_return_in_while_continue() {
    assert_kernel_matches_scalar!(int, range = 25, k_adv_return_in_while_continue, |x| {
        let mut i = 0;
        let mut acc = 0;
        while i < x {
            i += 1;
            if (x + i) % 4 == 0 {
                continue;
            }
            if acc > 12 {
                return acc * 10;
            }
            acc += i;
        }
        acc
    });
}

#[kernel]
fn k_adv_scatter_overlap(x: Varying<i32>, out: &mut [i32]) {
    let idx = x & 3;
    if x % 2 == 0 {
        out[idx] = x * 10 + 1;
    }
}

#[test]
fn adv_scatter_overlap_partial_mask() {
    let mut rng = XorShift::new(0xDEAD_BEE5);
    for _ in 0..BATCHES {
        let mut xs = [0i32; 8];
        for x in &mut xs {
            *x = rng.next_i32(25);
        }
        let mut o8 = [7i32; 4];
        k_adv_scatter_overlap::<8, _>(AllOn, Varying::from_array(xs), &mut o8);
        let mut o1 = [7i32; 4];
        for &x in &xs {
            k_adv_scatter_overlap::<1, _>(AllOn, Varying::from_array([x]), &mut o1);
        }
        let mut os = [7i32; 4];
        for &x in &xs {
            if x % 2 == 0 {
                os[(x & 3) as usize] = x * 10 + 1;
            }
        }
        assert_eq!(o8, os, "N=8 scatter order vs scalar (xs={xs:?})");
        assert_eq!(o1, os, "N=1 sequential vs scalar (xs={xs:?})");
    }
}
