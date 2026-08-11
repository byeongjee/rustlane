use rustlane::kernel;
use rustlane::prelude::*;

const W: usize = 8;

fn check_i32(
    name: &str,
    xs: [i32; W],
    k8: impl Fn(Varying<i32, 8>) -> Varying<i32, 8>,
    k1: impl Fn(Varying<i32, 1>) -> Varying<i32, 1>,
    reference: impl Fn(i32) -> i32,
) {
    let got8 = k8(Varying::from_array(xs)).to_array();
    for (l, &x) in xs.iter().enumerate() {
        let want = reference(x);
        assert_eq!(got8[l], want, "{name}: N=8 lane {l} (x={x})");
        let got1 = k1(Varying::from_array([x])).to_array()[0];
        assert_eq!(got1, want, "{name}: N=1 (x={x})");
    }
}

#[kernel]
fn k_nested_if(x: Varying<i32>) -> Varying<i32> {
    let mut r = Varying::splat(0);
    if x > 10 {
        if x > 20 {
            r = 3;
        } else {
            r = 2;
        }
    } else {
        r = 1;
        if x == 5 {
            r = 5;
        }
    }
    r
}

#[test]
fn nested_if_else() {
    check_i32(
        "nested_if",
        [-3, 0, 5, 10, 11, 20, 21, 42],
        |v| k_nested_if::<8, _>(AllOn, v),
        |v| k_nested_if::<1, _>(AllOn, v),
        |x| {
            if x > 10 {
                if x > 20 {
                    3
                } else {
                    2
                }
            } else if x == 5 {
                5
            } else {
                1
            }
        },
    );
}

#[kernel]
fn k_while_break(n: Varying<i32>) -> Varying<i32> {
    let mut i = Varying::splat(0);
    let mut acc = Varying::splat(0);
    while i < n {
        acc += i;
        if acc > 10 {
            break;
        }
        i += 1;
    }
    acc
}

fn ref_while_break(n: i32) -> i32 {
    let mut i = 0;
    let mut acc = 0;
    while i < n {
        acc += i;
        if acc > 10 {
            break;
        }
        i += 1;
    }
    acc
}

#[test]
fn while_with_break() {
    check_i32(
        "while_break",
        [0, 1, 2, 3, 5, 7, 10, 100],
        |v| k_while_break::<8, _>(AllOn, v),
        |v| k_while_break::<1, _>(AllOn, v),
        ref_while_break,
    );
}

#[kernel]
fn k_for_continue(x: Varying<i32>) -> Varying<i32> {
    let mut acc = Varying::splat(0);
    for i in 0..10 {
        if (x + i) % 2 == 0 {
            continue;
        }
        acc += 1;
    }
    acc
}

#[test]
fn for_with_continue() {
    check_i32(
        "for_continue",
        [-4, -1, 0, 1, 2, 3, 6, 9],
        |v| k_for_continue::<8, _>(AllOn, v),
        |v| k_for_continue::<1, _>(AllOn, v),
        |x| {
            let mut acc = 0;
            for i in 0..10 {
                if (x + i) % 2 == 0 {
                    continue;
                }
                acc += 1;
            }
            acc
        },
    );
}

#[kernel]
fn k_while_continue(n: Varying<i32>) -> Varying<i32> {
    let mut i = Varying::splat(0);
    let mut acc = Varying::splat(0);
    while i < n {
        i += 1;
        if i % 3 == 0 {
            continue;
        }
        acc += i;
    }
    acc
}

fn ref_while_continue(n: i32) -> i32 {
    let mut i = 0;
    let mut acc = 0;
    while i < n {
        i += 1;
        if i % 3 == 0 {
            continue;
        }
        acc += i;
    }
    acc
}

#[test]
fn while_with_continue() {
    check_i32(
        "while_continue",
        [0, 1, 2, 3, 4, 6, 9, 13],
        |v| k_while_continue::<8, _>(AllOn, v),
        |v| k_while_continue::<1, _>(AllOn, v),
        ref_while_continue,
    );
}

#[kernel]
fn k_early_return(x: Varying<i32>) -> Varying<i32> {
    if x < 0 {
        return Varying::splat(-1);
    }
    let mut r = Varying::splat(0);
    for i in 0..4 {
        r += x + i;
        if r > 50 {
            return r * 10;
        }
    }
    r
}

fn ref_early_return(x: i32) -> i32 {
    if x < 0 {
        return -1;
    }
    let mut r = 0;
    for i in 0..4 {
        r += x + i;
        if r > 50 {
            return r * 10;
        }
    }
    r
}

#[test]
fn early_return() {
    check_i32(
        "early_return",
        [-7, -1, 0, 1, 5, 13, 20, 100],
        |v| k_early_return::<8, _>(AllOn, v),
        |v| k_early_return::<1, _>(AllOn, v),
        ref_early_return,
    );
}

#[kernel]
fn k_unmasked(x: Varying<i32>) -> Varying<i32> {
    let mut r = Varying::splat(0);
    let mut all = Varying::splat(0);
    if x > 0 {
        r = 1;
        unmasked! {
            all = 7;
        }
    }
    r + all
}

#[test]
fn unmasked_semantics() {
    check_i32(
        "unmasked",
        [-3, -1, 0, 1, 2, 5, -9, 8],
        |v| k_unmasked::<8, _>(AllOn, v),
        |v| k_unmasked::<1, _>(AllOn, v),
        |x| if x > 0 { 1 + 7 } else { 7 },
    );
}

#[kernel]
fn k_scale(a: &[f32], out: &mut [f32]) {
    foreach!(i in 0..a.len() {
        let x = a[i];
        if x > 3.0 {
            out[i] = x * 2.0;
        } else {
            out[i] = x + 1.0;
        }
    })
}

#[test]
fn foreach_with_tail() {
    let n = 13;
    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.7).collect();
    let want: Vec<f32> = a
        .iter()
        .map(|&x| if x > 3.0 { x * 2.0 } else { x + 1.0 })
        .collect();

    let mut out8 = vec![0.0f32; n];
    k_scale::<8, _>(AllOn, &a, &mut out8);
    assert_eq!(out8, want, "foreach N=8");

    let mut out1 = vec![0.0f32; n];
    k_scale::<1, _>(AllOn, &a, &mut out1);
    assert_eq!(out1, want, "foreach N=1");
}

#[kernel]
fn k_grid(w: usize, a: &[f32], out: &mut [f32]) {
    foreach_2d!(y in 0..3, x in 0..w {
        let i = x + (y * w) as i32;
        out[i] = a[i] * 2.0 + y as i32 as f32;
    })
}

#[test]
fn foreach_2d_with_tail() {
    let (h, w) = (3usize, 13usize);
    let a: Vec<f32> = (0..h * w).map(|i| i as f32 * 0.3).collect();
    let want: Vec<f32> = (0..h * w).map(|i| a[i] * 2.0 + (i / w) as f32).collect();

    let mut out8 = vec![0.0f32; h * w];
    k_grid::<8, _>(AllOn, w, &a, &mut out8);
    assert_eq!(out8, want, "foreach_2d N=8");

    let mut out1 = vec![0.0f32; h * w];
    k_grid::<1, _>(AllOn, w, &a, &mut out1);
    assert_eq!(out1, want, "foreach_2d N=1");
}

#[kernel]
fn k_tiled(w: i32, counts: &mut [i32], vals: &mut [i32]) {
    foreach_tiled!(y in 0..5, x in 0..13 {
        let i = y * w + x;
        counts[i] += 1;
        vals[i] = y * 100 + x;
    })
}

#[test]
fn foreach_tiled_covers_each_cell_once() {
    let (h, w) = (5usize, 13usize);
    let want_vals: Vec<i32> = (0..h * w)
        .map(|i| (i / w) as i32 * 100 + (i % w) as i32)
        .collect();

    let mut counts8 = vec![0i32; h * w];
    let mut vals8 = vec![-1i32; h * w];
    k_tiled::<8, _>(AllOn, w as i32, &mut counts8, &mut vals8);
    assert_eq!(
        counts8,
        vec![1i32; h * w],
        "foreach_tiled N=8: visit counts"
    );
    assert_eq!(vals8, want_vals, "foreach_tiled N=8: coordinates");

    let mut counts1 = vec![0i32; h * w];
    let mut vals1 = vec![-1i32; h * w];
    k_tiled::<1, _>(AllOn, w as i32, &mut counts1, &mut vals1);
    assert_eq!(
        counts1,
        vec![1i32; h * w],
        "foreach_tiled N=1: visit counts"
    );
    assert_eq!(vals1, want_vals, "foreach_tiled N=1: coordinates");
}

#[kernel]
fn k_cwhile(n: Varying<i32>) -> Varying<i32> {
    let mut i = Varying::splat(0);
    let mut acc = Varying::splat(0);
    cwhile!(i < n => {
        acc += i * 2;
        i += 1;
    });
    acc
}

#[test]
fn coherent_while() {
    check_i32(
        "cwhile",
        [0, 1, 2, 3, 5, 8, 13, 21],
        |v| k_cwhile::<8, _>(AllOn, v),
        |v| k_cwhile::<1, _>(AllOn, v),
        |n| {
            let mut i = 0;
            let mut acc = 0;
            while i < n {
                acc += i * 2;
                i += 1;
            }
            acc
        },
    );
}

#[kernel]
fn k_gather_scatter(tbl: &[i32], idx: Varying<i32>, out: &mut [i32]) {
    let v = tbl[idx];
    out[idx] = v * 2;
    out[idx] += 1;
}

#[test]
fn gather_scatter() {
    let tbl: Vec<i32> = (0..16).map(|i| 100 + i).collect();
    let idx = [3, 0, 7, 12, 5, 9, 15, 1];

    let mut out8 = vec![0i32; 16];
    k_gather_scatter::<8, _>(AllOn, &tbl, Varying::from_array(idx), &mut out8);

    let mut out1 = vec![0i32; 16];
    for &i in &idx {
        k_gather_scatter::<1, _>(AllOn, &tbl, Varying::from_array([i]), &mut out1);
    }

    let mut want = vec![0i32; 16];
    for &i in &idx {
        want[i as usize] = tbl[i as usize] * 2 + 1;
    }
    assert_eq!(out8, want, "gather/scatter N=8");
    assert_eq!(out1, want, "gather/scatter N=1");
}

#[kernel]
fn k_masked_scatter(sel: Varying<i32>, idx: Varying<i32>, out: &mut [i32]) {
    if sel > 0 {
        out[idx] += 1;
    }
}

#[test]
fn masked_scatter_inactive_lanes() {
    let mut out = vec![0i32; 8];
    let sel = [1, 0, 5, -1, 2, 0, 1, 0];
    let idx = [0, 999, 2, 999, 4, 999, 6, 999];
    k_masked_scatter::<8, _>(
        AllOn,
        Varying::from_array(sel),
        Varying::from_array(idx),
        &mut out,
    );
    assert_eq!(out, [1, 0, 1, 0, 1, 0, 1, 0]);
}

#[kernel]
fn k_cif(x: Varying<i32>) -> Varying<i32> {
    let mut r = Varying::splat(0);
    cif!(x > 0 => {
        r = 1;
    } else {
        r = 2;
    });
    r
}

#[test]
fn coherent_if() {
    check_i32(
        "cif",
        [-3, -2, -1, 0, 1, 2, 3, 4],
        |v| k_cif::<8, _>(AllOn, v),
        |v| k_cif::<1, _>(AllOn, v),
        |x| if x > 0 { 1 } else { 2 },
    );
}

#[kernel]
fn k_loop(n: Varying<i32>) -> Varying<i32> {
    let mut c = Varying::splat(0);
    loop {
        c += 1;
        if c >= n {
            break;
        }
    }
    c
}

#[test]
fn bare_loop_varying_break() {
    check_i32(
        "loop",
        [1, 2, 3, 5, 8, 13, 21, 34],
        |v| k_loop::<8, _>(AllOn, v),
        |v| k_loop::<1, _>(AllOn, v),
        |n| n.max(1),
    );
}

#[kernel]
fn k_logic(x: Varying<i32>, lo: i32) -> Varying<i32> {
    let mut r = Varying::splat(0);
    if x > lo && x < 10 || x == 42 {
        r = 1;
    }
    if !(x < 0) {
        r += 10;
    }
    r
}

#[test]
fn logic_ops() {
    let lo = 2;
    check_i32(
        "logic",
        [-5, 0, 2, 3, 9, 10, 42, 7],
        |v| k_logic::<8, _>(AllOn, v, lo),
        |v| k_logic::<1, _>(AllOn, v, lo),
        |x| {
            let mut r = 0;
            if x > lo && x < 10 || x == 42 {
                r = 1;
            }
            if !(x < 0) {
                r += 10;
            }
            r
        },
    );
}

#[kernel]
fn k_guarded_gather(i: Varying<i32>, a: &[i32], n: i32, j: usize, m: usize) -> Varying<i32> {
    let mut r = Varying::splat(0);
    if i < n && a[i] > 10 {
        r = 1;
    }
    if i >= n || a[i] > 20 {
        r += 2;
    }
    if j < m && a[j] > 10 {
        r += 4;
    }
    r
}

// The rhs of `&&`/`||` must be evaluated under the lhs-narrowed mask: lanes
// (and the uniform path) failing the guard must not perform the `a[..]` reads.
#[test]
fn short_circuit_masks_rhs() {
    let a = [0i32, 15, 25, 5, 30];
    let n = a.len() as i32;
    let iv = Varying::from_array([0, 1, 2, 3, 4, 5, 6, 7]);
    let got = k_guarded_gather::<8, _>(AllOn, iv, &a, n, 9, a.len()).to_array();
    for (l, want) in [0, 1, 3, 0, 3, 2, 2, 2].into_iter().enumerate() {
        assert_eq!(got[l], want, "lane {l}");
    }
}

#[kernel]
fn k_cast(x: Varying<f32>) -> Varying<i32> {
    let mut r = Varying::splat(0);
    if x > 2.0 {
        r = (x * 3.0) as i32;
    }
    r + 5.5 as i32
}

#[test]
fn casts() {
    let xs = [-1.5f32, 0.0, 2.0, 2.5, 3.9, -2.7, 10.1, 2.01];
    let reference = |x: f32| {
        let mut r = 0;
        if x > 2.0 {
            r = (x * 3.0) as i32;
        }
        r + 5.5 as i32
    };
    let got8 = k_cast::<8, _>(AllOn, Varying::from_array(xs)).to_array();
    for (l, &x) in xs.iter().enumerate() {
        assert_eq!(got8[l], reference(x), "cast N=8 lane {l}");
        let got1 = k_cast::<1, _>(AllOn, Varying::from_array([x])).to_array()[0];
        assert_eq!(got1, reference(x), "cast N=1 (x={x})");
    }
}

#[kernel]
fn k_uniform_if(x: Varying<i32>, flag: i32) -> Varying<i32> {
    let mut r = Varying::splat(0);
    if flag > 0 {
        r = x + 1;
    } else {
        r = x - 1;
    }
    if x > 0 {
        if flag > 1 {
            r += 10;
        }
    }
    r
}

#[test]
fn uniform_and_nested_uniform_if() {
    for flag in [-1, 0, 1, 2] {
        check_i32(
            "uniform_if",
            [-3, -1, 0, 1, 2, 5, 7, 11],
            |v| k_uniform_if::<8, _>(AllOn, v, flag),
            |v| k_uniform_if::<1, _>(AllOn, v, flag),
            |x| {
                let mut r = if flag > 0 { x + 1 } else { x - 1 };
                if x > 0 && flag > 1 {
                    r += 10;
                }
                r
            },
        );
    }
}

#[kernel]
fn k_double(x: Varying<i32>) -> Varying<i32> {
    x * 2
}

#[kernel]
fn k_call(x: Varying<i32>) -> Varying<i32> {
    let mut r = Varying::splat(0);
    if x > 0 {
        r = k_double(x) + 1;
    }
    r
}

#[test]
fn kernel_calls() {
    check_i32(
        "kernel_call",
        [-2, -1, 0, 1, 2, 3, 4, 5],
        |v| k_call::<8, _>(AllOn, v),
        |v| k_call::<1, _>(AllOn, v),
        |x| if x > 0 { x * 2 + 1 } else { 0 },
    );
}

struct Ops;

#[kernel]
impl Ops {
    fn shift(x: Varying<i32>, k: i32) -> Varying<i32> {
        let mut r = x;
        if x > 0 {
            r += k;
        }
        r
    }
}

#[test]
fn impl_block_kernel() {
    check_i32(
        "impl_kernel",
        [-2, -1, 0, 1, 2, 3, 4, 5],
        |v| Ops::shift::<8, _>(AllOn, v, 100),
        |v| Ops::shift::<1, _>(AllOn, v, 100),
        |x| if x > 0 { x + 100 } else { x },
    );
}

#[kernel]
fn k_local_array(x: Varying<f32>) -> Varying<f32> {
    let mut v = [Varying::splat(0.0f32); 4];
    for j in 0..4usize {
        v[j] = x * (j as f32);
    }
    let mut acc = Varying::splat(0.0f32);
    for j in 0..4usize {
        acc += v[j];
    }
    acc
}

#[test]
fn local_varying_array() {
    let xs = [0.5f32, 1.0, -2.0, 3.25, 0.0, 7.5, -0.25, 2.0];
    let reference = |x: f32| x * (0.0 + 1.0 + 2.0 + 3.0);
    let got8 = k_local_array::<8, _>(AllOn, Varying::from_array(xs)).to_array();
    for (l, &x) in xs.iter().enumerate() {
        assert_eq!(got8[l], reference(x), "local_array N=8 lane {l}");
        let got1 = k_local_array::<1, _>(AllOn, Varying::from_array([x])).to_array()[0];
        assert_eq!(got1, reference(x), "local_array N=1 (x={x})");
    }
}
