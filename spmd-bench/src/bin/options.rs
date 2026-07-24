#![feature(portable_simd)]

use spmd::prelude::*;
use spmd::{export, kernel};
use std::time::Instant;


#[kernel]
fn cnd(x: Varying<f32>) -> Varying<f32> {
    let l = math::abs(x);

    let k = 1.0 / (1.0 + 0.2316419 * l);
    let k2 = k * k;
    let k3 = k2 * k;
    let k4 = k2 * k2;
    let k5 = k3 * k2;

    let inv_sqrt_2pi = 0.39894228040f32;
    let mut w = 0.31938153 * k - 0.356563782 * k2 + 1.781477937 * k3
        - 1.821255978 * k4 + 1.330274429 * k5;
    w = w * (inv_sqrt_2pi * math::exp(-l * l * 0.5));

    if x > 0.0 {
        w = 1.0 - w;
    }
    w
}

#[kernel]
fn black_scholes(
    s: Varying<f32>,
    x: Varying<f32>,
    t: Varying<f32>,
    r: Varying<f32>,
    v: Varying<f32>,
) -> Varying<f32> {
    let d1 = (math::log(s / x) + (r + v * v * 0.5) * t) / (v * math::sqrt(t));
    let d2 = d1 - v * math::sqrt(t);
    s * cnd(d1) - x * math::exp(-r * t) * cnd(d2)
}

#[kernel]
fn binomial_put(
    s: Varying<f32>,
    x: Varying<f32>,
    t: Varying<f32>,
    r: Varying<f32>,
    v: Varying<f32>,
) -> Varying<f32> {
    let mut vv = [Varying::splat(0.0f32); 64];

    let dt = t / 64.0f32;
    let u = math::exp(v * math::sqrt(dt));
    let d = 1.0 / u;
    let disc = math::exp(r * dt);
    let pu = (disc - d) / (u - d);

    for j in 0..64usize {
        let upow = math::pow(u, Varying::splat((2 * j as i32 - 64) as f32));
        vv[j] = math::max(Varying::splat(0.0f32), x - s * upow);
    }

    let omp = 1.0 - pu;
    for jr in 0..64usize {
        let j = 63 - jr;
        for k in 0..j {
            let nv = math::fma(omp, vv[k], pu * vv[k + 1]) / disc;
            vv[k] = nv;
        }
    }
    vv[0]
}


#[export]
fn bs_entry(
    sa: &[f32],
    xa: &[f32],
    ta: &[f32],
    ra: &[f32],
    va: &[f32],
    result: &mut [f32],
    count: i32,
) {
    let n = count as usize;
    foreach!(i in 0..n {
        let s = sa[i];
        let x = xa[i];
        let t = ta[i];
        let r = ra[i];
        let v = va[i];
        result[i] = black_scholes(s, x, t, r, v);
    });
}

#[export]
fn binomial_entry(
    sa: &[f32],
    xa: &[f32],
    ta: &[f32],
    ra: &[f32],
    va: &[f32],
    result: &mut [f32],
    count: i32,
) {
    let n = count as usize;
    foreach!(i in 0..n {
        let s = sa[i];
        let x = xa[i];
        let t = ta[i];
        let r = ra[i];
        let v = va[i];
        result[i] = binomial_put(s, x, t, r, v);
    });
}


fn load_ref(path: &str, n: usize) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    assert_eq!(bytes.len(), n * 4, "unexpected size for {path}");
    let mut out = vec![0.0f32; n];
    for (i, o) in out.iter_mut().enumerate() {
        *o = f32::from_le_bytes([
            bytes[i * 4],
            bytes[i * 4 + 1],
            bytes[i * 4 + 2],
            bytes[i * 4 + 3],
        ]);
    }
    out
}

fn compare(got: &[f32], want: &[f32], tol: f32) -> (f32, usize) {
    let mut max_rel = 0.0f32;
    let mut mism = 0usize;
    for (&g, &w) in got.iter().zip(want.iter()) {
        let denom = if w.abs() > 1e-6 { w.abs() } else { 1.0 };
        let rel = (g - w).abs() / denom;
        if rel > max_rel {
            max_rel = rel;
        }
        if rel > tol {
            mism += 1;
        }
    }
    (max_rel, mism)
}

fn bench<F: FnMut()>(mut f: F, warmup: usize, reps: usize) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t0 = Instant::now();
        f();
        let dt = t0.elapsed().as_secs_f64() * 1e3;
        if dt < best {
            best = dt;
        }
    }
    best
}

fn main() {
    const N_OPTIONS: usize = 128 * 1024; 
    const WARMUP: usize = 3;
    const REPS: usize = 15;

    let s = vec![100.0f32; N_OPTIONS];
    let x = vec![98.0f32; N_OPTIONS];
    let t = vec![2.0f32; N_OPTIONS];
    let r = vec![0.02f32; N_OPTIONS];
    let v = vec![5.0f32; N_OPTIONS];
    let mut res_bs = vec![0.0f32; N_OPTIONS];
    let mut res_bin = vec![0.0f32; N_OPTIONS];
    let count = N_OPTIONS as i32;

    let best_bs = bench(
        || bs_entry(&s, &x, &t, &r, &v, &mut res_bs, count),
        WARMUP,
        REPS,
    );
    let best_bin = bench(
        || binomial_entry(&s, &x, &t, &r, &v, &mut res_bin, count),
        WARMUP,
        REPS,
    );

    let sum_bs: f64 = res_bs.iter().map(|&a| a as f64).sum();
    let sum_bin: f64 = res_bin.iter().map(|&a| a as f64).sum();

    println!("CHECKSUM {:.6}", sum_bs + sum_bin);
    println!("MS black_scholes {:.4}", best_bs);
    println!("MS binomial_put {:.4}", best_bin);
    println!("SUM bs={:.6} binomial={:.6}", sum_bs, sum_bin);

    let ref_bs = load_ref(
        concat!(env!("CARGO_MANIFEST_DIR"), "/../ispc-ref/ref-out/options_bs.bin"),
        N_OPTIONS,
    );
    let ref_bin = load_ref(
        concat!(env!("CARGO_MANIFEST_DIR"), "/../ispc-ref/ref-out/options_binomial.bin"),
        N_OPTIONS,
    );

    let bs_tol = 1e-4f32; 
    let bin_tol = 1e-4f32; 
    let (bs_max_rel, bs_mism) = compare(&res_bs, &ref_bs, bs_tol);
    let (bin_max_rel, bin_mism) = compare(&res_bin, &ref_bin, bin_tol);

    println!(
        "VALIDATE bs: max_rel={:.3e} mism={}/{} (tol {:.0e})",
        bs_max_rel, bs_mism, N_OPTIONS, bs_tol
    );
    println!(
        "VALIDATE binomial: max_rel={:.3e} mism={}/{} (tol {:.0e})",
        bin_max_rel, bin_mism, N_OPTIONS, bin_tol
    );

    let ok = bs_max_rel <= bs_tol && bin_max_rel <= bin_tol;
    if ok {
        println!("options: OK");
    } else {
        eprintln!("options: VALIDATION FAILED");
        std::process::exit(1);
    }
}
