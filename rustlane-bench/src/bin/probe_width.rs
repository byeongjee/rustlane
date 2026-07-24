#![feature(portable_simd)]

use rustlane::export;
use rustlane::prelude::*;

#[export]
fn dispatched_width(out: &mut [i32]) {
    foreach!(i in 0..out.len() {
        out[i] = Varying::splat(N as i32);
    });
}

#[export(targets("avx2", "neon"))]
fn dispatched_width_avx2(out: &mut [i32]) {
    foreach!(i in 0..out.len() {
        out[i] = Varying::splat(N as i32);
    });
}

fn main() {
    let mut a = [0i32; 64];
    let mut b = [0i32; 64];
    dispatched_width(&mut a);
    dispatched_width_avx2(&mut b);
    println!("default_export_width={}", a[0]);
    println!("avx2_pinned_width={}", b[0]);
    println!("NATIVE_LANES={}", NATIVE_LANES);
}
